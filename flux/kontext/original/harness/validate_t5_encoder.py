"""Measure Kontext's full T5 encoder and record its actual FP8 differences."""

import argparse
import hashlib
import importlib
import json
import site
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    assert args.samples >= 3
    model, output = Path(args.model), Path(args.output)
    assert (model / ".verified-complete").exists()
    output.mkdir(parents=True, exist_ok=False)
    site.addsitedir(str(ROOT / "dependencies"))

    import torch
    from transformers import T5EncoderModel, T5TokenizerFast
    from vllm.config.load import LoadConfig
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import LinearBase
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )
    from vllm_omni.diffusion.models.t5_encoder.quantization import prepare_t5_fp8
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    source = json.loads((ROOT / "evidence/flux-source-manifest.json").read_text())
    sources = {}
    for name in [
        "vllm_omni.diffusion.models.t5_encoder.quantization",
        "vllm_omni.diffusion.model_loader.diffusers_loader",
    ]:
        path = Path(importlib.import_module(name).__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == source["files"][path.relative_to(ROOT / "flux").as_posix()]
        sources[name] = digest
    prompts = [
        "Change the teapot to green while keeping the cups and background unchanged.",
        "Add a small flower beside the cups while preserving the scene.",
        "Make the lighting warmer while preserving every object.",
    ]
    tokenizer = T5TokenizerFast.from_pretrained(
        model, subfolder="tokenizer_2", local_files_only=True
    )
    inputs = [
        tokenizer(
            p,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).input_ids.cuda()
        for p in prompts
    ]
    encoder = (
        T5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder_2",
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    report = {
        "source_commit": source["commit"],
        "sources": sources,
        "model_revision": "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
        "gpu": torch.cuda.get_device_name(),
        "scope": "T5 encoder only, 512 tokens; no DiT, VAE, offload, compile or graph. Not whole-request E2E.",
        "attention_mask": "Omitted, matching FluxKontextPipeline._get_t5_prompt_embeds",
        "prompts": prompts,
        "timings": {},
        "comparisons": [],
        "artifacts": {},
    }

    def record():
        (output / "results.json").write_text(json.dumps(report, indent=2))

    def save(name, value):
        path = output / name
        torch.save(value, path)
        report["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record()

    def forward(ids):
        return encoder(ids, output_hidden_states=False)[0]

    def benchmark(label):
        with torch.inference_mode():
            for ids in inputs:
                forward(ids)
            torch.cuda.current_stream().synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for index in range(args.samples):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin = time.perf_counter()
                start.record()
                value = forward(inputs[index % 3])
                end.record()
                end.synchronize()
                samples.append(
                    {
                        "prompt": index % 3,
                        "cuda_ms": start.elapsed_time(end),
                        "wall_ms": (time.perf_counter() - begin) * 1000,
                    }
                )
                del value
        report["timings"][label] = {
            "samples": samples,
            "mean_cuda_ms": statistics.mean(s["cuda_ms"] for s in samples),
            "mean_wall_ms": statistics.mean(s["wall_ms"] for s in samples),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        record()

    record()
    save("input-ids.pt", [ids.cpu() for ids in inputs])
    with torch.inference_mode():
        reference = [forward(ids).cpu() for ids in inputs]
    save("bf16-hidden-states.pt", reference)
    benchmark("bf16")
    config = OmniDiffusionConfig(
        model=str(model),
        dtype=torch.bfloat16,
        quantization_config={"text_encoder_2": {"method": "fp8"}},
    )
    with (
        set_forward_context(
            vllm_config=create_diffusion_vllm_config(torch.device("cuda:0"), config),
            omni_diffusion_config=config,
        ),
        ExitStack() as cleanup,
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{output}/tp1-store",
            backend="gloo",
        )
        cleanup.callback(destroy_distributed_environment)
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        cleanup.callback(destroy_model_parallel)
        replaced = prepare_t5_fp8(encoder, config.quantization_config, "text_encoder_2")
        assert replaced == 6 * encoder.config.num_layers
        DiffusersPipelineLoader(LoadConfig(), config)._process_weights_after_loading(
            encoder, torch.device("cuda:0")
        )
        projections = [
            {
                "name": name,
                "dtype": str(layer.weight.dtype),
                "backend": type(layer.quant_method.fp8_linear).__name__,
            }
            for name, layer in encoder.named_modules()
            if isinstance(layer, LinearBase)
        ]
        report["projections"] = projections
        assert len(projections) == replaced
        assert all(
            p["dtype"] == "torch.float8_e4m3fn"
            and p["backend"] == "CutlassFP8ScaledMMLinearKernel"
            for p in projections
        )
        retained = [
            {"name": name, "dtype": str(layer.weight.dtype)}
            for name, layer in encoder.named_modules()
            if name.endswith(".wo")
        ]
        assert len(retained) == encoder.config.num_layers
        assert all(p["dtype"] == "torch.bfloat16" for p in retained)
        report["retained_ffn_output"] = retained
        with torch.inference_mode():
            actual = [forward(ids).cpu() for ids in inputs]
        save("fp8-hidden-states.pt", actual)
        for value, expected in zip(actual, reference, strict=True):
            assert value.shape == expected.shape and torch.isfinite(value).all()
            delta = value.float() - expected.float()
            report["comparisons"].append(
                {
                    "shape": list(value.shape),
                    "max_abs": delta.abs().max().item(),
                    "normalized_rmse": (
                        delta.square().mean() / expected.float().square().mean()
                    )
                    .sqrt()
                    .item(),
                }
            )
        record()
        benchmark("fp8")
        with (
            torch.inference_mode(),
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile,
        ):
            forward(inputs[0])
            torch.cuda.current_stream().synchronize()
        profile.export_chrome_trace(str(output / "cuda-trace.json"))
    (output / "completed").write_text(
        "Full T5 encoder latency, hidden states and native profile recorded.\n"
    )


if __name__ == "__main__":
    main()
