"""Validate the full pretrained Klein encoder; timings cover the encoder only."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import site
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = (9, 18, 27)
PROMPTS = [
    "A product photograph of a red teapot beside two blue cups on a wooden table.",
    "A watercolor storefront with a clearly readable OPEN sign and a bicycle outside.",
    "Three yellow rubber ducks floating in a turquoise pool, overhead photograph.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    assert args.samples >= 3
    model = Path(args.model).resolve()
    assert (model / ".verified-complete").exists()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    site.addsitedir(str(ROOT / "dependencies"))

    import torch
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
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
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.flux2_klein.quantization import (
        Flux2KleinTextEncoderGraph,
        prepare_flux2_klein_text_encoder_fp8,
    )
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    assert torch.cuda.is_available()
    source_manifest = json.loads((ROOT / "evidence/flux-source-manifest.json").read_text())
    sources = {}
    for name in [
        "vllm_omni.diffusion.models.flux2_klein.quantization",
        "vllm_omni.diffusion.model_loader.diffusers_loader",
    ]:
        path = Path(importlib.import_module(name).__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == source_manifest["files"][path.relative_to(ROOT / "flux").as_posix()]
        sources[name] = {"path": str(path), "sha256": digest}
    report = {
        "arguments": vars(args),
        "source_commit": source_manifest["commit"],
        "sources": sources,
        "model_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "vllm", "transformers"]},
        "scope": "Full pretrained encoder only; no DiT, VAE, offload or torch.compile; not whole-request E2E.",
        "hidden_state_layers": LAYERS,
        "timings": {},
        "comparisons": {},
        "artifacts": {},
    }

    def record():
        (output / "results.json").write_text(json.dumps(report, indent=2))

    def save(name, value):
        path = output / name
        torch.save(value, path)
        report["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record()

    record()
    encoder = (
        Qwen3ForCausalLM.from_pretrained(
            str(model), subfolder="text_encoder", local_files_only=True, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    report["attention_implementation"] = encoder.config._attn_implementation
    record()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(model), subfolder="tokenizer", local_files_only=True)

    def reject_lm_head(*args, **kwargs):
        raise AssertionError("Intermediate hidden states must not execute the LM head")

    encoder.lm_head.forward = reject_lm_head
    inputs = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs.append((tokens.input_ids.cuda(), tokens.attention_mask.cuda()))
    changed_mask = inputs[0][1].clone()
    positions = changed_mask[0].nonzero().flatten()
    changed_mask[0, positions[-2:]] = 0
    inputs.append((inputs[0][0].clone(), changed_mask))
    save("inputs.pt", [(ids.cpu(), mask.cpu()) for ids, mask in inputs])

    def eager(ids, mask):
        result = encoder.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        return torch.stack([result.hidden_states[index] for index in LAYERS], dim=1)

    def benchmark(label, call):
        with torch.inference_mode():
            for ids, mask in inputs[:3]:
                call(ids, mask)
            torch.cuda.current_stream().synchronize()
            samples = []
            for index in range(args.samples):
                ids, mask = inputs[index % 3]
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                result = call(ids, mask)
                end.record()
                end.synchronize()
                samples.append(
                    {
                        "prompt": index % 3,
                        "cuda_ms": start.elapsed_time(end),
                        "wall_ms": (time.perf_counter() - begin) * 1000,
                    }
                )
                del result
        report["timings"][label] = {
            "samples": samples,
            "mean_cuda_ms": statistics.mean(sample["cuda_ms"] for sample in samples),
            "mean_wall_ms": statistics.mean(sample["wall_ms"] for sample in samples),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        record()

    with torch.inference_mode():
        bf16 = [eager(ids, mask).cpu() for ids, mask in inputs]
    save("bf16-hidden-states.pt", bf16)
    torch.cuda.reset_peak_memory_stats()
    benchmark("bf16_eager", eager)
    config = OmniDiffusionConfig(
        model=str(model), dtype=torch.bfloat16, quantization_config={"text_encoder": {"method": "fp8"}}
    )
    with (
        set_forward_context(
            vllm_config=create_diffusion_vllm_config(torch.device("cuda:0"), config), omni_diffusion_config=config
        ),
        ExitStack() as cleanup,
    ):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{output}/tp1-store", backend="gloo"
        )
        cleanup.callback(destroy_distributed_environment)
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        cleanup.callback(destroy_model_parallel)
        report["tp1_metadata_backend"] = "gloo; no encoder collectives"
        record()
        replaced = prepare_flux2_klein_text_encoder_fp8(encoder, config.quantization_config, torch.device("cuda:0"))
        assert replaced == 252
        DiffusersPipelineLoader(LoadConfig(), config)._process_weights_after_loading(encoder, torch.device("cuda:0"))
        projections = [
            {
                "name": name,
                "method": type(layer.quant_method).__name__,
                "backend": type(layer.quant_method.fp8_linear).__name__,
                "dtype": str(layer.weight.dtype),
                "shape": list(layer.weight.shape),
                "device": str(layer.weight.device),
            }
            for name, layer in encoder.named_modules()
            if isinstance(layer, LinearBase)
        ]
        report["projections"] = projections
        record()
        assert len(projections) == 252
        assert all(item["dtype"] == "torch.float8_e4m3fn" for item in projections)
        assert all(item["method"] == "Fp8PerTensorOnlineLinearMethod" for item in projections)
        assert all(item["backend"] == "CutlassFP8ScaledMMLinearKernel" for item in projections)
        with torch.inference_mode():
            fp8 = [eager(ids, mask).cpu() for ids, mask in inputs]
        save("fp8-eager-hidden-states.pt", fp8)

        def compare(label, actual, expected, exact):
            delta = actual.float() - expected.float()
            report["comparisons"][label] = {
                "shape": list(actual.shape),
                "max_abs": delta.abs().max().item(),
                "rmse": delta.square().mean().sqrt().item(),
                "normalized_rmse": (delta.square().mean() / expected.float().square().mean()).sqrt().item(),
                "equal_elements": (actual == expected).sum().item(),
                "elements": actual.numel(),
            }
            record()
            assert torch.isfinite(actual).all()
            if exact:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        for index in range(4):
            compare(f"fp8_vs_bf16_{index}", fp8[index], bf16[index], exact=False)
        torch.cuda.reset_peak_memory_stats()
        benchmark("fp8_eager", eager)
        graph = Flux2KleinTextEncoderGraph(encoder, LAYERS)
        with torch.inference_mode():
            replayed = [graph(ids, mask).cpu() for ids, mask in inputs]
        save("fp8-graph-hidden-states.pt", replayed)
        report["graph_captured"] = graph.graph is not None
        record()
        assert graph.graph is not None
        captured = graph.graph
        for index in range(4):
            compare(f"graph_vs_eager_{index}", replayed[index], fp8[index], exact=True)
        for name, ids, mask in [
            ("short", inputs[0][0][:, -32:], inputs[0][1][:, -32:]),
            ("batch2", torch.cat([item[0] for item in inputs[:2]]), torch.cat([item[1] for item in inputs[:2]])),
        ]:
            with torch.inference_mode():
                compare(name, graph(ids, mask).cpu(), eager(ids, mask).cpu(), exact=True)
            assert graph.graph is captured
        ready = torch.cuda.Event()
        ready.record()
        streams = [torch.cuda.Stream() for _ in inputs]

        def concurrent(index):
            with torch.cuda.device(0), torch.cuda.stream(streams[index]), torch.inference_mode():
                streams[index].wait_event(ready)
                torch.cuda._sleep(1_000_000)
                return graph(*inputs[index])

        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(concurrent, range(4)))
        for stream in streams:
            stream.synchronize()
        for index, value in enumerate(actual):
            compare(f"concurrent_{index}", value.cpu(), fp8[index], exact=True)
        assert graph.graph is captured
        torch.cuda.reset_peak_memory_stats()
        benchmark("fp8_graph", graph)
        report["lm_head_skipped"] = True
        record()
        (output / "completed").write_text("Full pretrained FP8 encoder checks and encoder-only timings completed.\n")


if __name__ == "__main__":
    main()
