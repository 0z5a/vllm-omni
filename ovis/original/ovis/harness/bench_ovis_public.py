"""Run a full model in a fresh Omni process without forced worker shutdown."""

import argparse
import hashlib
import json
import os
import site
import time
from pathlib import Path

PROMPTS = [
    "A product photograph of a red teapot beside two blue cups on a wooden table.",
    "A watercolor storefront with a clearly readable OPEN sign and a bicycle outside.",
    "Three yellow rubber ducks floating in a turquoise pool, overhead photograph.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfg-size", type=int, choices=[1, 2], default=2)
    parser.add_argument("--hsdp", choices=["none", "block", "resident"], default="none")
    parser.add_argument("--kind", choices=["ovis"], required=True)
    parser.add_argument("--variant", choices=["bf16", "encoder"], required=True)
    parser.add_argument("--offload", choices=["none"], default="none")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--weight-load-threads", type=int, default=4)
    parser.add_argument(
        "--encoder-graph", choices=["native", "eager"], default="native"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--prompts-file", help="JSON list of three prompts or image-edit instructions"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.kind == "ovis" and args.variant in ("bf16", "encoder")
    assert args.offload == "none" and args.eager
    assert args.hsdp == "none" or args.cfg_size == 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    site.addsitedir(str(Path(__file__).resolve().parents[1] / "dependencies"))

    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # StageDiffusionClient sends the normal shutdown message first. The
    # manager then waits for natural exit; its upstream signal escalation is
    # replaced only in this validation process. Single-GPU workers run inline.
    from harness.natural_worker_shutdown import install

    install()
    quantization = (
        {"text_encoder": {"method": "fp8"}} if args.variant == "encoder" else None
    )
    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "num_gpus": args.cfg_size,
        "num_weight_load_threads": args.weight_load_threads,
        "distributed_executor_backend": "uni" if args.cfg_size == 1 else "mp",
        "parallel_config": {
            "cfg_parallel_size": args.cfg_size,
            "use_hsdp": args.hsdp != "none",
            "hsdp_shard_size": 2 if args.hsdp != "none" else -1,
        },
        "additional_config": {"hsdp_reshard_after_forward": args.hsdp != "resident"},
        "diffusion_attention_config": {"default": {"backend": "TORCH_SDPA"}},
        "enforce_eager": args.eager,
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "quantization_config": quantization,
        "worker_extension_cls": "harness.audit_ovis_worker_public.OvisPublicAudit",
        "enable_diffusion_pipeline_profiler": True,
        "log_stats": False,
        "init_timeout": 600,
        "stage_init_timeout": 600,
    }
    size, steps, guidance, seed = 512, 30, 4.0, 142
    sampling = OmniDiffusionSamplingParams(
        height=args.height or size,
        width=args.width or size,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    prompts = PROMPTS
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())
        assert len(prompts) == 3 and all(
            isinstance(prompt, str) and prompt for prompt in prompts
        )

    def make_prompt(prompt_index):
        prompt = {"prompt": prompts[prompt_index], "modalities": ["image"]}
        prompt["negative_prompt"] = "blurry, distorted"
        return prompt

    manifest = {
        "arguments": vars(args),
        "omni_config": config,
        "pid": os.getpid(),
        "prompts": prompts,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "forced_signals": False,
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    omni = Omni(**config)
    (output / "load.json").write_text(
        json.dumps({"seconds": time.perf_counter() - started})
    )

    def snapshot():
        return omni.engine.collective_rpc(method="pipeline_snapshot", stage_ids=[0])

    try:
        (output / "worker-before.json").write_text(json.dumps(snapshot(), indent=2))
        with (output / "requests.jsonl").open("w") as records:
            for index in range(args.warmups + args.requests):
                prompt_index = (
                    index % len(prompts)
                    if index < args.warmups
                    else (index - args.warmups) % len(prompts)
                )
                prompt = make_prompt(prompt_index)
                started = time.perf_counter()
                result = omni.generate(prompt, sampling_params_list=[sampling])
                elapsed = time.perf_counter() - started
                images = extract_images_from_outputs(result)
                assert len(images) == 1, "Each full request must produce one image"
                path = output / f"request-{index:02d}-prompt-{prompt_index}.png"
                images[0].save(path)
                record = {
                    "index": index,
                    "prompt_index": prompt_index,
                    "warmup": index < args.warmups,
                    "wall_ms": elapsed * 1000,
                    "stage_durations": result[0].stage_durations,
                    "peak_memory_mb": result[0].peak_memory_mb,
                    "image": path.name,
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(json.dumps(record), flush=True)
        (output / "worker-after.json").write_text(json.dumps(snapshot(), indent=2))
        if args.profile:
            before = omni.engine.collective_rpc(
                method="ovis_begin_audit", stage_ids=[0]
            )
            (output / "hsdp-before.json").write_text(json.dumps(before, indent=2))
            omni.engine.collective_rpc(method="audit_start_profile", stage_ids=[0])
            omni.generate(make_prompt(0), sampling_params_list=[sampling])
            omni.engine.collective_rpc(
                method="audit_stop_profile",
                args=(str(output / "cuda-trace.json"),),
                stage_ids=[0],
            )
            after = omni.engine.collective_rpc(method="ovis_end_audit", stage_ids=[0])
            (output / "hsdp-after.json").write_text(json.dumps(after, indent=2))
            assert len(after[0]) == args.cfg_size
            for rank in after[0]:
                assert "gathers" in rank and "before_vae" in rank, rank
                if args.hsdp != "none":
                    counts = [
                        count
                        for name, count in rank["gathers"].items()
                        if "transformer_blocks." in name
                    ]
                    assert counts and all(
                        count == (1 if args.hsdp == "resident" else steps)
                        for count in counts
                    )
            if args.hsdp == "resident":
                cleanup = omni.engine.collective_rpc(
                    method="ovis_exception_cleanup", stage_ids=[0]
                )
                (output / "hsdp-exception.json").write_text(
                    json.dumps(cleanup, indent=2)
                )
                assert all("after_injected_exception" in rank for rank in cleanup[0]), (
                    cleanup
                )
    finally:
        omni.close()
    (output / "completed").write_text(
        "All requests completed; engine shutdown and worker exit returned naturally.\n"
    )


if __name__ == "__main__":
    main()
