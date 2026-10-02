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
IMAGE_PROMPTS = [
    "Change the teapot to green while keeping the cups and background unchanged.",
    "Add a small flower beside the cups while preserving the scene.",
    "Make the lighting warmer while preserving every object.",
]


def join_naturally(processes, timeout=None):
    for process in processes:
        process.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["klein", "kontext", "ovis"], required=True)
    parser.add_argument("--variant", choices=["bf16", "encoder", "dit", "both"], required=True)
    parser.add_argument("--offload", choices=["none", "cpu", "layerwise"], default="cpu")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--weight-load-threads", type=int, default=4)
    parser.add_argument("--lazy-weight-load", action="store_true")
    parser.add_argument("--encoder-graph", choices=["native", "eager"], default="native")
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--input-image")
    parser.add_argument("--prompts-file", help="JSON list of three prompts or image-edit instructions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.encoder_graph == "eager":
        assert args.kind == "klein" and args.variant in ("encoder", "both")
        assert args.offload == "none" and not args.eager
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    site.addsitedir(str(Path(__file__).resolve().parents[1] / "dependencies"))

    from PIL import Image
    from vllm_omni.diffusion import stage_diffusion_proc
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # StageDiffusionClient sends the normal shutdown message first. The
    # manager then waits for natural exit; its upstream signal escalation is
    # replaced only in this validation process. Single-GPU workers run inline.
    stage_diffusion_proc.shutdown = join_naturally
    encoder = "text_encoder_2" if args.kind == "kontext" else "text_encoder"
    quantization = {}
    if args.variant in ("encoder", "both"):
        quantization[encoder] = {"method": "fp8"}
    if args.variant in ("dit", "both"):
        quantization["transformer"] = {"method": "fp8"}
    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "num_gpus": 1,
        "num_weight_load_threads": args.weight_load_threads,
        "enable_multithread_weight_load": not args.lazy_weight_load,
        "distributed_executor_backend": "uni",
        "enforce_eager": args.eager,
        "enable_cpu_offload": args.offload == "cpu",
        "enable_layerwise_offload": args.offload == "layerwise",
        "quantization_config": quantization or None,
        "worker_extension_cls": "harness.audit_worker.PipelineAudit",
        "enable_diffusion_pipeline_profiler": True,
        "log_stats": False,
        "init_timeout": 600,
        "stage_init_timeout": 600,
    }
    if args.no_pin:
        assert args.offload != "none"
        config["enable_cpu_offload"] = False
        config["enable_layerwise_offload"] = False
        config["diffusion_offload_config"] = {
            "mode": "module" if args.offload == "cpu" else "layer",
            "components": ["dit", "text_encoder"] if args.offload == "cpu" else ["dit"],
            "pin_memory": False,
        }
    settings = {
        "klein": (1024, 4, 0.0, 42),
        "kontext": (512, 28, 2.5, 42),
        "ovis": (512, 30, 4.0, 142),
    }
    size, steps, guidance, seed = settings[args.kind]
    sampling = OmniDiffusionSamplingParams(
        height=args.height or size,
        width=args.width or size,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    image = Image.open(args.input_image).convert("RGB") if args.input_image else None
    prompts = IMAGE_PROMPTS if image is not None else PROMPTS
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())
        assert len(prompts) == 3 and all(isinstance(prompt, str) and prompt for prompt in prompts)

    def make_prompt(prompt_index):
        prompt = {"prompt": prompts[prompt_index], "modalities": ["image"]}
        if args.kind == "ovis":
            prompt["negative_prompt"] = "blurry, distorted"
        if image is not None:
            prompt["multi_modal_data"] = {"image": [image]}
        return prompt

    manifest = {
        "arguments": vars(args),
        "omni_config": config,
        "pid": os.getpid(),
        "prompts": prompts,
        "input_image_sha256": hashlib.sha256(Path(args.input_image).read_bytes()).hexdigest()
        if args.input_image
        else None,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "forced_signals": False,
        "shutdown_policy": "Production offloader shutdown; normal stage message and natural worker join",
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    omni = Omni(**config)
    (output / "load.json").write_text(json.dumps({"seconds": time.perf_counter() - started}))

    def snapshot():
        return omni.engine.collective_rpc(method="pipeline_snapshot", stage_ids=[0])

    try:
        (output / "worker-before.json").write_text(json.dumps(snapshot(), indent=2))
        if args.encoder_graph == "eager":
            control = omni.engine.collective_rpc(method="audit_disable_encoder_graph", stage_ids=[0])
            (output / "encoder-graph-control.json").write_text(json.dumps(control, indent=2))
            after_control = snapshot()
            (output / "worker-after-control.json").write_text(json.dumps(after_control, indent=2))
            assert after_control[0][0]["text_encoder_graph"] == {"eligible": False, "captured": False}, control
        with (output / "requests.jsonl").open("w") as records:
            for index in range(args.warmups + args.requests):
                prompt_index = index % len(prompts) if index < args.warmups else (index - args.warmups) % len(prompts)
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
            omni.engine.collective_rpc(method="audit_start_profile", stage_ids=[0])
            omni.generate(make_prompt(0), sampling_params_list=[sampling])
            omni.engine.collective_rpc(
                method="audit_stop_profile",
                args=(str(output / "cuda-trace.json"),),
                stage_ids=[0],
            )
    finally:
        omni.close()
    (output / "completed").write_text("All requests completed; engine shutdown and worker exit returned naturally.\n")


if __name__ == "__main__":
    main()
