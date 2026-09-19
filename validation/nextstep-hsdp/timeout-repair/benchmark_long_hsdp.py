"""Compare TP2 and HSDP2 full NextStep generation; different parallel strategies."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tp2", "hsdp2"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=(
            DiffusionParallelConfig(tensor_parallel_size=2)
            if args.mode == "tp2" else
            DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2)
        ),
        diffusion_model_runner_cls="nextstep_profile_runner.StageProfileRunner" if args.profile else None,
        enable_diffusion_pipeline_profiler=args.profile,
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=7200,
        stage_init_timeout=7200,
    )
    ready_seconds = time.perf_counter() - started
    prompt = {
        "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
        "negative_prompt": "blurry, distorted",
        "modalities": ["image"],
    }
    with (args.output / "results.jsonl").open("w") as records:
        for width, height in (((512, 512),) if args.smoke else ((512, 512), (768, 512), (512, 512))):
            for iteration in range(args.warmups + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=20,
                    guidance_scale=4.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                elapsed = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1
                assert images[0].size == (width, height)
                image_path = args.output / f"{width}x{height}-{records.tell()}-{iteration}.png"
                images[0].save(image_path)
                record = {
                    "pid": os.getpid(),
                    "mode": args.mode,
                    "intrusive_profile": args.profile,
                    "source_sha": args.source_sha,
                    "width": width,
                    "height": height,
                    "iteration": iteration,
                    "warmup": iteration < args.warmups,
                    "seconds": elapsed,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
