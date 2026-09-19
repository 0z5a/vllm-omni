"""Fresh-process full-pipeline comparison with identical CFG topology."""

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
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            cfg_parallel_size=2,
            use_hsdp=args.hsdp,
            hsdp_shard_size=2 if args.hsdp else -1,
        ),
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    prompt = {
        "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
        "negative_prompt": "blurry, distorted",
        "modalities": ["image"],
    }
    with (args.output / "results.jsonl").open("w") as records:
        for width, height in ((512, 512), (1024, 768)):
            for iteration in range(2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=30,
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
                image_path = args.output / f"{width}x{height}-{iteration}.png"
                images[0].save(image_path)
                record = {
                    "pid": os.getpid(),
                    "hsdp": args.hsdp,
                    "cfg_parallel_size": 2,
                    "width": width,
                    "height": height,
                    "iteration": iteration,
                    "warmup": iteration < 2,
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
