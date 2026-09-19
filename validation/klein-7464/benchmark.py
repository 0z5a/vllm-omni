"""Klein 4B existing PR 7464 validation with identical Ulysses topology."""

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
    parser.add_argument("--parallel-vae", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=2,
            ulysses_mode="strict",
            vae_patch_parallel_size=2 if args.parallel_vae else 1,
        ),
        vae_use_tiling=True,
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
        for case, (width, height, batch) in enumerate(((512, 512, 1), (1152, 1024, 1), (768, 512, 2), (512, 512, 1))):
            for iteration in range(2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    seed=142,
                    num_outputs_per_prompt=batch,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                elapsed = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == batch
                hashes = []
                for index, image in enumerate(images):
                    assert image.size == (width, height)
                    image_path = args.output / f"{case}-{iteration}-{index}.png"
                    image.save(image_path)
                    hashes.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
                record = {
                    "pid": os.getpid(),
                    "parallel_vae": args.parallel_vae,
                    "ulysses_degree": 2,
                    "ulysses_mode": "strict",
                    "case": case,
                    "batch": batch,
                    "source": "PR7464@159202d6" if args.parallel_vae else "698f716",
                    "checkpoint_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
                    "width": width,
                    "height": height,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": elapsed,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashes,
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
