"""Fresh-process full-pipeline comparison with identical Ulysses topology."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=2,
            ulysses_mode="advanced_uaa",
            vae_patch_parallel_size=2,
        ),
        diffusion_model_runner_cls="reference_rng_runner.ReferenceRngRunner",
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
    references = []
    for width, height, color in ((512, 512, "red"), (768, 512, "blue"), (512, 768, "green")):
        image = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(image).ellipse((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=color)
        references.append(image)
    with (args.output / "results.jsonl").open("w") as records:
        for reference_count in (1, 2, 3, 1):
            width, height = 512, 512
            prompt["multi_modal_data"] = {"image": references[:reference_count]}
            prompt["prompt"] = (
                "Arrange the objects from the reference images together on a wooden table, soft daylight."
            )
            for iteration in range(2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=30,
                    guidance_scale=4.0,
                    guidance_scale_2=2.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                elapsed = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1
                assert images[0].size == (width, height)
                image_path = args.output / f"{reference_count}-refs-{records.tell()}-{iteration}.png"
                images[0].save(image_path)
                record = {
                    "pid": os.getpid(),
                    "source_sha": args.source_sha,
                    "vae_parallel_size": 2,
                    "global_posterior_rng_seed": 142,
                    "reference_count": reference_count,
                    "reference_sha256": [
                        hashlib.sha256(image.tobytes()).hexdigest() for image in references[:reference_count]
                    ],
                    "ulysses_degree": 2,
                    "ulysses_mode": "advanced_uaa",
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
