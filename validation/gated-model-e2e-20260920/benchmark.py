# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Balanced full-model benchmark for FLUX.1 VAE parallelism and SD3.5 HSDP."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def conditioning_image() -> Image.Image:
    image = Image.new("RGB", (512, 512))
    pixels = image.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x // 2, y // 2, (x + y) // 4)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pipeline", choices=("flux", "kontext", "sd3"), required=True)
    parser.add_argument("--feature", action="store_true")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.pipeline == "sd3":
        parallel = (
            DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2)
            if args.feature
            else DiffusionParallelConfig(tensor_parallel_size=2)
        )
    else:
        parallel = DiffusionParallelConfig(
            ulysses_degree=2,
            ulysses_mode="strict",
            vae_patch_parallel_size=2 if args.feature else 1,
        )

    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=parallel,
        vae_use_tiling=True,
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=1800,
        stage_init_timeout=1800,
    )
    ready_seconds = time.perf_counter() - started
    prompt = {
        "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
        "negative_prompt": "blurry, distorted",
        "modalities": ["image"],
    }
    if args.pipeline == "kontext":
        prompt = {
            "prompt": "Turn the ceramic teapot emerald green while preserving the table and lighting.",
            "multi_modal_data": {"img2img": conditioning_image()},
            "modalities": ["img2img"],
        }

    cases = ((512, 512), (1024, 768), (512, 512))
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height) in enumerate(cases):
            for iteration in range(2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=4,
                    guidance_scale=1.0 if args.pipeline != "sd3" else 4.5,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1 and images[0].size == (width, height)
                image_path = args.output / f"{case}-{iteration}.png"
                images[0].save(image_path)
                row = {
                    "pid": os.getpid(),
                    "pipeline": args.pipeline,
                    "feature": args.feature,
                    "source_sha": args.source_sha,
                    "checkpoint_revision": args.checkpoint_revision,
                    "case": case,
                    "width": width,
                    "height": height,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
