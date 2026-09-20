# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a fresh HiDream full-model admission and two complete requests."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parallel_config(mode: str) -> DiffusionParallelConfig:
    if mode == "cfg2":
        return DiffusionParallelConfig(cfg_parallel_size=2)
    if mode == "hsdp2":
        return DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2)
    if mode == "vae2":
        return DiffusionParallelConfig(vae_patch_parallel_size=2)
    return DiffusionParallelConfig()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--mode", choices=("cache", "cfg2", "hsdp2", "vae2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {}
    if args.mode == "cache":
        kwargs = {"cache_backend": "tea_cache", "cache_config": {"rel_l1_thresh": 0.2}}
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        auxiliary_text_encoder=args.llama,
        parallel_config=parallel_config(args.mode),
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=7200,
        stage_init_timeout=7200,
        **kwargs,
    )
    ready_seconds = time.perf_counter() - started
    requests = (
        ("A red ceramic teapot beside two blue cups on a wooden table, soft daylight.", 512, 512, 142),
        ("A small sailboat crossing a quiet lake beneath snowy mountains.", 768, 512, 314),
    )
    with (args.output / "results.jsonl").open("w") as records:
        for index, (prompt, width, height, seed) in enumerate(requests):
            params = OmniDiffusionSamplingParams(
                width=width,
                height=height,
                num_inference_steps=20,
                guidance_scale=1.0,
                true_cfg_scale=4.0 if args.mode == "cfg2" else 1.0,
                seed=seed,
                num_outputs_per_prompt=1,
            )
            started = time.perf_counter()
            outputs = engine.generate(
                {"prompt": prompt, "negative_prompt": "blurry, distorted", "modalities": ["image"]},
                sampling_params_list=[params],
            )
            seconds = time.perf_counter() - started
            images = extract_images_from_outputs(outputs)
            assert len(images) == 1 and images[0].size == (width, height)
            image_path = args.output / f"request-{index}.png"
            images[0].save(image_path)
            row = {
                "mode": args.mode,
                "request": index,
                "width": width,
                "height": height,
                "seed": seed,
                "ready_seconds": ready_seconds,
                "seconds": seconds,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
            records.write(json.dumps(row) + "\n")
            records.flush()
            print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
