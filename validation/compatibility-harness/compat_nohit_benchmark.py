"""Untimed TeaCache no-hit control through the supported coefficient configuration."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--tea-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    assert args.probe and args.tea_cache
    assert args.threshold == 0.2
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["COMPAT_PROBE_DIR"] = str(args.output.resolve())
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=2,
            use_hsdp=args.hsdp,
            hsdp_shard_size=2 if args.hsdp else -1,
        ),
        cache_backend="tea_cache" if args.tea_cache else None,
        cache_config={"rel_l1_thresh": args.threshold, "coefficients": [0.0, 0.0, 0.0, 0.0, 1.0]},
        diffusion_model_runner_cls="compat_probe.ProbeRunner" if args.probe else None,
        dtype="bfloat16",
        enforce_eager=True,
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height, prompt) in enumerate(
            (
                (
                    512,
                    512,
                    "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
                ),
                (768, 512, "A green bicycle beside a brick wall, afternoon sunshine."),
                (
                    512,
                    512,
                    "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
                ),
            )
        ):
            for iteration in range(1 if args.probe else 2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(
                    {"prompt": prompt, "modalities": ["image"]},
                    sampling_params_list=[params],
                )
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1 and images[0].size == (width, height)
                path = args.output / f"case-{case}-{iteration}.png"
                images[0].save(path)
                record = {
                    "case": case,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "probe": args.probe,
                    "hsdp": args.hsdp,
                    "tea_cache": args.tea_cache,
                    "threshold": args.threshold,
                    "forced_no_hit": True,
                    "coefficients": [0.0, 0.0, 0.0, 0.0, 1.0],
                    "width": width,
                    "height": height,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
