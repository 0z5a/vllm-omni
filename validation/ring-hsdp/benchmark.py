# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fresh-process ZImage Ring attention with optional HSDP."""

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
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["RING_HSDP_PROBE_DIR"] = str(args.output.resolve())
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            ring_degree=2,
            use_hsdp=args.hsdp,
            hsdp_shard_size=2 if args.hsdp else -1,
        ),
        diffusion_model_runner_cls="probe.ProbeRunner" if args.probe else None,
        dtype="bfloat16",
        enforce_eager=True,
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    cases = (
        (512, 512, "A red ceramic teapot beside two blue cups on a wooden table, soft daylight."),
        (768, 512, "A green bicycle beside a brick wall, afternoon sunshine."),
        (512, 512, "A red ceramic teapot beside two blue cups on a wooden table, soft daylight."),
    )
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height, prompt) in enumerate(cases):
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
