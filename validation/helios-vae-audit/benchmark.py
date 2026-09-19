"""Full Helios comparison with identical TP2 and independently timed requests."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-vae", action="store_true")
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        os.environ["HELIOS_PROBE_DIR"] = str(args.output)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2, vae_patch_parallel_size=2 if args.parallel_vae else 1
        ),
        vae_use_tiling=True,
        enforce_eager=True,
        dtype="bfloat16",
        step_execution=args.step_execution,
        diffusion_model_runner_cls="probe.VaeProbeRunner" if args.probe else None,
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    width, height = 640, 384
    image = torch.full((1, 3, height, width), -0.5)
    image[:, 0, 96:224, 192:320] = 1.0
    image[:, 1:, 96:224, 192:320] = -1.0
    video = torch.stack([torch.roll(image, frame, dims=-1) for frame in range(33)], dim=2)
    prompt = {"prompt": "A red block moves slowly across a blue table in soft daylight.", "modalities": ["video"]}
    cases = (("text", 33), ("image", 33), ("video", 66), ("text-reentry", 33))
    with (args.output / "results.jsonl").open("w") as records:
        for case, (kind, frames) in enumerate(cases[:1] if args.smoke else cases):
            for iteration in range(1 if args.probe or args.smoke else 7):
                extra = {"is_enable_stage2": True, "pyramid_num_inference_steps_list": [10, 10, 10]}
                if kind == "image":
                    extra["image"] = image
                elif kind == "video":
                    extra["video"] = video
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_frames=frames,
                    num_inference_steps=30,
                    guidance_scale=1.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                    extra_args=extra,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                seconds = time.perf_counter() - started
                assert len(outputs) == 1
                array = np.asarray(outputs[0].images)
                if array.ndim == 5:
                    assert array.shape[0] == 1
                    array = array[0]
                assert array.shape == (frames, height, width, 3), array.shape
                assert np.isfinite(array).all()
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                artifact = args.output / f"frames-{digest}.npz"
                if not artifact.exists():
                    np.savez_compressed(artifact, frames=array)
                if iteration == 0:
                    for index in (0, frames // 2, frames - 1):
                        Image.fromarray((array[index].clip(0, 1) * 255).round().astype(np.uint8)).save(
                            args.output / f"case-{case}-frame-{index}.png"
                        )
                row = {
                    "case": case,
                    "kind": kind,
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "parallel_vae": args.parallel_vae,
                    "step_execution": args.step_execution,
                    "probe": args.probe,
                    "dtype": str(array.dtype),
                    "output_sha256": digest,
                    "source": "75e089b" if args.parallel_vae else "698f716",
                    "checkpoint_revision": "b991c0379a018f4de3227d95468237f56066f5bb",
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
