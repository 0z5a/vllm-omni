"""Helios TeaCache lifecycle, quality, and speed validation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def make_config(model: str, mode: str, threshold: float | None, probe: bool) -> OmniDiffusionConfig:
    kwargs: dict[str, object] = {}
    if mode != "baseline":
        cache_config: dict[str, object] = {"rel_l1_thresh": threshold}
        if mode == "nohit":
            cache_config["coefficients"] = [0.0, 0.0, 0.0, 0.0, 1.0]
        kwargs = {"cache_backend": "tea_cache", "cache_config": cache_config}
    config = OmniDiffusionConfig.from_kwargs(
        model=model,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        diffusion_model_runner_cls="probe.ProbeRunner" if probe else None,
        enforce_eager=True,
        dtype="bfloat16",
        vae_use_tiling=True,
        init_timeout=7200,
        stage_init_timeout=7200,
        **kwargs,
    )
    config.enrich_config()
    assert config.model_class_name == "HeliosPyramidPipeline"
    assert not config.step_execution
    return config


def make_request(request_id: str, frames: int) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={
            "prompt": "A red wooden toy train moves slowly across a blue table in soft daylight.",
            "modalities": ["video"],
        },
        sampling_params=OmniDiffusionSamplingParams(
            width=640,
            height=384,
            num_frames=frames,
            num_inference_steps=6,
            guidance_scale=1.0,
            seed=142,
            num_outputs_per_prompt=1,
            extra_args={
                "is_enable_stage2": True,
                "pyramid_num_inference_steps_list": [2, 2, 2],
                "is_amplify_first_chunk": True,
            },
        ),
    )


def render(engine: DiffusionEngine, output: DiffusionOutput, frames: int) -> np.ndarray:
    assert not output.error and not output.aborted, output
    assert isinstance(output.output, torch.Tensor)
    array = np.asarray(engine.post_process_func(output.output, output_type="np"))
    assert array.shape == (1, frames, 384, 640, 3), array.shape
    assert np.isfinite(array).all()
    return array


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("baseline", "nohit", "cache"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.mode == "cache" and args.threshold is None:
        parser.error("--threshold is required for cache mode")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HELIOS_TEACACHE_PROBE_DIR"] = str(args.output.resolve())

    started = time.perf_counter()
    engine = DiffusionEngine.make_engine(make_config(args.model, args.mode, args.threshold, args.probe))
    ready_seconds = time.perf_counter() - started
    try:
        cases = (("a", 33), ("b", 66), ("a-reentry", 33)) if args.probe else tuple((str(i), 33) for i in range(7))
        reference_digest = None
        with (args.output / "results.jsonl").open("w") as records:
            for index, (case, frames) in enumerate(cases):
                started = time.perf_counter()
                output = engine.add_req_and_wait_for_response(make_request(case, frames))
                array = render(engine, output, frames)
                seconds = time.perf_counter() - started
                output_digest = digest(array)
                if case == "a":
                    reference_digest = output_digest
                if case == "a-reentry":
                    assert output_digest == reference_digest
                if args.probe or index == 2:
                    np.savez_compressed(args.output / f"{case}.npz", frames=array)
                row = {
                    "case": case,
                    "frames": frames,
                    "iteration": index,
                    "warmup": not args.probe and index < 2,
                    "probe": args.probe,
                    "mode": args.mode,
                    "threshold": args.threshold,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "output_sha256": output_digest,
                    "checkpoint_revision": "b991c0379a018f4de3227d95468237f56066f5bb",
                    "seed": 142,
                    "width": 640,
                    "height": 384,
                    "stage_steps": [2, 2, 2],
                    "guidance_scale": 1.0,
                    "dtype": "bfloat16",
                    "gpus": 2,
                    "parallel": "tp2",
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
