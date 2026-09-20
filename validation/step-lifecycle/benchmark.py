"""Full Helios request/step × HSDP/module-offload comparisons."""

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
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODES = ("native", "step", "hsdp", "step-hsdp", "module", "step-module")


def make_config(model: str, mode: str, probe: bool) -> OmniDiffusionConfig:
    step = mode.startswith("step")
    hsdp = "hsdp" in mode
    config = OmniDiffusionConfig.from_kwargs(
        model=model,
        parallel_config=DiffusionParallelConfig(use_hsdp=hsdp, hsdp_shard_size=2 if hsdp else -1),
        diffusion_offload_config={"mode": "module", "components": ["dit", "text_encoder"]}
        if "module" in mode
        else None,
        worker_extension_cls="probe.LifecycleProbe" if probe else None,
        step_execution=step,
        streaming_output=step,
        max_num_seqs=1,
        enforce_eager=True,
        dtype="bfloat16",
        vae_use_tiling=True,
    )
    config.enrich_config()
    assert config.model_class_name == "HeliosPyramidPipeline"
    return config


def request(request_id: str, frames: int) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={"prompt": "A red block moves slowly across a blue table in soft daylight.", "modalities": ["video"]},
        sampling_params=OmniDiffusionSamplingParams(
            width=640,
            height=384,
            num_frames=frames,
            num_inference_steps=30,
            guidance_scale=1.0,
            seed=142,
            num_outputs_per_prompt=1,
            extra_args={"is_enable_stage2": True, "pyramid_num_inference_steps_list": [10, 10, 10]},
        ),
    )


def generate(engine: DiffusionEngine, request_id: str, frames: int, action: str = "normal") -> DiffusionOutput:
    original_execute = engine.execute_fn
    chunks: list[torch.Tensor] = []
    ticks = 0

    def execute(schedule: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        nonlocal ticks
        result = original_execute(schedule)
        ticks += 1
        output = result.get_request_output(request_id)
        assert output is not None
        if output.result is not None and isinstance(output.result.output, torch.Tensor):
            chunks.append(output.result.output)
        if ticks == 3 and action == "pause":
            engine.executor.collective_rpc("lifecycle_snapshot", args=("before-pause",))
            for _ in range(2):
                empty = DiffusionSchedulerOutput(schedule.step_id, [], CachedRequestData([]), set(), 1, 0)
                paused = engine.executor.execute_step(empty)
                assert paused.get_request_output(request_id) is None
            engine.executor.collective_rpc("lifecycle_snapshot", args=("after-pause",))
        if ticks == 3 and action == "cancel":
            engine.abort(request_id)
        return result

    engine.execute_fn = execute
    try:
        output = engine.add_req_and_wait_for_response(request(request_id, frames))
    finally:
        engine.execute_fn = original_execute
    if chunks and not output.aborted and not output.error:
        output.output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    if action in ("pause", "cancel"):
        assert ticks >= 3, (action, ticks)
    return output


def render_video(engine: DiffusionEngine, output: DiffusionOutput, frames: int) -> np.ndarray:
    assert not output.error and not output.aborted, output
    assert isinstance(output.output, torch.Tensor)
    array = np.asarray(engine.post_process_func(output.output, output_type="np"))
    assert array.shape == (1, frames, 384, 640, 3), array.shape
    return array


def save_video(array: np.ndarray, directory: Path) -> str:
    assert np.isfinite(array).all()
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    target = directory / f"frames-{digest}.npz"
    if not target.exists():
        np.savez_compressed(target, frames=array)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["STEP_LIFECYCLE_OUTPUT"] = str(args.output.resolve())
    config = make_config(args.model, args.mode, args.probe)
    started = time.perf_counter()
    engine = DiffusionEngine.make_engine(config)
    ready_seconds = time.perf_counter() - started
    try:
        if args.probe:
            engine.executor.collective_rpc("lifecycle_install")
            engine.executor.collective_rpc("lifecycle_snapshot", args=("initial",))
        with (args.output / "results.jsonl").open("w") as records:
            cases = [(str(case), frames, "normal") for case, frames in enumerate((33, 66, 33))]
            if args.probe and config.step_execution:
                cases += [
                    ("pause", 33, "pause"),
                    ("cancel", 33, "cancel"),
                    ("after-cancel", 33, "normal"),
                    ("failure", 33, "failure"),
                    ("after-failure", 33, "normal"),
                ]
            reference = None
            for case, frames, action in cases:
                for iteration in range(1 if args.probe else 7):
                    request_id = f"{case}-{iteration}"
                    if action == "failure":
                        failed_rank = 1 if "hsdp" in args.mode else 0
                        engine.executor.collective_rpc("lifecycle_fail_next", args=(request_id, failed_rank))
                    started = time.perf_counter()
                    output = generate(engine, request_id, frames, action)
                    array = render_video(engine, output, frames) if not output.aborted and not output.error else None
                    seconds = time.perf_counter() - started
                    digest = None
                    if action == "cancel":
                        assert output.aborted and not engine.scheduler.has_requests()
                    elif action == "failure":
                        assert output.error and not engine.scheduler.has_requests()
                        assert "rank-local preparation failure" in output.error or "another DiT rank" in output.error
                    else:
                        assert array is not None
                        digest = save_video(array, args.output)
                        if case == "0" and iteration == 0:
                            reference = digest
                        if case in ("pause", "after-cancel", "after-failure", "2"):
                            assert digest == reference, (case, digest, reference)
                    row = {
                        "case": case,
                        "frames": frames,
                        "mode": args.mode,
                        "iteration": iteration,
                        "warmup": iteration < 2,
                        "probe": args.probe,
                        "seconds": seconds,
                        "ready_seconds": ready_seconds,
                        "output_sha256": digest,
                        "aborted": output.aborted,
                        "error": output.error,
                        "source": "bbd391d",
                        "checkpoint_revision": "b991c0379a018f4de3227d95468237f56066f5bb",
                        "seed": 142,
                        "width": 640,
                        "height": 384,
                        "steps": [10, 10, 10],
                        "guidance_scale": 1.0,
                        "dtype": "bfloat16",
                        "vae_dtype": "float32",
                        "gpus": 2 if "hsdp" in args.mode else 1,
                    }
                    records.write(json.dumps(row) + "\n")
                    records.flush()
                    if args.probe:
                        engine.executor.collective_rpc("lifecycle_snapshot", args=(f"after-{case}",))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
