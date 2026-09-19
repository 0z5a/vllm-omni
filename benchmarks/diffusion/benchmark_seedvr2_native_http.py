# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fresh-process APPA/PAAP validation of the actual video HTTP endpoint."""

import argparse
import hashlib
import io
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import requests


def gpu_snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    )


def request(fixture, url, width, height):
    started = time.perf_counter()
    with fixture.open("rb") as uploaded:
        response = requests.post(
            url + "/v1/videos/sync",
            data={
                "prompt": " ",
                "size": f"{width}x{height}",
                "seed": "7723",
                "num_frames": "5",
                "num_inference_steps": "1",
                "guidance_scale": "1.0",
            },
            files={"input_references": (fixture.name, uploaded, "video/x-matroska")},
            timeout=180,
        )
    seconds = time.perf_counter() - started
    response.raise_for_status()
    return seconds, response.content


def check_media(body, width, height):
    with av.open(io.BytesIO(body)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        assert (len(frames), stream.width, stream.height, str(stream.average_rate)) == (5, width, height, "25")
        assert [frame.pts for frame in frames] == [0, 512, 1024, 1536, 2048]
        pixels = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])
    with av.open(io.BytesIO(body)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.sample_rate == 16000
        assert float(stream.duration * stream.time_base) == 0.2
        audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=-1)
    return hashlib.sha256(pixels.tobytes()).hexdigest(), hashlib.sha256(audio.tobytes()).hexdigest()


def run_arm(directory, arm, args):
    directory.mkdir(parents=True)
    checkout = args.baseline if arm == "A" else args.candidate
    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="4",
        PYTHONPATH=os.pathsep.join((str(checkout), str(Path(__file__).resolve().parent), env.get("PYTHONPATH", ""))),
        SEEDVR2_AUDIT_DIR=str(directory),
    )
    command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        str(args.models),
        "--omni",
        "--model-class-name",
        "SeedVR2Pipeline",
        "--dtype",
        "float16",
        "--enforce-eager",
        "--num-gpus",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--worker-extension-cls",
        "seedvr2_native_worker_audit.NativeWorkerAudit",
    ]
    if args.server_cpus:
        command = ["taskset", "-c", args.server_cpus, *command]
    record = {
        "arm": arm,
        "checkout": str(checkout),
        "command": command,
        "gpu_before": gpu_snapshot(),
        "started": time.time(),
    }
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        record["pid"] = process.pid
        try:
            deadline = time.monotonic() + 180
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}: {directory}")
                try:
                    response = requests.get(url + "/health", timeout=2)
                    ready = response.status_code == 200
                except requests.ConnectionError:
                    pass  # Listener is absent during normal model startup.
                if ready:
                    break
                time.sleep(1)
            assert ready, "server readiness timeout"
            audits = list(directory.glob("worker-*.json"))
            assert len(audits) == 1, "missing worker execution-source evidence"
            audit = json.loads(audits[0].read_text())
            rope = audit["modules"]["vllm_omni.diffusion.models.seedvr2.rope"]
            assert Path(rope["path"]).is_relative_to(checkout)
            assert (
                rope["sha256"]
                == hashlib.sha256((checkout / "vllm_omni/diffusion/models/seedvr2/rope.py").read_bytes()).hexdigest()
            )
            record["warmup_seconds"] = [request(args.input, url, args.width, args.height)[0] for _ in range(5)]
            assert json.loads(audits[0].read_text())["audited_forwards"] == 6, "warmup audit did not finish"
            record["samples"] = []
            for repeat in range(24):
                seconds, body = request(args.input, url, args.width, args.height)
                video_hash, audio_hash = check_media(body, args.width, args.height)
                if repeat == 0:
                    (directory / "output.mp4").write_bytes(body)
                record["samples"].append({"seconds": seconds, "video_sha256": video_hash, "audio_sha256": audio_hash})
            record["median_seconds"] = statistics.median(row["seconds"] for row in record["samples"])
            record["gpu_after"] = gpu_snapshot()
            record["status"] = "complete"
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["forced_shutdown"] = True
            record["exit_code"] = process.returncode
            record["ended"] = time.time()
            (directory / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"path": str(directory), "arm": arm, "median_seconds": record["median_seconds"]}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=19824)
    parser.add_argument("--server-cpus")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--video-mae-limit", type=float, default=0)
    parser.add_argument("--video-max-error-limit", type=float, default=0)
    for name in ("baseline", "candidate", "models", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    args.baseline, args.candidate = args.baseline.resolve(), args.candidate.resolve()
    args.models, args.input = args.models.resolve(), args.input.resolve()
    destination.mkdir(exist_ok=False)
    contract = {
        "fixture_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "warmup_requests": 5,
        "measured_requests_per_arm": 24,
        "orders": ["AA"] if args.pilot else ["APPA", "PAAP", "APPA", "PAAP", "APPA"],
        "metric": "upload-through-response-download seconds",
        "MES_latency_reduction": 0.01,
        "AA_drift_limit": 0.15,
        "independent_unit": "fresh-process quartet",
        "gpu": args.gpu,
        "output_size": [args.width, args.height],
        "video_mae_limit": args.video_mae_limit,
        "video_max_error_limit": args.video_max_error_limit,
        "exclusive_host": False,
        "server_cpu_affinity": args.server_cpus,
        "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (destination / "contract.json").write_text(json.dumps(contract, indent=2))
    hashes = {"A": set(), "P": set()}
    first_outputs = {}
    for block, order in enumerate(contract["orders"]):
        for position, arm in enumerate(order):
            row = run_arm(destination / f"{block:02d}-{position}-{arm}", arm, args)
            hashes[arm].update((sample["video_sha256"], sample["audio_sha256"]) for sample in row["samples"])
            assert len(hashes[arm]) == 1, "within-variant decoded media mismatch"
            first_outputs.setdefault(arm, destination / f"{block:02d}-{position}-{arm}" / "output.mp4")
            if len(first_outputs) == 2:
                assert next(iter(hashes["A"]))[1] == next(iter(hashes["P"]))[1], "audio mismatch"
                pixels = []
                for variant in ("A", "P"):
                    with av.open(str(first_outputs[variant])) as container:
                        pixels.append(
                            np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
                        )
                error = np.abs(pixels[0].astype(np.float32) - pixels[1].astype(np.float32)) / 255
                quality = {"mae": float(error.mean()), "max_error": float(error.max())}
                (destination / "media-comparison.json").write_text(json.dumps(quality, indent=2))
                assert quality["mae"] <= args.video_mae_limit
                assert quality["max_error"] <= args.video_max_error_limit
    print("ALL_ARMS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
