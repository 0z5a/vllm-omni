# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paced realtime capture for the MOSS-VL-Realtime timestamped-stream fixture.

The offline capture adapter replays a fixed input tensor set. This one drives the
reference model's realtime session along the fixture's arrival timeline instead:
prompts and frames are pushed at their ``at_ms`` offsets with a media clock, and
every acceptance, output chunk and lifecycle event is recorded with wall-clock
time plus the media timestamp it carried.

Session output is scheduled by a background loop, so this run is a behavioural
capture, not a bit-reproducible one: the artifacts record the timeline that
produced them and the comparison must use the same timeline, not the same text.

Example:
    CUDA_VISIBLE_DEVICES=1 python tests/assets/moss_vl_realtime/capture_realtime.py \
        --checkpoint /path/to/MOSS-VL-Realtime \
        --case timestamped_stream --playback-speed 1.0 \
        --out artifacts/moss/2cb8df5c-0e016ef7
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch

ASSETS = Path(__file__).resolve().parent


def load_case(cases_path: Path, case_id: str) -> dict[str, Any]:
    fixtures = json.loads(cases_path.read_text(encoding="utf-8"))
    return next(case for case in fixtures["cases"] if case["id"] == case_id)


def asset_path(manifest: dict[str, Any], name: str) -> Path:
    return (ASSETS / manifest["assets"][name]["path"]).resolve()


def environment(checkpoint: Path, revision: str | None, case: dict[str, Any]) -> dict[str, Any]:
    import transformers

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": revision,
        "case": case["id"],
        "seed": 0,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor

    checkpoint = Path(args.checkpoint).resolve()
    case = load_case(Path(args.cases), args.case)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    artifact_root = Path(args.out).resolve()
    out_dir = artifact_root / case["id"] / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict[str, Any]] = []

    def record(kind: str, **fields: Any) -> None:
        events.append({"kind": kind, "wall": time.time() - started, **fields})

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        device_map={"": args.device},
    )
    model.eval()

    steps: list[dict[str, Any]] = []
    if args.dump_steps:
        original_forward = model.forward

        def forward_wrapper(*fargs: Any, **fkwargs: Any) -> Any:
            out = original_forward(*fargs, **fkwargs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            input_ids = fkwargs.get("input_ids")
            position_ids = fkwargs.get("position_ids")
            steps.append(
                {
                    "step": len(steps),
                    "input_ids": None if input_ids is None else input_ids[0].tolist(),
                    "position_ids": None if position_ids is None else position_ids[:, 0].tolist(),
                    "argmax": int(logits[0, -1].float().argmax()),
                    "max": float(logits[0, -1].float().max()),
                }
            )
            return out

        model.forward = forward_wrapper

    from PIL import Image

    prompts = [event for event in case["events"] if event["kind"] == "prompt"]
    initial_prompt = prompts[0]["text"] if prompts else ""
    session = model.create_realtime_session(
        processor,
        initial_prompt=initial_prompt,
        frame_queue_size=args.frame_queue_size,
        max_tokens_per_turn=10**9,
        do_sample=False,
    )
    session.start()

    started = time.monotonic()
    record("session_open", initial_prompt=initial_prompt)

    outputs: list[dict[str, Any]] = []
    pushed_prompts = {prompts[0]["id"]} if prompts else set()
    frames_pushed = 0

    def drain() -> None:
        while True:
            chunk = session.poll_output(timeout=0.0)
            if chunk is None:
                return
            outputs.append({"wall": time.monotonic() - started, "text": chunk})
            record("output", text=chunk)

    for event in case["events"]:
        if event["kind"] == "prompt" and event["id"] in pushed_prompts:
            continue
        target = event["at_ms"] / 1000.0 / args.playback_speed
        while (time.monotonic() - started) < target:
            drain()
            time.sleep(0.01)

        if event["kind"] == "prompt":
            session.push_prompt(event["text"])
            pushed_prompts.add(event["id"])
            record("prompt_pushed", event_id=event["id"], text=event["text"], pending_frames=session.pending_frames)
        else:
            image = Image.open(asset_path(manifest, event["asset"])).convert("RGB")
            dropped = session.push_frame(image, timestamp=event["media_timestamp_ms"] / 1000.0, drop_oldest=True)
            frames_pushed += 1
            record(
                "frame_pushed",
                event_id=event["id"],
                asset=event["asset"],
                media_timestamp_ms=event["media_timestamp_ms"],
                dropped_older=dropped,
                pending_frames=session.pending_frames,
            )

    deadline = time.monotonic() + args.drain_seconds
    while time.monotonic() < deadline:
        drain()
        time.sleep(0.05)
    drain()

    session.close(timeout=args.close_timeout)
    record("session_closed", active=session.active, pending_frames=session.pending_frames)

    report = {
        "environment": environment(checkpoint, args.revision, case),
        "case": case,
        "playback_speed": args.playback_speed,
        "frames_pushed": frames_pushed,
        "output_chunks": len(outputs),
        "combined_output": "".join(chunk["text"] for chunk in outputs),
        "outputs": outputs,
        "event_count": len(events),
    }
    (out_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(report["environment"], indent=2) + "\n", encoding="utf-8")
    if args.dump_steps:
        report["steps"] = len(steps)
        (out_dir / "steps.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8")
    (out_dir / "realtime-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "result.md").write_text(
        "\n".join(
            [
                f"# {case['id']} (paced realtime)",
                "",
                f"- frames pushed: {frames_pushed}",
                f"- output chunks: {len(outputs)}",
                f"- playback speed: {args.playback_speed}",
                f"- combined output: {report['combined_output']!r}",
                "",
                "Reproduce:",
                "",
                "```bash",
                f"CUDA_VISIBLE_DEVICES={args.device} python tests/assets/moss_vl_realtime/capture_realtime.py \\",
                f"  --checkpoint <checkpoint> --case {case['id']} --out {artifact_root}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("frames_pushed", "output_chunks", "combined_output")}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cases", default=str(ASSETS / "cases.json"))
    parser.add_argument("--manifest", default=str(ASSETS / "manifest.json"))
    parser.add_argument("--case", default="timestamped_stream")
    parser.add_argument("--out", required=True)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--frame-queue-size", type=int, default=256)
    parser.add_argument("--drain-seconds", type=float, default=15.0)
    parser.add_argument(
        "--dump-steps",
        action="store_true",
        help="record every forward's inputs, positions and argmax for step-level comparison",
    )
    parser.add_argument("--close-timeout", type=float, default=5.0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
