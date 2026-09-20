# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Summarize fresh HiDream L20 admissions and any completed E2E requests."""

import argparse
import json
from pathlib import Path

LABELS = {
    "cache": "B1 TeaCache",
    "cfg2": "B2 CFG parallel",
    "hsdp2": "B3 HSDP",
    "vae2": "B4 VAE decode",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# HiDream-I1-Full fresh two-L20 admission",
        "",
        "Exact HiDream and auxiliary Llama revisions were verified before these fresh processes. "
        "Failures are retained as hardware/loader evidence; no encoder or model component is removed.",
        "",
        "| Scope | Exit | Ready time | Request 1 | Request 2 | Speed comparison |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode, label in LABELS.items():
        mode_root = args.root / mode
        exit_code = int((mode_root / "exit-code").read_text())
        result_path = mode_root / "results.jsonl"
        rows = [json.loads(line) for line in result_path.read_text().splitlines()] if result_path.is_file() else []
        ready = f"{rows[0]['ready_seconds']:.3f} s" if rows else "N/A"
        first = f"{rows[0]['seconds']:.3f} s" if len(rows) > 0 else "N/A"
        second = f"{rows[1]['seconds']:.3f} s" if len(rows) > 1 else "N/A"
        comparison = "completed requests; paired baseline pending" if len(rows) == 2 else "N/A: model not ready"
        lines.append(f"| {label} | {exit_code} | {ready} | {first} | {second} | {comparison} |")
    lines += [
        "",
        "The four candidates are isolated: TeaCache is not combined with HSDP, CFG parallel, or VAE parallel. "
        "HSDP is tested before the other two-rank paths because it is the only candidate intended to reduce "
        "transformer residency. Detailed startup failures and GPU samples remain in each mode directory.",
    ]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
