# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate MOSS-VL-Realtime baseline and parity artifacts into one report.

Reads the artifact tree written by ``capture_reference.py`` and ``runner.py``
(one directory per case and run id) and produces the baseline table used in the
model documentation and the pull request: per-case prompt length, generated
tokens, reference and native wall time, decode rate, peak memory, and the
numerical comparison when a native run recorded one.

Example:
    python -m vllm_omni.model_executor.models.moss_vl_realtime.baseline \
        --artifacts artifacts/moss/2cb8df5c-0e016ef7e \
        --native-root artifacts/moss/native-timed \
        --out artifacts/moss/baseline.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latest_run(case_dir: Path) -> Path | None:
    runs = sorted(entry for entry in case_dir.iterdir() if entry.is_dir())
    return runs[-1] if runs else None


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def reference_row(case: str, run: Path | None) -> dict[str, Any]:
    if run is None:
        return {"case": case, "status": "no reference run"}
    performance = read_json(run / "performance.json") or {}
    numerical = read_json(run / "numerical-report.json") or {}
    environment = read_json(run / "environment.json") or {}
    return {
        "case": case,
        "status": "captured",
        "run": run.name,
        "prompt_tokens": performance.get("prompt_len"),
        "generated_tokens": performance.get("generated_tokens"),
        "wall_seconds": round(performance.get("wall_seconds", 0.0), 3),
        "decode_tokens_per_second": (
            round(performance["decode_tokens_per_second"], 2) if performance.get("decode_tokens_per_second") else None
        ),
        "device": environment.get("device_name"),
        "attention_backend": environment.get("attn_implementation"),
        "text": numerical.get("decoded_text"),
    }


def native_row(case: str, run_dir: Path | None) -> dict[str, Any]:
    report = read_json(run_dir / "native-report.json") if run_dir else None
    if report is None:
        return {"case": case, "status": "no native run"}
    comparison = report.get("comparison") or {}
    row: dict[str, Any] = {
        "case": case,
        "status": "replayed",
        "prompt_tokens": report["prompt_len"],
        "generated_tokens": len(report["generated_token_ids"]),
        "prefill_seconds": round(report["prefill_seconds"], 3),
        "decode_seconds": round(report["decode_seconds"], 3),
        "decode_tokens_per_second": (
            round(report["decode_tokens_per_second"], 2) if report.get("decode_tokens_per_second") else None
        ),
        "wall_seconds": round(report["prefill_seconds"] + report["decode_seconds"], 3),
        "peak_memory_bytes": report.get("peak_memory_bytes"),
        "loaded_keys": report["load_report"]["loaded_keys"],
        "missing_keys": len(report["load_report"]["missing_keys"]),
        "unexpected_keys": len(report["load_report"]["unexpected_keys"]),
    }
    if comparison:
        row["reference_wall_seconds"] = round(comparison["reference_wall_seconds"], 3)
        row["wall_speedup"] = round(comparison["wall_speedup"], 2)
        row["token_prefix_matches"] = comparison["token_prefix_matches"]
        row["step_argmax_agreement"] = comparison["step_argmax_agreement"]
        logits = comparison.get("prefill_logits_delta") or {}
        row["prefill_logits_max_abs"] = round(logits["max_abs"], 4) if logits else None
    return row


def markdown(reference: list[dict[str, Any]], native: list[dict[str, Any]]) -> str:
    lines = ["# MOSS-VL-Realtime baseline", "", "## Reference capture", ""]
    lines += [
        "| case | prompt tokens | generated | wall (s) | decode (tok/s) | backend | device |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in reference:
        if row.get("status") != "captured":
            lines.append(f"| {row['case']} | — | — | — | — | — | {row['status']} |")
            continue
        lines.append(
            f"| {row['case']} | {row['prompt_tokens']} | {row['generated_tokens']} | {row['wall_seconds']} | "
            f"{row['decode_tokens_per_second']} | {row['attention_backend']} | {row['device']} |"
        )

    lines += ["", "## Native replay", ""]
    header = (
        "| case | prompt | generated | prefill (s) | decode (s) | decode (tok/s) | wall (s) "
        "| reference wall (s) | speedup | tokens | logits max Δ |"
    )
    lines += [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in native:
        if row.get("status") != "replayed":
            lines.append(f"| {row['case']} | — | — | — | — | — | — | — | — | — | {row['status']} |")
            continue
        reference_wall = row.get("reference_wall_seconds", "—")
        speedup = row.get("wall_speedup", "—")
        tokens = row.get("token_prefix_matches", "—")
        logits = row.get("prefill_logits_max_abs", "—")
        lines.append(
            f"| {row['case']} | {row['prompt_tokens']} | {row['generated_tokens']} | {row['prefill_seconds']} | "
            f"{row['decode_seconds']} | {row['decode_tokens_per_second']} | {row['wall_seconds']} | "
            f"{reference_wall} | {speedup} | {tokens} | {logits} |"
        )

    lines += ["", "## Loading", ""]
    for row in native:
        if row.get("status") != "replayed":
            continue
        lines.append(
            f"- `{row['case']}`: loaded {row['loaded_keys']} keys, {row['missing_keys']} missing, "
            f"{row['unexpected_keys']} unexpected, peak {row['peak_memory_bytes'] / 2**30:.2f} GiB"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="artifact root holding <case>/<run-id> reference runs")
    parser.add_argument("--native-root", default=None, help="directory holding <case>/native-report.json")
    parser.add_argument("--cases", default="prompt_only,image,short_video")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    artifacts = Path(args.artifacts).resolve()
    native_root = Path(args.native_root).resolve() if args.native_root else None
    cases = [case.strip() for case in args.cases.split(",") if case.strip()]

    reference = []
    for case in cases:
        case_dir = artifacts / case
        reference.append(reference_row(case, latest_run(case_dir) if case_dir.is_dir() else None))
    native = [
        native_row(case, native_root / case if native_root and (native_root / case).is_dir() else None)
        for case in cases
    ]

    report = {
        "artifacts": str(artifacts),
        "native_root": str(native_root) if native_root else None,
        "reference": reference,
        "native": native,
    }
    text = markdown(reference, native)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
