# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate K3 outputs and write the speed comparison table."""

import json
import statistics
from pathlib import Path
from typing import TypedDict


class Row(TypedDict):
    case: int
    iteration: int
    warmup: bool
    seconds: float
    ready_seconds: float
    image_sha256: str


def read_arm(root: Path, name: str) -> list[Row]:
    path = root / name / "results.jsonl"
    rows: list[Row] = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 21 and sum(not row["warmup"] for row in rows) == 15
    assert (root / name / "status").read_text().strip() == "complete"
    return rows


def main() -> None:
    root = Path(__file__).parent / "evidence"
    arms = {name: read_arm(root, name) for name in ("native0", "hsdp0", "hsdp1", "native1")}
    labels = {0: "512×512", 1: "768×512", 2: "512×512 reentry"}
    lines = [
        "# K3 Ring attention × HSDP results",
        "",
        "All arms use Ring degree 2 on GPUs 2 and 3. Each median contains ten measured requests",
        "from two fresh processes; two warmups per process are excluded.",
        "",
        "| Case | Ring median (s) | Ring + HSDP median (s) | Speedup | Latency reduction | PNG parity |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case, label in labels.items():
        reference = [
            row["seconds"]
            for name in ("native0", "native1")
            for row in arms[name]
            if row["case"] == case and not row["warmup"]
        ]
        feature = [
            row["seconds"]
            for name in ("hsdp0", "hsdp1")
            for row in arms[name]
            if row["case"] == case and not row["warmup"]
        ]
        hashes = {row["image_sha256"] for rows in arms.values() for row in rows if row["case"] == case}
        assert len(reference) == len(feature) == 10 and len(hashes) == 1
        ref_median = statistics.median(reference)
        feature_median = statistics.median(feature)
        speedup = ref_median / feature_median
        reduction = (ref_median - feature_median) / ref_median * 100
        lines.append(
            f"| {label} | {ref_median:.6f} | {feature_median:.6f} | {speedup:.3f}× | {reduction:+.2f}% | exact |"
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
