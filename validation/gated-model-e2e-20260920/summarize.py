# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def load(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 21
    assert sum(not row["warmup"] for row in rows) == 15
    assert len(list(path.glob("*.png"))) == 21
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    groups: dict[str, list[Path]] = defaultdict(list)
    for arm in sorted(args.evidence.iterdir()):
        if arm.is_dir() and (arm / "status").read_text().strip() == "complete":
            groups[arm.name.rsplit("-", 1)[0]].append(arm)

    lines = [
        "# A2/A4/A5/H1 full E2E results",
        "",
        "| Scope | Resolution | Reference median (s) | Feature median (s) | "
        "Speedup | Latency reduction | Mean RGB delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, arms in sorted(groups.items()):
        assert len(arms) == 4
        by_suffix = {arm.name.rsplit("-", 1)[1]: arm for arm in arms}
        arm_rows = {suffix: load(arm) for suffix, arm in by_suffix.items()}
        for case in range(3):
            ref = [
                (by_suffix[suffix], row)
                for suffix in ("ref0", "ref1")
                for row in arm_rows[suffix]
                if row["case"] == case and not row["warmup"]
            ]
            feature = [
                (by_suffix[suffix], row)
                for suffix in ("new0", "new1")
                for row in arm_rows[suffix]
                if row["case"] == case and not row["warmup"]
            ]
            assert len(ref) == len(feature) == 10
            ref_seconds = statistics.median(float(row["seconds"]) for _, row in ref)
            feature_seconds = statistics.median(float(row["seconds"]) for _, row in feature)
            deltas = []
            for (left_arm, left), (right_arm, right) in zip(ref, feature):
                left_image = Image.open(left_arm / f"{case}-{left['iteration']}.png")
                right_image = Image.open(right_arm / f"{case}-{right['iteration']}.png")
                deltas.append(sum(ImageStat.Stat(ImageChops.difference(left_image, right_image)).mean) / 3)
            resolution = f"{ref[0][1]['width']}×{ref[0][1]['height']}"
            lines.append(
                f"| {scope} | {resolution} | {ref_seconds:.3f} | {feature_seconds:.3f} | "
                f"{ref_seconds / feature_seconds:.3f}× | {(ref_seconds - feature_seconds) / ref_seconds:.2%} | "
                f"{statistics.mean(deltas):.6f} |"
            )
    (args.evidence / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
