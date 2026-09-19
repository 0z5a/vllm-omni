"""Verify every raw-reference output and summarize the controlled E2E quartet."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    arms = ("A0", "P0", "P1", "A1")
    rows = {}
    expected_counts = (1, 2, 3, 1)
    for arm in arms:
        root = args.evidence / arm
        assert (root / "status").read_text().strip() == "complete"
        rows[arm] = []
        offset = 0
        for index, line in enumerate((root / "results.jsonl").read_text().splitlines(keepends=True)):
            row = json.loads(line)
            case = index // 7
            assert row["reference_count"] == expected_counts[case]
            assert row["iteration"] == index % 7 and row["warmup"] == (index % 7 < 2)
            assert row["vae_parallel_size"] == 2 and row["global_posterior_rng_seed"] == 142
            assert row["ulysses_degree"] == 2 and row["ulysses_mode"] == "advanced_uaa"
            assert row["source_sha"] == ("f2c1684" if arm.startswith("P") else "15840a6")
            path = root / f"{row['reference_count']}-refs-{offset}-{row['iteration']}.png"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
            assert Image.open(path).size == (row["width"], row["height"]) == (512, 512)
            row["case"] = case
            rows[arm].append(row)
            offset += len(line.encode())
        assert len(rows[arm]) == 28
    table = [
        "| Reference images | Serial encode E2E (s) | Parallel encode E2E (s) | Speedup | Speed change | Output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    details = []
    for case, label in enumerate(("1 (first)", "2", "3", "1 (reentry)")):
        hashes = {row["image_sha256"] for arm in arms for row in rows[arm] if row["case"] == case}
        inputs = {tuple(row["reference_sha256"]) for arm in arms for row in rows[arm] if row["case"] == case}
        assert len(hashes) == len(inputs) == 1
        medians = {}
        ranges = {}
        for arm in arms:
            values = [row["seconds"] for row in rows[arm] if row["case"] == case and not row["warmup"]]
            assert len(values) == 5 and min(values) > 0
            medians[arm] = statistics.median(values)
            ranges[arm] = [min(values), max(values)]
        native = math.sqrt(medians["A0"] * medians["A1"])
        parallel = math.sqrt(medians["P0"] * medians["P1"])
        table.append(
            f"| {label} | {native:.4f} | {parallel:.4f} | {native / parallel:.4f}× | "
            f"{100 * (native / parallel - 1):+.2f}% | Exact |"
        )
        details.append(
            {
                "case": case,
                "reference_count": expected_counts[case],
                "medians": medians,
                "ranges": ranges,
                "output_sha256": next(iter(hashes)),
            }
        )
    assert details[0]["output_sha256"] == details[3]["output_sha256"]
    table.extend(
        [
            "",
            "All 112 PNG hashes checked; outputs are identical across arms and repeats for each reference "
            "set, including A→B→C→A reentry. Both arms use the same two-rank VAE decode and Ulysses "
            "topology. Only reference encoding changes; source f2c1684 is compared against its dependency "
            "15840a6.",
            "",
            "One balanced A/P/P/A quartet, two warmups and five timed requests per case and fresh process. "
            "Reported latency is the geometric mean of process medians. Worker-global posterior RNG is reset "
            "to 142 on both sides for controlled replay. Shared-host results; no confidence interval or "
            "isolated-device claim. Small differences do not establish a speed benefit.",
            "",
        ]
    )
    (args.evidence / "analysis.json").write_text(json.dumps(details, indent=2) + "\n")
    result = "\n".join(table)
    (args.evidence / "speed-comparison.md").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
