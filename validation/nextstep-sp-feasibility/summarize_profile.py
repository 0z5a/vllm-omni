"""Summarize separate intrusive NextStep phase traces, never benchmark speedups."""

import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sessions: dict[int, list[dict[str, object]]] = {0: [], 1: []}
shape_pattern = re.compile(r'\{"profile_rank":.*\}')
time_pattern = re.compile(r"NextStep11Pipeline\.(model\.forward|model\.image_head\.sample|vae\.decode) took ([0-9.]+)s")
rank_pattern = re.compile(r"DiffusionWorker_TP([01]) pid=")
for line in args.log.read_text().splitlines():
    shape_match = shape_pattern.search(line)
    if shape_match:
        row = json.loads(shape_match.group())
        rank = row["profile_rank"]
        if row["call"] == 0:
            sessions[rank].append({"shapes": [], "model.forward": [], "model.image_head.sample": [], "vae.decode": []})
        sessions[rank][-1]["shapes"].append(row)
    timing = time_pattern.search(line)
    if timing:
        rank_match = rank_pattern.search(line)
        assert rank_match, line
        rank = int(rank_match[1])
        assert sessions[rank], line
        sessions[rank][-1][timing[1]].append(float(timing[2]))
assert len(sessions[0]) == len(sessions[1]) == 3
report = []
for rank, requests in sessions.items():
    for request, sample in enumerate(requests):
        backbone = sample["model.forward"]
        head = sample["model.image_head.sample"]
        vae = sample["vae.decode"]
        assert len(backbone) == 1025 and len(head) == 1024 and len(vae) == 1, (
            rank,
            request,
            len(backbone),
            len(head),
            len(vae),
        )
        parts = {"prefill": backbone[0], "decode": sum(backbone[1:]), "flow_head": sum(head), "vae": sum(vae)}
        total = sum(parts.values())
        report.append(
            {
                "rank": rank,
                "request": request,
                "phase": ("startup", "warmup", "diagnostic")[request],
                "warmup": request < 2,
                "seconds": parts,
                "fraction": {key: value / total for key, value in parts.items()},
                "profiled_total_seconds": total,
                "shapes": sample["shapes"],
                "ideal_prefill_only_speedup_bound": total / (total - parts["prefill"]),
            }
        )
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "phase-summary.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# NextStep intrusive phase profile",
    "",
    "Source `43c4e79`, TP2, full checkpoint, 512², seed 142. Startup, one warmup and one diagnostic request. "
    "These synchronized/logged phase timings are not E2E benchmark samples.",
    "",
    "| Rank | Request | Prefill (s) | AR decode (s) | Flow head (s) | VAE (s) | "
    "Prefill share | Perfect-prefill upper bound |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in report:
    s = row["seconds"]
    lines.append(
        f"| {row['rank']} | {row['phase']} | {s['prefill']:.6f} | {s['decode']:.6f} | "
        f"{s['flow_head']:.6f} | {s['vae']:.6f} | {100 * row['fraction']['prefill']:.4f}% | "
        f"{row['ideal_prefill_only_speedup_bound']:.6f}× |"
    )
lines += [
    "",
    "The bound assumes zero prefill time and no added communication, using only the profiled phases. "
    "It is not an observed speedup. Startup uses nine prefill tokens; "
    "the measured prompt uses 24 with CFG batch two. Decode queries have length one "
    "while the measured prompt cache grows to 1,048 positions. "
    "Longer text and image-conditioned prefill remain unmeasured, "
    "so this result does not establish general SP feasibility. "
    "No SP implementation or distributed prefill correctness is claimed.",
]
(args.output / "README.md").write_text("\n".join(lines) + "\n")
