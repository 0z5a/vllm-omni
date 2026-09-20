"""Validate raw full-model artifacts before reporting paired timing results."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

MODES = ("native", "step", "hsdp", "step-hsdp", "module", "step-module")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_artifacts(directory: Path) -> list[dict]:
    records = rows(directory / "results.jsonl")
    for record in records:
        digest = record["output_sha256"]
        if digest is None:
            assert record["case"] in ("cancel", "failure")
            continue
        with np.load(directory / f"frames-{digest}.npz") as artifact:
            array = artifact["frames"]
            assert array.shape == (1, record["frames"], 384, 640, 3)
            assert np.isfinite(array).all()
            assert hashlib.sha256(array.tobytes()).hexdigest() == digest
    return records


def check_probes(root: Path) -> None:
    manifest = [line.split(maxsplit=1) for line in Path(__file__).with_name("runtime.sha256").read_text().splitlines()]
    runner_sha = next(sha for sha, name in manifest if name.endswith("/worker/diffusion_model_runner.py"))
    probe_sha = hashlib.sha256(Path(__file__).with_name("probe.py").read_bytes()).hexdigest()
    baseline = {r["case"]: r["output_sha256"] for r in check_artifacts(root / "native-probe")}
    for mode in MODES:
        directory = root / f"{mode}-probe"
        records = check_artifacts(directory)
        assert len(records) == (8 if mode.startswith("step") else 3)
        for record in records:
            if record["case"] in ("0", "1", "2"):
                assert record["output_sha256"] == baseline[record["case"]], (mode, record["case"])
        ranks = 2 if "hsdp" in mode else 1
        assert len(list(directory.glob("rank-*.jsonl"))) == ranks
        all_waves = []
        for rank in range(ranks):
            snapshots = rows(directory / f"rank-{rank}.jsonl")
            by_label = {r["label"]: r for r in snapshots}
            final = snapshots[-1]
            assert all(r["rank"] == rank for r in snapshots)
            assert all(r["runner_sha256"] == runner_sha and r["probe_sha256"] == probe_sha for r in snapshots)
            assert all("/dev/shm/0z5a-step-lifecycle-bbd391d/" in r["runner_file"] for r in snapshots)
            assert all(r["module_offload_enabled"] == ("module" in mode) for r in snapshots)
            assert all(bool(r["fsdp_modules"]) == ("hsdp" in mode) for r in snapshots)
            if "hsdp" in mode:
                assert final["components"]["dit"]["dtensors"] > 0
            if "module" in mode:
                assert final["moves"]
                assert any(move["target"] == "cpu" for move in final["moves"])
                assert any(move["target"].startswith("cuda") for move in final["moves"])
            else:
                assert not final["moves"]
            assert {e["component"] for e in final["events"]} >= {"dit", "text_encoder", "vae"}
            if mode.startswith("step"):
                before, after = by_label["before-pause"], by_label["after-pause"]
                for key in ("states", "batch_ids", "components", "events", "moves"):
                    assert before[key] == after[key], (mode, rank, key)
                assert len(after["waves"]) == len(before["waves"]) + 2
                assert all(not wave["scheduled"] and not wave["outputs"] for wave in after["waves"][-2:])
                for label in ("after-cancel", "after-failure", "after-after-cancel", "after-after-failure"):
                    assert not by_label[label]["states"] and by_label[label]["batch_ids"] is None
                assert not by_label["after-cancel"]["lora_ids"]
                failures = [event for event in final["events"] if event["component"] == "injected_failure"]
                assert len(failures) == int(rank == ranks - 1)
                if "module" in mode:
                    first_waves = {}
                    for index, wave in enumerate(final["waves"]):
                        for request_id in wave["scheduled"]:
                            first_waves.setdefault(request_id, index)
                    assert all(move["wave"] in first_waves.values() for move in final["moves"])
            all_waves.append(final["waves"])
        assert all(waves == all_waves[0] for waves in all_waves)
    print("All six full-model probes passed output, lifecycle and actual feature gates.")


def summarize(root: Path) -> None:
    check_probes(root)
    timings = {}
    for mode in MODES:
        arms = [check_artifacts(root / f"{mode}{index}") for index in (0, 1)]
        for arm in arms:
            assert len(arm) == 21
            assert {(r["case"], r["iteration"]) for r in arm} == {(str(c), i) for c in range(3) for i in range(7)}
            assert all(not r["error"] and not r["aborted"] for r in arm)
            assert len({r["output_sha256"] for r in arm if r["case"] in ("0", "2")}) == 1
        for case in range(3):
            case_rows = [r for arm in arms for r in arm if r["case"] == str(case)]
            assert len({r["output_sha256"] for r in case_rows}) == 1
            probe = next(r for r in rows(root / f"{mode}-probe/results.jsonl") if r["case"] == str(case))
            assert case_rows[0]["output_sha256"] == probe["output_sha256"]
            timings[mode, case] = statistics.median(r["seconds"] for r in case_rows if not r["warmup"])
    lines = [
        "# Helios full E2E lifecycle comparison",
        "",
        "Two fresh timing processes per configuration, 2 warmups + 5 measured requests per case per process.",
        "",
        "| Comparison | Frames/case | Reference (s) | Candidate (s) | Speedup | Latency reduction | GPU-s ref/new |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for left, right in (
        ("native", "step"),
        ("native", "hsdp"),
        ("step", "step-hsdp"),
        ("native", "module"),
        ("step", "step-module"),
        ("hsdp", "step-hsdp"),
        ("module", "step-module"),
    ):
        for case, frames in enumerate((33, 66, 33)):
            a, b = timings[left, case], timings[right, case]
            gpu_a = a * (2 if "hsdp" in left else 1)
            gpu_b = b * (2 if "hsdp" in right else 1)
            lines.append(
                f"| {left} → {right} | {frames}/{case} | {a:.4f} | {b:.4f} | {a / b:.4f}× | "
                f"{(1 - b / a) * 100:.2f}% | {gpu_a:.4f}/{gpu_b:.4f} |"
            )
    lines += [
        "",
        "Single-GPU modes use GPU 2; HSDP uses GPUs 2 and 3.",
        "GPU-seconds = latency × allocated GPU count.",
        "All comparisons require exact full-video hashes; probes are excluded from timing.",
        "Shared-host results do not establish isolated-host performance.",
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--probes-only", action="store_true")
    args = parser.parse_args()
    check_probes(args.root) if args.probes_only else summarize(args.root)
