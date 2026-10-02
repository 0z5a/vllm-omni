"""Execute a sealed sequence of independent full-request validation processes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    source_manifest = ROOT / "evidence/flux-source-manifest.json"
    source = json.loads(source_manifest.read_text())
    assert source["commit"] == plan["source_commit"]
    for name, digest in plan["inputs"].items():
        assert (
            hashlib.sha256((ROOT / "inputs" / name).read_bytes()).hexdigest() == digest
        ), name
    evidence = ROOT / "evidence" / args.run_id
    evidence.mkdir(exist_ok=False)
    manifest = {
        "plan": plan,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_commit": source["commit"],
        "source_manifest_sha256": hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest(),
        "harness": {
            name: hashlib.sha256((ROOT / "harness" / name).read_bytes()).hexdigest()
            for name in [
                "bench_pipeline.py",
                "audit_worker.py",
                "run_thor_validation.py",
                "run_thor_campaign.py",
            ]
        },
        "forced_signals": False,
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for index, run in enumerate(plan["runs"]):
        status = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (evidence / f"gpu-before-{index}.csv").write_text(status)
        command = [
            sys.executable,
            str(ROOT / "harness/run_thor_validation.py"),
            "--run-id",
            run["id"],
            "bench_pipeline.py",
            *run["arguments"],
            "--output",
            str(ROOT / "evidence" / run["id"] / "pipeline"),
        ]
        result = subprocess.run(command, check=False)
        (evidence / f"run-{index}.exit").write_text(str(result.returncode))
        if result.returncode:
            (evidence / "exit").write_text(str(result.returncode))
            raise SystemExit(result.returncode)
        assert (ROOT / "evidence" / run["id"] / "pipeline/completed").exists()
    (evidence / "completed").write_text(
        "All planned full-request processes completed and naturally exited.\n"
    )
    (evidence / "exit").write_text("0")


if __name__ == "__main__":
    main()
