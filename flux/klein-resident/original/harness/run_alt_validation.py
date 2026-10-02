"""Run a validation in the task-private D-drive runtime and wait for natural exit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["flux", "ovis"], default="flux")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("driver")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    evidence = ROOT / "evidence" / args.run_id
    evidence.mkdir(exist_ok=False)
    driver = ROOT / "harness" / args.driver
    assert driver.parent == ROOT / "harness" and driver.suffix == ".py"
    source_manifest = ROOT / "evidence" / f"{args.source}-source-manifest.json"
    source = json.loads(source_manifest.read_text())
    env = os.environ.copy()
    env.update(json.loads((ROOT / "inputs/alt-runtime-env.json").read_text()))
    env.update(
        PYTHONPATH=os.pathsep.join(str(p) for p in [ROOT / args.source, ROOT, ROOT / "dependencies"]),
        PYTHONPYCACHEPREFIX=str(ROOT / "cache/pyc"),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
        VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"),
        XDG_CACHE_HOME=str(ROOT / "cache/xdg"),
        HF_HOME=str(ROOT / "cache/hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="0",
    )
    command = [sys.executable, str(driver), *args.arguments]
    (evidence / "start.json").write_text(
        json.dumps(
            {
                "command": command,
                "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
                "source": args.source,
                "source_commit": source["commit"],
                "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                "environment": {key: env[key] for key in ["CC", "CXX", "CPATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES"]},
                "forced_signals": False,
            },
            indent=2,
        )
    )
    with (evidence / "process.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT / args.source, env=env, stdout=log, stderr=subprocess.STDOUT)
        (evidence / "pid").write_text(str(child.pid))
        print(args.run_id, "PID", child.pid, flush=True)
        result = child.wait()
    (evidence / "exit").write_text(str(result))
    print(args.run_id, "natural exit", result, flush=True)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
