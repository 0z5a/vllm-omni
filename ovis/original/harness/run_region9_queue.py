"""Run a finite, explicitly authorized A100 queue without process signals."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    assert plan["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    assert [uuids[int(i)].strip() for i in plan["devices"].split(",")] == plan["gpu_uuids"]
    for name in plan["required_success"]:
        assert (root / name).read_text().strip() == "0", name
    for name in plan["required_files"]:
        assert (root / name).exists(), name
    evidence = root / "evidence" / args.run_id
    evidence.mkdir(exist_ok=False)
    with open("/tmp/0z5a-a100-perf.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        assert not processes.strip(), processes
        (evidence / "start.json").write_text(json.dumps({"plan": plan,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "lock": "/tmp/0z5a-a100-perf.lock", "lock_acquired": True,
            "authorization": "Human provided this host for quantization and authorized Ovis validation",
            "forced_signals": False}, indent=2))
        for index, step in enumerate(plan["steps"]):
            env = os.environ.copy()
            env.update(step["environment"])
            with (evidence / f"step-{index}.log").open("x") as log:
                child = subprocess.Popen(step["argv"], cwd=step["cwd"], env=env,
                    stdout=log, stderr=subprocess.STDOUT)
                (evidence / f"step-{index}.pid").write_text(str(child.pid))
                print(args.run_id, index, "PID", child.pid, flush=True)
                result = child.wait()
            (evidence / f"step-{index}.exit").write_text(str(result))
            if result:
                (evidence / "exit").write_text(str(result))
                raise SystemExit(result)
        (evidence / "exit").write_text("0")
        (evidence / "completed").write_text("All finite steps naturally exited0.\n")
        print(args.run_id, "natural exit0", flush=True)


if __name__ == "__main__":
    main()
