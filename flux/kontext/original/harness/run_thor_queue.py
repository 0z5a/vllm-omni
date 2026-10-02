"""Run a finite Thor validation queue in the coordinator's released GPU window."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    lease = json.loads(Path(args.lease).read_text())
    assert lease["released"] and lease["devices"] == "0"
    assert lease["coordinator"] == "01a0f5cc-5aea-75b3-896c-ea80f3e737b5"
    assert (
        lease["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    )
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    assert len(uuids) == 1 and lease["gpu_uuids"] == [uuids[0].strip()]
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    assert not processes.strip(), processes
    assert (ROOT / "evidence/thor-runtime-prepare-v4/exit").read_text().strip() == "0"
    assert (
        ROOT / "evidence/thor-cpu-runtime-import-v5/exit"
    ).read_text().strip() == "0"
    assert (ROOT / "evidence/thor-native-gpu-probe.completed").exists()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    for model in plan["models"]:
        assert (ROOT / "models" / model / ".verified-complete").exists()
    with open("/tmp/codex-thor-perf.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = ROOT / "evidence" / args.run_id
        evidence.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            THOR_QUEUE_ID=args.run_id,
            OMP_NUM_THREADS="4",
            PYTHONPATH=os.pathsep.join(
                str(p)
                for p in [ROOT / "dependencies-quick-v1", ROOT, ROOT / "dependencies"]
            ),
            PYTHONPYCACHEPREFIX=str(ROOT / "cache/pyc"),
            HF_HOME=str(ROOT / "cache/hf"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TMPDIR=str(ROOT / "tmp"),
        )
        (evidence / "start.json").write_text(
            json.dumps(
                {
                    "lease": lease,
                    "plan": plan,
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "lock": "/tmp/codex-thor-perf.lock",
                    "lock_acquired": True,
                    "forced_signals": False,
                },
                indent=2,
            )
        )
        for index, command in enumerate(plan["commands"]):
            with (evidence / f"step-{index}.log").open("x") as log:
                child = subprocess.Popen(
                    [sys.executable, *command],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                (evidence / f"step-{index}.pid").write_text(str(child.pid))
                print(args.run_id, index, "PID", child.pid, flush=True)
                result = child.wait()
            (evidence / f"step-{index}.exit").write_text(str(result))
            if result:
                (evidence / "exit").write_text(str(result))
                raise SystemExit(result)
        (evidence / "exit").write_text("0")
        (evidence / "completed").write_text(
            "All finite queue steps naturally exited0.\n"
        )
        print(args.run_id, "natural exit0", flush=True)


if __name__ == "__main__":
    main()
