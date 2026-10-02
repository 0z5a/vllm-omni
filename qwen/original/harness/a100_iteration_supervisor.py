"""Continue sealed finite validation plans independently of the SSH client."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "evidence/iteration-supervisor-v1"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def publish(phase, **details):
    write_json(
        STATE / "state.json",
        {
            "phase": phase,
            "pid": os.getpid(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def receipt(request_id, phase, **details):
    write_json(STATE / "receipts" / f"{request_id}.json", {"phase": phase, **details})


def execute(path):
    request = json.loads(path.read_text())
    request_id = request["id"]
    if (STATE / "receipts" / f"{request_id}.json").exists():
        return
    terminals = [ROOT / name for name in request["after_terminal"]]
    if not all(marker.exists() for marker in terminals):
        return
    if any(marker.read_text().strip() != "0" for marker in terminals):
        receipt(request_id, "dependency_failed", markers=list(map(str, terminals)))
        return
    if "screen" in request:
        arms = [json.loads((ROOT / name).read_text()) for name in request["screen"]]
        assert all(
            arm["ok"] and not arm["synthetic"] and len(arm["repetitions"]) == 3
            for arm in arms
        )
        for key in (
            "height",
            "width",
            "steps",
            "seed",
            "prompt_sha256",
            "cfg_scale",
            "extra_engine_kwargs",
        ):
            assert arms[0][key] == arms[1][key], key
        settings = [
            {
                key: value
                for key, value in arm["resolved_config"].items()
                if key != "repo"
            }
            for arm in arms
        ]
        assert settings[0] == settings[1]
        reference, candidate = [arm["repetitions"][2]["wall_ms"] for arm in arms]
        reduction = (1 - candidate / reference) * 100
        if reduction <= 5:
            receipt(
                request_id,
                "needs_iteration",
                screening_reduction_pct=reduction,
                final_acceptance=False,
            )
            return
    plan_path = ROOT / request["plan"]
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == request["plan_sha256"]
    for deployment in request["sources"]:
        manifest = json.loads((ROOT / deployment["manifest"]).read_text())
        assert manifest["commit"] == deployment["commit"]
        for name, expected in manifest["files"].items():
            source = ROOT / deployment["directory"] / name
            data = (
                os.readlink(source).encode()
                if source.is_symlink()
                else source.read_bytes()
            )
            assert hashlib.sha256(data).hexdigest() == expected, name
    runner = ROOT / "harness/run_region9_queue.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == request["runner_sha256"]
    run_id = request["run_id"]
    assert not (ROOT / "evidence" / run_id).exists()
    with (ROOT / "evidence" / f"{run_id}-entry.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, str(runner), "--plan", str(plan_path), "--run-id", run_id],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        (ROOT / "evidence" / f"{run_id}-entry.pid").write_text(str(child.pid))
        while child.poll() is None:
            publish("running", request=request_id, run_id=run_id, child_pid=child.pid)
            time.sleep(20)
        status = child.returncode
    (ROOT / "evidence" / f"{run_id}-entry.exit").write_text(str(status))
    receipt(
        request_id,
        "queue_completed" if status == 0 else "queue_failed",
        exit=status,
        final_acceptance=False,
    )


def main():
    STATE.mkdir(exist_ok=True)
    for directory in ("inbox", "receipts"):
        (STATE / directory).mkdir(exist_ok=True)
    (STATE / "pid").write_text(str(os.getpid()))
    while not (STATE / "finished").exists():
        publish("watching")
        for path in sorted((STATE / "inbox").glob("*.json")):
            execute(path)
        time.sleep(20)
    publish("finished")


if __name__ == "__main__":
    main()
