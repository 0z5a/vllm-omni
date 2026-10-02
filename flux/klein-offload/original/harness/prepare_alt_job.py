"""Prepare a task-private runtime or pinned model; all children exit naturally."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=["runtime", "klein", "kontext", "ovis"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    name = f"alt-{args.job}-{args.run_id}"
    evidence = ROOT / "evidence"
    with (evidence / f"{name}.started").open("x") as file:
        file.write(str(os.getpid()))
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "bootstrap"),
        TMPDIR=str(ROOT / "tmp"),
        HF_HOME=str(ROOT / "cache/hf"),
        HF_HUB_DISABLE_XET="1",
        PIP_CACHE_DIR=str(ROOT / "cache" / name),
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    pip = [sys.executable, "-m", "pip", "install", "--no-compile"]
    if args.job == "runtime":
        commands = [
            [
                *pip,
                "--target",
                str(ROOT / "dependencies"),
                "--report",
                str(evidence / f"{name}-pip-report.json"),
                "-r",
                str(ROOT / "inputs/alt-runtime-requirements.txt"),
            ]
        ]
    else:
        commands = []
        hub_metadata = ROOT / "dependencies/download/huggingface_hub-1.31.0.dist-info/METADATA"
        if not hub_metadata.exists():
            commands.append([*pip, "--target", str(ROOT / "dependencies/download"), "huggingface-hub==1.31.0"])
        model = {"klein": "FLUX.2-klein-4B", "kontext": "FLUX.1-Kontext-dev", "ovis": "Ovis-Image-7B"}[args.job]
        commands.append([sys.executable, str(ROOT / "harness/fetch_pinned_model.py"), model])
    for index, command in enumerate(commands):
        print(json.dumps({"phase": index, "command": command}), flush=True)
        result = subprocess.run(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL, check=False)
        (evidence / f"{name}-phase{index}.exit").write_text(str(result.returncode))
        if result.returncode:
            (evidence / f"{name}.exit").write_text(str(result.returncode))
            raise SystemExit(result.returncode)
    (evidence / f"{name}.exit").write_text("0")


if __name__ == "__main__":
    main()
