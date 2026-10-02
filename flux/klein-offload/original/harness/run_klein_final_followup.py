"""Run final-source Klein encoder and full-request comparisons."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
NAME = "alt-klein-final-campaign-v1"
source = json.loads((EVIDENCE / "flux-source-manifest.json").read_text())
assert source["commit"] == "1d39542fb477d4b1954be489115f95850106e5de"
assert (EVIDENCE / "loader-fp8-host-init-regression-v1/exit").read_text() == "0"
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
(EVIDENCE / (NAME + ".pid")).write_text(str(__import__("os").getpid()))
commands = [
    ["harness/verify_final_klein_model.py"],
    [
        "harness/run_alt_validation.py",
        "--run-id",
        "alt-klein-final-encoder-v1",
        "validate_klein_encoder.py",
        "--model",
        str(ROOT / "models/FLUX.2-klein-4B"),
        "--output",
        str(EVIDENCE / "alt-klein-final-encoder-v1/encoder"),
    ],
    [
        "harness/run_alt_campaign.py",
        "--plan",
        str(ROOT / "inputs/alt-klein-final-plan-v1.json"),
        "--run-id",
        "alt-klein-final-matrix-v1",
    ],
]
(EVIDENCE / (NAME + ".start.json")).write_text(
    json.dumps(
        {
            "source": source["commit"],
            "controller_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "forced_signals": False,
            "commands": commands,
        },
        indent=2,
    )
)
for index, command in enumerate(commands):
    with (EVIDENCE / f"{NAME}-{index}.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, str(ROOT / command[0]), *command[1:]],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        (EVIDENCE / f"{NAME}-{index}.pid").write_text(str(child.pid))
        print(index, "entry PID", child.pid, flush=True)
        code = child.wait()
    (EVIDENCE / f"{NAME}-{index}.exit").write_text(str(code))
    print(index, "natural exit", code, flush=True)
    if code:
        (EVIDENCE / (NAME + ".exit")).write_text(str(code))
        raise SystemExit(code)
(EVIDENCE / (NAME + ".exit")).write_text("0")
(EVIDENCE / (NAME + ".completed")).write_text(
    "All final-source Klein validation stages naturally completed.\n"
)
