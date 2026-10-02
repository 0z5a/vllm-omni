"""Run the final Kontext matrix after a complete encoder smoke."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

r = Path("/mnt/d/0z5a/flux-ovis-consolidation-20261001")
e = r / "evidence"
n = "alt-kontext-validated-campaign-v6"
assert (e / "alt-flux-shutdown-regression-v3.exit").read_text() == "0"
assert (e / "alt-kontext-validated-campaign-v5.exit").read_text() == "2"
assert (
    "required: --output" in (e / "alt-kontext-encoder-smoke-v1/process.log").read_text()
)
assert not (e / (n + ".pid")).exists()
deploy = json.loads((e / "flux-followup-deployment-v1.json").read_text())
assert deploy["commit"] == "56c92df7b6c4a41e8d3fd6b0ab61af711346ac07"
assert all(
    hashlib.sha256((r / p).read_bytes()).hexdigest() == h
    for p, h in deploy["helpers"].items()
)
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
(e / (n + ".pid")).write_text(str(os.getpid()))
(e / (n + ".start.json")).write_text(
    json.dumps(
        {
            "commit": deploy["commit"],
            "helpers": deploy["helpers"],
            "controller_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "forced_signals": False,
        },
        indent=2,
    )
)
print(n, "controller PID", os.getpid(), flush=True)
plan = json.loads((r / "inputs/alt-kontext-plan-v5.json").read_text())
arguments = list(plan["runs"][1]["arguments"])
arguments[arguments.index("--warmups") + 1] = "0"
arguments[arguments.index("--requests") + 1] = "1"
commands = [
    (
        "smoke",
        [
            sys.executable,
            str(r / "harness/run_alt_validation.py"),
            "--run-id",
            "alt-kontext-encoder-smoke-v2",
            "bench_pipeline.py",
            *arguments,
            "--output",
            str(e / "alt-kontext-encoder-smoke-v2/pipeline"),
        ],
    ),
    (
        "matrix",
        [
            sys.executable,
            str(r / "harness/run_alt_campaign.py"),
            "--plan",
            str(r / "inputs/alt-kontext-plan-v5.json"),
            "--run-id",
            "alt-kontext-campaign-v5",
        ],
    ),
    (
        "t5-audit",
        [
            sys.executable,
            str(r / "harness/run_alt_validation.py"),
            "--run-id",
            "alt-kontext-t5-encoder-audit-v1",
            "validate_t5_encoder.py",
            "--model",
            str(r / "models/FLUX.1-Kontext-dev"),
            "--output",
            str(e / "alt-kontext-t5-encoder-audit-v1/encoder"),
        ],
    ),
]
for stage, command in commands:
    with (e / (n + "-" + stage + ".log")).open("x") as log:
        child = subprocess.Popen(command, cwd=r, stdout=log, stderr=subprocess.STDOUT)
        (e / (n + "-" + stage + ".pid")).write_text(str(child.pid))
        print(stage, "entry PID", child.pid, flush=True)
        code = child.wait()
    (e / (n + "-" + stage + ".exit")).write_text(str(code))
    print(stage, "natural exit", code, flush=True)
    if code:
        (e / (n + ".exit")).write_text(str(code))
        raise SystemExit(code)
    if stage == "smoke":
        p = e / "alt-kontext-encoder-smoke-v2/pipeline"
        assert (p / "completed").exists()
        snap = json.loads((p / "worker-after.json").read_text())[0][0]
        encoder = [
            v for v in snap["projections"] if v["name"].startswith("text_encoder_2.")
        ]
        assert len(encoder) == 144, len(encoder)
        assert all(
            v["dtype"] == "torch.float8_e4m3fn"
            and v["backend"] == "CutlassFP8ScaledMMLinearKernel"
            for v in encoder
        )
        assert (
            not snap["torch_compile"]["compiled_forwards"]
            and snap["torch_compile"]["enforce_eager"] is True
        )
(e / (n + ".exit")).write_text("0")
(e / (n + ".completed")).write_text(
    "Final source encoder full-request smoke, complete repeated matrix, and T5 audit naturally completed.\n"
)
