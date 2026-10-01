# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record a NIXL baseline and require all eight native CUDA cases to run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = {
    "connector": "tests/distributed/omni_connectors/test_nixl_connector.py",
    "layout": "tests/diffusion/diffusion_kv/test_layout.py",
    "native": "tests/distributed/omni_connectors/test_nixl_connector_native.py",
}
NATIVE_CASES = {
    f"test_native_two_process_structured_mixed_device_transfer[{direct}-{kind}]"
    for direct in ("False", "True")
    for kind in ("generic", "h3_ref2va", "h3_fl2va", "h3_t2va")
}


def junit_summary(path: Path, *, native: bool = False) -> dict[str, int]:
    cases = list(ET.parse(path).iter("testcase"))
    passed = {
        case.attrib["name"]
        for case in cases
        if all(case.find(status) is None for status in ("failure", "error", "skipped"))
    }
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    if not cases or failed:
        raise RuntimeError(f"{path.name}: tests={len(cases)}, failed={failed}, skipped={skipped}")
    if native and (skipped or not NATIVE_CASES.issubset(passed)):
        raise RuntimeError(f"Native NIXL cases did not execute successfully: {sorted(NATIVE_CASES - passed)}")
    return {"tests": len(cases), "passed": len(cases) - failed - skipped, "failed": failed, "skipped": skipped}


def capture(repo: Path, out: Path, name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    (out / name).write_text(result.stdout + result.stderr)
    return result


def preflight(repo: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    commands = {
        "sha.txt": ["git", "rev-parse", "HEAD"],
        "git-status.txt": ["git", "status", "--short"],
        "pip-freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "pip-check.txt": [sys.executable, "-m", "pip", "check"],
        "gpu.txt": ["nvidia-smi", "-q"],
        "topology.txt": ["nvidia-smi", "topo", "-m"],
    }
    returncodes = {name: capture(repo, out, name, command).returncode for name, command in commands.items()}
    packages = {
        distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()
    }
    manifest = {
        "python": sys.version,
        "executable": sys.executable,
        "repo": str(repo),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {name: packages.get(name) for name in ("torch", "vllm", "vllm-omni", "nixl", "transformers")},
        "returncodes": returncodes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(returncodes.values()):
        raise RuntimeError(f"Preflight failed: {returncodes}")
    import torch
    from vllm.distributed.nixl_utils import NixlWrapper

    if NixlWrapper is None or not torch.accelerator.is_available() or torch.accelerator.device_count() < 2:
        raise RuntimeError("Native validation requires NIXL and two visible CUDA devices")
    import vllm

    import vllm_omni

    (out / "imports.json").write_text(
        json.dumps({"vllm": vllm.__file__, "vllm_omni": vllm_omni.__file__}, indent=2) + "\n"
    )


def run_tests(repo: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int | float]] = {}
    for name, test in TESTS.items():
        junit = out / f"{name}.xml"
        command = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-s", test, f"--junitxml={junit}"]
        env = dict(os.environ, UCX_RCACHE_MAX_UNRELEASED="1024")
        start = time.monotonic()
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {out / f'{name}.log'}")
        summary[name] = {**junit_summary(junit, native=name == "native"), "elapsed_s": time.monotonic() - start}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{name}: {summary[name]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    preflight(repo, out / "preflight")
    run_tests(repo, out / "baseline")


if __name__ == "__main__":
    main()
