"""Acquire both host leases before the paired, finite VRAM READ starts."""

import fcntl
import json
import os
from pathlib import Path
import runpy
import select
import subprocess
import sys
import time

import torch
from nixl import _api  # noqa: F401


def main() -> None:
    role, host, output = sys.argv[1:4]
    root = Path(os.environ["NIXL_TASK_ROOT"])
    assert not torch.cuda.is_initialized()
    with open(os.environ["NIXL_GPU_LOCK"], "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        assert uuid == os.environ["NIXL_EXPECT_GPU_UUID"]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        assert boot == os.environ["NIXL_EXPECT_BOOT_ID"]
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        assert not active, active
        if "NIXL_QUEUE_STATUS" in os.environ:
            assert Path(os.environ["NIXL_QUEUE_STATUS"]).read_text().startswith("WAITING_GPU_MEMORY_DISK")
        Path(output + ".lease.json").write_text(json.dumps({"pid": os.getpid(), "role": role, "gpu_uuid": uuid, "boot_id": boot, "cuda_initialized": False}) + "\n")
        print("LEASE_READY", role, flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], 45)
        if not ready or sys.stdin.readline().strip() != "GO":
            print("No paired start; lease returned without a CUDA context", flush=True)
            return
        os.environ["NIXL_GPU_LOCK_FD"] = str(lease.fileno())
        os.environ["NIXL_GPU_LEASE_STARTED"] = str(started)
        sys.argv = [str(root / "check_native_peer_read.py"), role, host, output, "--gpu"]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        finally:
            Path(output + ".scope-end.json").write_text(json.dumps({"pid": os.getpid(), "epoch": time.time(), "lease_body_elapsed_s": time.perf_counter() - started}) + "\n")


if __name__ == "__main__":
    main()
