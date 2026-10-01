"""Verify file storage and the native pageable offloader without model weights."""

import hashlib
import json
import os
import fcntl
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import nixl_disk_offload as offload


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


records = []
for dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
    source = (torch.arange(1024, dtype=torch.float32).reshape(32, 32) % 16).to(dtype).T
    with torch.device("cuda"):
        mapped = offload.disk_backed(source)
    assert mapped.device.type == "cpu" and mapped.stride() == source.stride()
    assert digest(mapped) == digest(source)
    module = nn.Linear(32, 32, bias=False)
    module.weight = nn.Parameter(mapped, requires_grad=False)
    before = len(list(offload.ROOT.glob("*.bin")))
    with offload.disk_loading_context(module, torch.device("cpu")):
        assert module.weight.data_ptr() == mapped.data_ptr()
    assert len(list(offload.ROOT.glob("*.bin"))) == before
    records.append({"dtype": str(dtype), "sha256": digest(mapped)})
assert not torch.cuda.is_initialized()

gpu = "--gpu" in sys.argv
if gpu:
    lease = open("/tmp/0z5a-sm120-perf.lock", "a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    assert not active, active
    print("GPU lease acquired", flush=True)
    offload.install()
    torch.manual_seed(123)
    module = nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn(3, 1024, dtype=torch.bfloat16, device="cuda")
    expected = F.linear(inputs, module.weight)
    module = offload.DiskUVAOffloader(16 * 1024**2)._maybe_offload_to_cpu(module)
    assert module.weight.device.type == "cpu"
    before = sorted(offload.ROOT.glob("*.bin"))
    for _ in range(3):
        torch.testing.assert_close(module(inputs), expected, rtol=0, atol=0)
    with torch.no_grad(), offload.disk_loading_context(module, torch.device("cuda")):
        module.weight.add_(1 / 64)
        expected = F.linear(inputs, module.weight)
    assert module.weight.device.type == "cpu"
    assert sorted(offload.ROOT.glob("*.bin")) == before
    for _ in range(3):
        torch.testing.assert_close(module(inputs), expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    records.append({"native_forwards": 6, "gpu_context_file_reused": True, "sha256": digest(module.weight.detach())})

path = Path(os.environ["NIXL_TASK_ROOT"]) / f"disk-offload-{'gpu' if gpu else 'cpu'}-proof-v2.json"
path.write_text(json.dumps({"pid": os.getpid(), "torch": torch.__version__, "cases": records}, indent=2) + "\n")
print(path, flush=True)
