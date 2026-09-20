"""Confirm the two trained adapters have genuinely different tensor contents."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
different = []
with (
    safe_open(root / "edit-r1/adapter_model.safetensors", framework="pt", device="cpu") as left,
    safe_open(root / "cocoedit/adapter_model.safetensors", framework="pt", device="cpu") as right,
):
    keys = sorted(left.keys())
    assert keys == sorted(right.keys()) and len(keys) == 960
    for key in keys:
        a, b = left.get_tensor(key), right.get_tensor(key)
        assert a.shape == b.shape and a.dtype == b.dtype
        if not torch.equal(a, b):
            different.append(key)
assert different
report = {"tensors": len(keys), "different_tensors": len(different), "different_keys": different}
(root / "qwen-adapter-differences.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"tensors": len(keys), "different_tensors": len(different)}), flush=True)
