"""Validate real PEFT tensors through the pinned diffusion manager's CPU loader."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.lora import lora_model

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

# Pinning requires a visible CUDA device; this host-memory-only probe disables it.
lora_model.PIN_MEMORY = False

root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
reports = []
for adapter_id, name in enumerate(("helio", "fuli"), 1):
    path = root / name
    config = json.loads((path / "adapter_config.json").read_text())
    # Exercise the real loader only; no fake device wrappers or model-fit claim.
    manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
    manager.pipeline = torch.nn.Module()
    manager.dtype = torch.float32
    manager._expected_lora_modules = set(config["target_modules"])
    model, helper = manager._load_adapter(LoRARequest(name, adapter_id, str(path)))
    assert len(model.loras) == 204
    assert helper.r == config["r"] and helper.lora_alpha == config["lora_alpha"]
    checked = set()
    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as raw:
        assert len(raw.keys()) == 2 * len(model.loras)
        for module_name, weights in model.loras.items():
            prefix = f"base_model.model.{module_name}"
            a = raw.get_tensor(prefix + ".lora_A.weight").float()
            b = raw.get_tensor(prefix + ".lora_B.weight").float()
            expected_b = b * (config["lora_alpha"] / config["r"])
            assert weights.scaling == 1
            torch.testing.assert_close(weights.lora_a, a, rtol=0, atol=0)
            torch.testing.assert_close(weights.lora_b, expected_b, rtol=0, atol=0)
            suffix = module_name.rsplit(".", 1)[-1]
            if suffix not in checked:
                x = torch.linspace(-0.25, 0.25, a.shape[1]).reshape(1, -1)
                reference = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
                reference *= config["lora_alpha"] / config["r"]
                actual = torch.nn.functional.linear(torch.nn.functional.linear(x, weights.lora_a), weights.lora_b)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
                checked.add(suffix)
    assert checked == set(config["target_modules"])
    report = {
        "adapter": name,
        "modules": len(model.loras),
        "rank": helper.r,
        "alpha": helper.lora_alpha,
        "scaling_folded_exact": True,
        "projection_suffixes_checked": sorted(checked),
        "device": "cpu",
        "dtype": "float32",
    }
    reports.append(report)
    print(json.dumps(report), flush=True)
(root / "cpu-loader-report.json").write_text(json.dumps(reports, indent=2) + "\n")
