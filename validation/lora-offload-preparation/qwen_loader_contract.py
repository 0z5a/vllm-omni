"""Admission check for the unmodified trained Qwen adapter on the pinned loader."""

import json
from pathlib import Path

import torch
from vllm.lora import lora_model

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

lora_model.PIN_MEMORY = False
path = Path("/home/kxqandccx/omni-1217-20260919/models/lora/edit-r1")
manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
manager.pipeline = torch.nn.Module()
manager.dtype = torch.float32
# Native Qwen attention stores the output projection as `to_out`, not a Sequential.
# Include logical Q/K/V names expanded from the native packed module mapping.
manager._expected_lora_modules = {
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
}
model, helper = manager._load_adapter(LoRARequest("edit-r1", 1, str(path)))
print(json.dumps({"loaded_modules": len(model.loras), "rank": helper.r, "alpha": helper.lora_alpha}), flush=True)
assert len(model.loras) == 480
for index in range(60):
    name = f"transformer.transformer_blocks.{index}.attn.to_out"
    assert manager._get_lora_weights(model, name) is not None, name
