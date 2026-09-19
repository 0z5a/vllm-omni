"""Real-weight loading/scaling with a metadata-only native projection tree."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.lora import lora_model
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm_omni.diffusion.models.qwen_image.lora import QwenImageLoRAMixin

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.lora.request import LoRARequest

lora_model.PIN_MEMORY = False
path = Path("/home/kxqandccx/omni-1217-20260919/models/lora/edit-r1")


class Pipeline(torch.nn.Module, QwenImageLoRAMixin):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
        torch.nn.Module.__init__(self.transformer)
        for name in ("to_qkv", "add_kv_proj", "to_out", "to_add_out"):
            layer = RowParallelLinear.__new__(RowParallelLinear)
            torch.nn.Module.__init__(layer)
            self.transformer.add_module(name, layer)


manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
manager.pipeline = Pipeline()
manager.dtype = torch.float32
manager._expected_lora_modules = {"to_out"}
model, helper = manager._load_adapter(LoRARequest("edit-r1", 1, str(path)))
assert len(model.loras) == 480
assert "attn.to_out" in helper.target_modules and "attn.to_out.0" not in helper.target_modules
with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as raw:
    for module_name, weights in model.loras.items():
        original = module_name + "."
        if original.endswith(".to_out."):
            original = original.removesuffix(".") + ".0."
        a = raw.get_tensor(original + "lora_A.weight").float()
        b = raw.get_tensor(original + "lora_B.weight").float() * 2
        torch.testing.assert_close(weights.lora_a, a, rtol=0, atol=0)
        torch.testing.assert_close(weights.lora_b, b, rtol=0, atol=0)
        assert weights.scaling == 1
for index in range(60):
    assert manager._get_lora_weights(model, f"transformer.transformer_blocks.{index}.attn.to_out") is not None
print(json.dumps({"modules": 480, "tensors": 960, "output_projections": 60, "scaling_exact": True}), flush=True)
