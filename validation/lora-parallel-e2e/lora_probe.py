"""Untimed worker-side LoRA activation and executed-projection observations."""

import hashlib
import json
import os
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.lora.request import LoRARequest

_original_set_active = DiffusionLoRAManager.set_active_adapter
_original_apply = DiffusionBaseLinearLayerWithLoRA.apply
_managers: list[DiffusionLoRAManager] = []
_calls: dict[int, int] = {}


def observe_activation(self: DiffusionLoRAManager, lora_request: LoRARequest | None, lora_scale: float = 1.0) -> None:
    _original_set_active(self, lora_request, lora_scale)
    if self not in _managers:
        _managers.append(self)
    expected = lora_request.lora_int_id if lora_request is not None and lora_scale != 0.0 else None
    assert self._active_adapter_id == expected
    if expected is not None:
        assert self._adapter_scales[expected] == round(lora_scale, 3)
        assert self._lora_modules
    active = 0
    for layer in self._lora_modules.values():
        assert isinstance(layer, DiffusionBaseLinearLayerWithLoRA)
        active += sum(layer._diffusion_lora_active_slices)
    assert active > 0 if expected is not None else active == 0
    _calls.clear()


def observe_apply(
    self: DiffusionBaseLinearLayerWithLoRA, x: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    _calls[id(self)] = _calls.get(id(self), 0) + 1
    for active, a, b in zip(self._diffusion_lora_active_slices, self.lora_a_stacked, self.lora_b_stacked):
        if active:
            assert a.device == b.device == x.device and a.dtype == b.dtype == x.dtype
            assert a.shape[-1] == x.shape[-1] and a.shape[-2] == b.shape[-1]
    return _original_apply(self, x, bias)


DiffusionLoRAManager.set_active_adapter = observe_activation
DiffusionBaseLinearLayerWithLoRA.apply = observe_apply


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class ProbeRunner(DiffusionModelRunner):
    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        manager = next(manager for manager in _managers if manager.pipeline is self.pipeline)
        layers = []
        for name, layer in manager._lora_modules.items():
            assert isinstance(layer, DiffusionBaseLinearLayerWithLoRA)
            layers.append(
                {
                    "name": name,
                    "active_slices": layer._diffusion_lora_active_slices,
                    "calls": _calls.get(id(layer), 0),
                    "a_shapes": [list(tensor.shape) for tensor in layer.lora_a_stacked],
                    "b_shapes": [list(tensor.shape) for tensor in layer.lora_b_stacked],
                    "devices": [str(tensor.device) for tensor in layer.lora_a_stacked + layer.lora_b_stacked],
                    "tp_size": layer.tp_size,
                    "a_hashes": [digest(tensor) for tensor in layer.lora_a_stacked],
                    "b_hashes": [digest(tensor) for tensor in layer.lora_b_stacked],
                    "a_half_hashes": [
                        [digest(chunk) for chunk in tensor.chunk(2, dim=-1)] for tensor in layer.lora_a_stacked
                    ],
                    "b_half_hashes": [
                        [digest(chunk) for chunk in tensor.chunk(2, dim=-2)] for tensor in layer.lora_b_stacked
                    ],
                }
            )
        assert any(layer["calls"] for layer in layers), "No injected projection executed"
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        record = {
            "request_id": req.request_id,
            "rank": rank,
            "pid": os.getpid(),
            "active_adapter": manager._active_adapter_id,
            "registered_adapters": sorted(manager._registered_adapters),
            "scales": manager._adapter_scales,
            "layers": layers,
        }
        with (Path(os.environ["LORA_PROBE_DIR"]) / f"rank-{rank}.jsonl").open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
