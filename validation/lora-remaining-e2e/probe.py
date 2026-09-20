# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe real LoRA execution at offload, VAE-parallel, and FP8 boundaries."""

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from lora_probe import ProbeRunner as LoRAProbeRunner
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeExecutor
from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob

_layer_events: list[dict[str, object]] = []
_module_events: list[dict[str, object]] = []
_vae_events: list[dict[str, object]] = []
_original_prefetch = LayerwiseOffloadHook.prefetch_layer
_original_offload = LayerwiseOffloadHook.offload_layer
_original_move = SequentialOffloadHook._move_params
_original_vae_execute = DistributedVaeExecutor.execute


def observe_prefetch(self: LayerwiseOffloadHook, non_blocking: bool = True) -> None:
    _original_prefetch(self, non_blocking)
    _layer_events.append({"operation": "prefetch", "materialized": self.next_block_parameters != {}})


def observe_offload(self: LayerwiseOffloadHook) -> None:
    _original_offload(self)
    _layer_events.append({"operation": "offload", "materialized": self.is_materialized})


def observe_move(
    module: torch.nn.Module,
    target_device: torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> bool:
    moved = _original_move(
        module,
        target_device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
    )
    if moved:
        _module_events.append(
            {
                "module": type(module).__name__,
                "target": target_device.type,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            }
        )
    return moved


def observe_vae_execute(
    self: DistributedVaeExecutor,
    z: torch.Tensor,
    operator: object,
    broadcast_result: bool = True,
) -> torch.Tensor:
    _vae_events.append(
        {
            "rank": self.rank,
            "world_size": self.world_size,
            "parallel_size": self.parallel_size,
            "input_shape": list(z.shape),
        }
    )
    return _original_vae_execute(self, z, operator, broadcast_result)


LayerwiseOffloadHook.prefetch_layer = observe_prefetch
LayerwiseOffloadHook.offload_layer = observe_offload
SequentialOffloadHook._move_params = staticmethod(observe_move)
DistributedVaeExecutor.execute = observe_vae_execute


class ProbeRunner(LoRAProbeRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        self.fp8_calls: dict[str, int] = {}
        self.fp8_kernels: dict[str, str] = {}
        for name, layer in self.pipeline.transformer.named_modules():
            if isinstance(layer, LinearBase) and isinstance(
                layer.quant_method,
                (Fp8LinearMethod, Fp8PerTensorOnlineLinearMethod),
            ):
                kernel = layer.quant_method.fp8_linear
                assert isinstance(kernel, FP8ScaledMMLinearKernel)
                self.fp8_kernels[name] = type(kernel).__name__
                self.observe_kernel(name, kernel)
        expected_fp8 = os.environ["LORA_REMAINING_FEATURE"] == "fp8"
        assert bool(self.fp8_kernels) == expected_fp8

    def observe_kernel(self, name: str, kernel: FP8ScaledMMLinearKernel) -> None:
        original = kernel.apply_scaled_mm

        def observed(
            *,
            A: torch.Tensor,  # noqa: N803
            B: torch.Tensor,  # noqa: N803
            out_dtype: torch.dtype,
            As: torch.Tensor,  # noqa: N803
            Bs: torch.Tensor,  # noqa: N803
            bias: torch.Tensor | None,
            output_shape: list[int],
        ) -> torch.Tensor:
            assert A.dtype == B.dtype == torch.float8_e4m3fn
            assert A.is_cuda and B.is_cuda and B.stride(0) == 1
            assert torch.isfinite(As).all() and torch.isfinite(Bs).all()
            self.fp8_calls[name] = self.fp8_calls.get(name, 0) + 1
            return original(A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias, output_shape=output_shape)

        kernel.apply_scaled_mm = observed

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        _layer_events.clear()
        _module_events.clear()
        _vae_events.clear()
        self.fp8_calls.clear()
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        feature = os.environ["LORA_REMAINING_FEATURE"]
        if feature == "layer":
            assert any(event["operation"] == "prefetch" for event in _layer_events)
            assert any(event["operation"] == "offload" for event in _layer_events)
        elif feature == "module":
            targets = {event["target"] for event in _module_events}
            assert targets == {"cpu", "cuda"}
        elif feature == "fp8":
            assert set(self.fp8_calls) == set(self.fp8_kernels)
        elif feature == "u2-vae2":
            assert _vae_events and all(event["parallel_size"] == 2 for event in _vae_events)
        record = {
            "request_id": req.request_id,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "feature": feature,
            "offload_backend": type(self.offload_backend).__name__ if self.offload_backend is not None else None,
            "layer_events": _layer_events,
            "module_events": _module_events,
            "vae_events": _vae_events,
            "fp8_kernels": self.fp8_kernels,
            "fp8_calls": self.fp8_calls,
        }
        path = Path(os.environ["LORA_REMAINING_PROBE_DIR"]) / f"feature-rank-{record['rank']}.jsonl"
        with path.open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
