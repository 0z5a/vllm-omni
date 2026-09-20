"""Untimed observations at the actual online FP8 GEMM dispatch boundary."""

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class ProbeRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, ZImagePipeline)
        self.calls: dict[str, int] = {}
        self.layouts: dict[str, dict[str, object]] = {}
        self.kernels: dict[str, str] = {}
        for name, layer in self.pipeline.transformer.named_modules():
            if isinstance(layer, LinearBase) and isinstance(
                layer.quant_method, (Fp8LinearMethod, Fp8PerTensorOnlineLinearMethod)
            ):
                kernel = layer.quant_method.fp8_linear
                assert isinstance(kernel, FP8ScaledMMLinearKernel)
                self.kernels[name] = type(kernel).__name__
                self.observe_kernel(name, kernel)
        assert bool(self.kernels) == (os.environ["FP8_PROBE_EXPECTED"] == "1")

    def observe_kernel(self, name: str, kernel: FP8ScaledMMLinearKernel) -> None:
        original = kernel.apply_scaled_mm

        def observed(
            *,
            A: torch.Tensor,  # noqa: N803 - upstream kernel keyword
            B: torch.Tensor,  # noqa: N803 - upstream kernel keyword
            out_dtype: torch.dtype,
            As: torch.Tensor,  # noqa: N803 - upstream kernel keyword
            Bs: torch.Tensor,  # noqa: N803 - upstream kernel keyword
            bias: torch.Tensor | None,
            output_shape: list[int],
        ) -> torch.Tensor:
            assert A.dtype == B.dtype == torch.float8_e4m3fn
            assert A.is_cuda and B.is_cuda and B.stride(0) == 1
            assert torch.isfinite(As).all() and torch.isfinite(Bs).all()
            self.calls[name] = self.calls.get(name, 0) + 1
            self.layouts[name] = {
                "a_dtype": str(A.dtype),
                "b_dtype": str(B.dtype),
                "b_shape": list(B.shape),
                "b_stride": list(B.stride()),
                "weight_scale_shape": list(Bs.shape),
                "weight_scale_dtype": str(Bs.dtype),
                "activation_scale_shape": list(As.shape),
            }
            return original(A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias, output_shape=output_shape)

        kernel.apply_scaled_mm = observed

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.calls.clear()
        self.layouts.clear()
        torch.accelerator.reset_peak_memory_stats()
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        assert set(self.calls) == set(self.kernels), "A configured FP8 layer did not execute its GEMM"
        sharded = sum(isinstance(parameter, DTensor) for parameter in self.pipeline.transformer.parameters())
        assert (sharded > 0) == self.od_config.parallel_config.use_hsdp
        rank = torch.distributed.get_rank()
        record = {
            "request_id": req.request_id,
            "rank": rank,
            "pid": os.getpid(),
            "kernels": self.kernels,
            "calls": self.calls,
            "layouts": self.layouts,
            "sharded_parameters_after_request": sharded,
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
        }
        with (Path(os.environ["FP8_PROBE_DIR"]) / f"rank-{rank}.jsonl").open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
