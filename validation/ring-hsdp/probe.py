# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Untimed Ring attention and HSDP residency evidence."""

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypedDict

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class AttentionRecord(TypedDict):
    calls: int
    strategy: str
    q_shape: list[int]
    output_shape: list[int]


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


class ProbeRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, ZImagePipeline)
        self.initial_fsdp_modules = sum(
            isinstance(module, FSDPModule) for module in self.pipeline.transformer.modules()
        )
        self.attention: dict[str, AttentionRecord] = {}
        self.positions: list[dict[str, object]] = []
        self.pipeline.transformer.layers[0].attention.register_forward_pre_hook(self.position_hook, with_kwargs=True)
        for name, module in self.pipeline.transformer.named_modules():
            if isinstance(module, Attention):
                module.register_forward_hook(self.attention_hook(name))

    def position_hook(self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        del module, args
        record = {}
        for name in ("attention_mask", "cos", "sin"):
            tensor = kwargs[name]
            assert isinstance(tensor, torch.Tensor)
            record[name] = {"shape": list(tensor.shape), "hash": digest(tensor)}
        self.positions.append(record)

    def attention_hook(self, name: str) -> Callable[[Attention, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def observed(module: Attention, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            previous = self.attention.get(name)
            self.attention[name] = {
                "calls": previous["calls"] + 1 if previous else 1,
                "strategy": module._get_active_parallel_strategy().name,
                "q_shape": list(args[0].shape),
                "output_shape": list(output.shape),
            }

        return observed

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.attention.clear()
        self.positions.clear()
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        config = self.od_config.parallel_config
        group = get_sp_group()
        assert group.ring_world_size == 2 and config.ring_degree == 2
        assert self.attention and {row["strategy"] for row in self.attention.values()} == {"ring"}
        transformer = self.pipeline.transformer
        sharded = sum(isinstance(parameter, DTensor) for parameter in transformer.parameters())
        wrapped = sum(isinstance(module, FSDPModule) for module in transformer.modules())
        assert (sharded > 0) == config.use_hsdp
        assert wrapped == self.initial_fsdp_modules
        record = {
            "request_id": req.request_id,
            "rank": torch.distributed.get_rank(),
            "hsdp": config.use_hsdp,
            "ring_size": group.ring_world_size,
            "sharded_parameters": sharded,
            "fsdp_modules": wrapped,
            "attention": self.attention,
            "positions": self.positions,
        }
        path = Path(os.environ["RING_HSDP_PROBE_DIR"]) / f"rank-{record['rank']}.jsonl"
        with path.open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
