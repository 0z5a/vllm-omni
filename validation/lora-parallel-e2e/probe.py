"""Untimed attention layout and parameter lifecycle evidence alongside LoRA switches."""

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypedDict

import torch
from lora_probe import ProbeRunner as LoRAProbeRunner
from lora_probe import digest
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob


class AttentionRecord(TypedDict):
    calls: int
    strategy: str
    q_shape: list[int]
    k_shape: list[int]
    v_shape: list[int]
    output_shape: list[int]


class ProbeRunner(LoRAProbeRunner):
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
        self.sources = {}
        for cls in (ZImagePipeline, LoRAProbeRunner, Attention):
            path = Path(inspect.getfile(cls)).resolve()
            self.sources[cls.__name__] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.attention: dict[str, AttentionRecord] = {}
        self.positions: list[dict[str, object]] = []
        self.pipeline.transformer.layers[0].attention.register_forward_pre_hook(self.position_hook, with_kwargs=True)
        for name, module in self.pipeline.transformer.named_modules():
            if isinstance(module, Attention):
                module.register_forward_hook(self.attention_hook(name))

    def position_hook(self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        record = {}
        for name in ("attention_mask", "cos", "sin"):
            tensor = kwargs[name]
            assert isinstance(tensor, torch.Tensor)
            record[name] = {
                "shape": list(tensor.shape),
                "hash": digest(tensor),
                "halves": [digest(chunk) for chunk in tensor.chunk(2, dim=1)],
            }
        self.positions.append(record)

    def attention_hook(self, name: str) -> Callable[[Attention, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def observed(module: Attention, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            strategy = module._get_active_parallel_strategy()
            previous = self.attention.get(name)
            self.attention[name] = {
                "calls": previous["calls"] + 1 if previous else 1,
                "strategy": strategy.name,
                "q_shape": list(args[0].shape),
                "k_shape": list(args[1].shape),
                "v_shape": list(args[2].shape),
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
        torch.accelerator.reset_peak_memory_stats()
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        config = self.od_config.parallel_config
        group = get_sp_group()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size == config.tensor_parallel_size
        assert group.ulysses_world_size == config.ulysses_degree
        assert group.ring_world_size == config.ring_degree
        assert self.attention
        strategies = {row["strategy"] for row in self.attention.values()}
        if config.ulysses_degree > 1:
            assert "ulysses" in strategies
        if config.ring_degree > 1:
            assert "ring" in strategies
        transformer = self.pipeline.transformer
        sharded = sum(isinstance(parameter, DTensor) for parameter in transformer.parameters())
        wrapped = sum(isinstance(module, FSDPModule) for module in transformer.modules())
        assert (sharded > 0) == config.use_hsdp and (wrapped > 0) == config.use_hsdp
        assert wrapped == self.initial_fsdp_modules
        record = {
            "request_id": req.request_id,
            "rank": torch.distributed.get_rank(),
            "tp_size": tp_size,
            "ulysses_size": group.ulysses_world_size,
            "ring_size": group.ring_world_size,
            "sharded_parameters_after_request": sharded,
            "fsdp_modules": wrapped,
            "initial_fsdp_modules": self.initial_fsdp_modules,
            "sources": self.sources,
            "attention": self.attention,
            "positions": self.positions,
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
        }
        path = Path(os.environ["LORA_PROBE_DIR"]) / f"parallel-rank-{record['rank']}.jsonl"
        with path.open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
