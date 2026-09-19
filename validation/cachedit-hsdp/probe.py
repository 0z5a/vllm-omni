"""Untimed DBCache block execution and request reset evidence."""

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from cache_dit.caching.cache_blocks.pattern_base import CachedBlocks_Pattern_Base
from torch.distributed.tensor import DTensor

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
        wrappers = [m for m in self.pipeline.transformer.modules() if isinstance(m, CachedBlocks_Pattern_Base)]
        assert len(wrappers) == 1
        self.wrapper = wrappers[0]
        self.steps: list[dict[str, object]] = []
        self.first_calls = 0
        self.middle_calls = 0
        self.wrapper.register_forward_pre_hook(self.before_step)
        self.wrapper.transformer_blocks[0].register_forward_pre_hook(self.count_first)
        self.wrapper.transformer_blocks[1].register_forward_pre_hook(self.count_middle)

    def before_step(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.wrapper.context_manager.set_context(self.wrapper.cache_context)
        context = self.wrapper.context_manager.get_context()
        buffers = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "dtensor": isinstance(value, DTensor)}
            for key, value in context.buffers.items()
            if isinstance(value, torch.Tensor)
        }
        if not self.steps:
            assert context.executed_steps == 0 and not context.buffers and not context.cached_steps
        self.steps.append({"executed_steps": context.executed_steps, "buffers": buffers})

    def count_first(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.first_calls += 1

    def count_middle(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.middle_calls += 1

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.steps.clear()
        self.first_calls = self.middle_calls = 0
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        self.wrapper.context_manager.set_context(self.wrapper.cache_context)
        context = self.wrapper.context_manager.get_context()
        hits = list(context.cached_steps)
        assert self.first_calls == len(self.steps)
        assert self.middle_calls == self.first_calls - len(hits)
        sharded = sum(isinstance(p, DTensor) for p in self.pipeline.transformer.parameters())
        if self.od_config.parallel_config.use_hsdp:
            assert sharded > 0
        if context.cache_config.max_cached_steps == 0:
            assert not hits
        record = {
            "request_id": req.request_id,
            "pid": os.getpid(),
            "rank": torch.distributed.get_rank(),
            "steps": self.steps,
            "cached_steps": hits,
            "first_block_calls": self.first_calls,
            "middle_block_calls": self.middle_calls,
            "sharded_parameters_after_request": sharded,
        }
        path = Path(os.environ["COMPAT_PROBE_DIR"]) / f"rank-{record['rank']}.jsonl"
        with path.open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
