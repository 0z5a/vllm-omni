"""Untimed ZImage TeaCache/HSDP evidence; never enable for timing arms."""

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class ObservedTeaCacheHook(TeaCacheHook):
    def __init__(self, config: TeaCacheConfig) -> None:
        super().__init__(config)
        self.decisions: list[dict[str, int | bool]] = []

    def _should_compute_full_transformer(
        self, state: TeaCacheState, modulated_inp: torch.Tensor
    ) -> bool:
        step = state.cnt
        residual_present = state.previous_residual is not None
        decision = super()._should_compute_full_transformer(state, modulated_inp)
        self.decisions.append(
            {"step": step, "compute": decision, "residual_present": residual_present}
        )
        return decision


class ProbeRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]]
        | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, ZImagePipeline)
        registry = HookRegistry.check_if_exists_or_initialize(self.pipeline.transformer)
        original = registry.get_hook("teacache")
        assert isinstance(original, TeaCacheHook)
        self.observed_cache = ObservedTeaCacheHook(original.config)
        registry.register_hook("teacache", self.observed_cache)
        self.block_calls = 0
        self.pipeline.transformer.layers[0].register_forward_pre_hook(self.count_block)

    def count_block(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.block_calls += 1

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.observed_cache.decisions.clear()
        self.block_calls = 0
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        decisions = self.observed_cache.decisions
        assert decisions and decisions[0] == {
            "step": 0,
            "compute": True,
            "residual_present": False,
        }
        computed = sum(
            row["compute"] or not row["residual_present"] for row in decisions
        )
        assert self.block_calls == computed
        sharded = sum(
            isinstance(parameter, DTensor)
            for parameter in self.pipeline.transformer.parameters()
        )
        if self.od_config.parallel_config.use_hsdp:
            assert sharded > 0
        record = {
            "request_id": req.request_id,
            "pid": os.getpid(),
            "rank": torch.distributed.get_rank(),
            "decisions": decisions,
            "block_calls": self.block_calls,
            "cache_hits": len(decisions) - computed,
            "sharded_parameters_after_request": sharded,
        }
        path = Path(os.environ["COMPAT_PROBE_DIR"]) / f"rank-{record['rank']}.jsonl"
        with path.open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
