"""Intrusive NextStep phase profiling in a separate, untimed process."""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

import torch
from transformers.cache_utils import StaticCache

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.models.nextstep_1_1.pipeline_nextstep_1_1 import NextStep11Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class StageProfileRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        NextStep11Pipeline._PROFILER_TARGETS = ["model.forward", "model.image_head.sample", "vae.decode"]
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, NextStep11Pipeline)
        self.calls = 0
        self.pipeline.model.register_forward_pre_hook(self.record_shape, with_kwargs=True)

    def record_shape(self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        tensor = cast(torch.Tensor, kwargs["inputs_embeds"] if "inputs_embeds" in kwargs else kwargs["input_ids"])
        cache = cast(StaticCache, kwargs["past_key_values"])
        mask = cast(torch.Tensor, kwargs["attention_mask"])
        if self.calls < 2 or self.calls % 256 == 0:
            print(
                json.dumps(
                    {
                        "profile_rank": torch.distributed.get_rank(),
                        "call": self.calls,
                        "query_shape": list(tensor.shape),
                        "kv_before": int(cache.get_seq_length()),
                        "mask_shape": list(mask.shape),
                        "cache_capacity": cache.get_max_cache_shape(),
                    }
                ),
                flush=True,
            )
        self.calls += 1

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.calls = 0
        return super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
