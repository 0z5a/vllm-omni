"""Record real Helios TeaCache decisions without changing timed arms."""

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch

from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
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
        assert isinstance(self.pipeline, HeliosPipeline)
        transformer = self.pipeline.transformer
        self.decisions: list[dict[str, object]] = []
        self.block_calls = 0
        original_decision = transformer._teacache_should_compute

        def observe(state: TeaCacheState, modulated_input: torch.Tensor) -> bool:
            branch = "positive" if state is transformer._tea_cache_states["positive"] else "negative"
            row: dict[str, object] = {
                "branch": branch,
                "cnt_before": state.cnt,
                "residual_present": state.previous_residual is not None,
                "shape": list(modulated_input.shape),
            }
            decision = original_decision(state, modulated_input)
            row["compute"] = decision
            row["accumulated_after"] = state.accumulated_rel_l1_distance
            self.decisions.append(row)
            return decision

        transformer._teacache_should_compute = observe
        transformer.blocks[0].register_forward_pre_hook(self.count_block)

    def count_block(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.block_calls += 1

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.decisions.clear()
        self.block_calls = 0
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        transformer = self.pipeline.transformer
        assert isinstance(transformer, HeliosTransformer3DModel)
        computed = sum(bool(row["compute"]) or not bool(row["residual_present"]) for row in self.decisions)
        if transformer._tea_cache_config is not None:
            assert self.decisions
            assert self.block_calls == computed
        row = {
            "request_id": req.request_id,
            "rank": torch.distributed.get_rank(),
            "pid": os.getpid(),
            "decisions": self.decisions,
            "block_calls": self.block_calls,
            "cache_hits": len(self.decisions) - computed,
            "request_resets": sum(decision["cnt_before"] == 0 for decision in self.decisions),
        }
        target = Path(os.environ["HELIOS_TEACACHE_PROBE_DIR"]) / f"rank-{row['rank']}.jsonl"
        with target.open("a") as records:
            records.write(json.dumps(row) + "\n")
        return output
