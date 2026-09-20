"""Untimed real CFG branch and LoRA activation evidence for Qwen Edit."""

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from lora_probe import ProbeRunner as LoRAProbeRunner

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob


class CFGProbeRunner(LoRAProbeRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, QwenImageEditPlusPipeline)
        self.conditioning: list[str] = []
        self.pipeline.transformer.register_forward_pre_hook(self.before_transformer, with_kwargs=True)

    def before_transformer(self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        embeddings = kwargs["encoder_hidden_states"]
        assert isinstance(embeddings, torch.Tensor)
        assert self.pipeline.transformer.do_true_cfg
        digest = hashlib.sha256(embeddings.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        self.conditioning.append(digest)

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.conditioning.clear()
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        size = get_classifier_free_guidance_world_size()
        rank = get_classifier_free_guidance_rank()
        assert size == self.od_config.parallel_config.cfg_parallel_size
        assert len(self.conditioning) == req.sampling_params.num_inference_steps * (2 if size == 1 else 1)
        assert len(set(self.conditioning)) == (2 if size == 1 else 1)
        record = {
            "request_id": req.request_id,
            "cfg_rank": rank,
            "cfg_world_size": size,
            "conditioning": self.conditioning,
        }
        with (Path(os.environ["LORA_PROBE_DIR"]) / f"cfg-rank-{rank}.jsonl").open("a") as destination:
            destination.write(json.dumps(record) + "\n")
        return output
