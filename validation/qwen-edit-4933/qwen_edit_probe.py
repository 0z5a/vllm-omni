"""Untimed Qwen Edit conditioning fingerprints, separate from speed measurements."""

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


def fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class ConditioningProbeRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, QwenImageEditPipeline | QwenImageEditPlusPipeline)
        self.probe_request_id = "initialization"
        self.reference_index = 0
        original = self.pipeline._encode_vae_image

        def capture(image: torch.Tensor, generator: torch.Generator | list[torch.Generator] | None) -> torch.Tensor:
            latent = original(image, generator)
            row = {
                "request_id": self.probe_request_id,
                "reference_index": self.reference_index,
                "rank": torch.distributed.get_rank(),
                "input_shape": list(image.shape),
                "latent_shape": list(latent.shape),
                "dtype": str(latent.dtype),
                "input_sha256": fingerprint(image),
                "latent_sha256": fingerprint(latent),
            }
            self.reference_index += 1
            with (Path(os.environ["QWEN_PROBE_DIR"]) / f"rank-{row['rank']}-conditioning.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            return latent

        self.pipeline._encode_vae_image = capture

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        self.probe_request_id = req.request_id
        self.reference_index = 0
        output = super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
        assert self.reference_index > 0
        return output
