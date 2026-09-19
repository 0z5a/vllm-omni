"""Controlled E2E replay: fix OmniGen2's global posterior RNG on every worker."""

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

logger = init_logger(__name__)


class ReferenceRngRunner(DiffusionModelRunner):
    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        seed = req.sampling_params.seed
        assert seed is not None
        torch.manual_seed(seed)
        logger.info("Controlled posterior RNG seed=%d", seed)
        return super().execute_model(req, kv_prefetch_job, diffusion_kv_metadata)
