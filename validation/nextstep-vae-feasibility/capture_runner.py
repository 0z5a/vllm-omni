"""Capture one real NextStep AR latent before the vendored VAE decode."""

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import torch

from vllm_omni.diffusion.models.nextstep_1_1.pipeline_nextstep_1_1 import NextStep11Pipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class LatentCaptureRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, NextStep11Pipeline)
        capture_path = Path(os.environ["NEXTSTEP_LATENT_PATH"])

        def capture(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
            if torch.distributed.get_rank() != 0 or capture_path.exists():
                return
            latent = cast(torch.Tensor, args[0]).detach().to(device="cpu", dtype=torch.float32)
            temporary = capture_path.with_suffix(".tmp")
            torch.save(latent, temporary)
            temporary.replace(capture_path)

        self.pipeline.vae.decoder.register_forward_pre_hook(capture)
