# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Existing SP configuration remains available to supported model families."""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "architecture,degree", [("FluxPipeline", 2), ("WanPipeline", 2), (None, 1), ("SeedVR2Pipeline", 2)]
)
def test_supported_identity_reaches_startup(monkeypatch, architecture, degree):
    config = OmniDiffusionConfig(
        model_class_name=architecture,
        dtype=torch.float16,
        enforce_eager=True,
        parallel_config={"ulysses_degree": degree},
    )

    def startup_sentinel(self, config):
        raise RuntimeError("startup reached")

    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", startup_sentinel)
    with pytest.raises(RuntimeError, match="startup reached"):
        DiffusionEngine(config)
