# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject window-SP for unrelated models before workers or hooks start."""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("architecture", [None, "FluxPipeline", "WanPipeline", "DiffusersAdapterPipeline"])
def test_window_parallel_identity_fails_before_startup(monkeypatch, architecture):
    config = OmniDiffusionConfig(model_class_name=architecture, parallel_config={"window_parallel_size": 2})

    def forbidden_startup(self, config):
        pytest.fail("process hooks must not start for an unsupported model")

    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", forbidden_startup)
    with pytest.raises(ValueError, match="requires SeedVR2Pipeline"):
        DiffusionEngine(config)


@pytest.mark.parametrize("architecture,degree", [("FluxPipeline", 1), (None, 1), ("SeedVR2Pipeline", 2)])
def test_supported_identity_reaches_startup(monkeypatch, architecture, degree):
    config = OmniDiffusionConfig(
        model_class_name=architecture,
        dtype=torch.float16,
        enforce_eager=True,
        parallel_config={"window_parallel_size": degree},
    )

    def startup_sentinel(self, config):
        raise RuntimeError("startup reached")

    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", startup_sentinel)
    with pytest.raises(RuntimeError, match="startup reached"):
        DiffusionEngine(config)
