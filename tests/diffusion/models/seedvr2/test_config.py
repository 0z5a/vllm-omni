# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unsupported SeedVR2 modes fail before allocating process resources."""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"dtype": torch.bfloat16}, "dtype=float16"),
        ({"enforce_eager": False}, "enforce_eager=True"),
        ({"cache_backend": "tea_cache"}, "single Euler step"),
        ({"vae_use_tiling": True}, "slicing or tiling"),
        ({"vae_use_slicing": True}, "slicing or tiling"),
        ({"enable_cpu_offload": True}, "CPU offload"),
        ({"enable_layerwise_offload": True}, "CPU offload"),
        ({"parallel_config": {"cfg_parallel_size": 2}}, "cfg_parallel_size=1"),
        ({"parallel_config": {"tensor_parallel_size": 2}}, "tensor_parallel_size=1"),
        ({"parallel_config": {"pipeline_parallel_size": 2}}, "pipeline_parallel_size=1"),
        ({"parallel_config": {"vae_patch_parallel_size": 2}}, "vae_patch_parallel_size=1"),
        ({"parallel_config": {"ulysses_degree": 2}}, "ulysses_degree=1"),
        ({"parallel_config": {"ring_degree": 2}}, "ring_degree=1"),
        ({"parallel_config": {"allgather_degree": 2}}, "allgather_degree=1"),
        ({"lora_path": "unsupported-adapter"}, "LoRA"),
    ],
)
def test_unsupported_config_fails_before_hooks(monkeypatch, overrides, message):
    options = {"model_class_name": "SeedVR2Pipeline", "dtype": torch.float16, "enforce_eager": True}
    options.update(overrides)
    config = OmniDiffusionConfig(**options)

    def forbidden_startup(self, config):
        pytest.fail("unsupported SeedVR2 configuration reached process hooks")

    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", forbidden_startup)
    with pytest.raises(ValueError, match=message):
        DiffusionEngine(config)
