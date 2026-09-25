# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check the upstream Z-Image fusion through Ming's conditioning adapter."""

import os

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    set_forward_context,
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image import z_image_transformer

pytestmark = [pytest.mark.cuda, pytest.mark.diffusion]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ming_layered_forward_uses_fusion_and_matches_eager(monkeypatch):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29529")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        try:
            model = (
                MingImageTransformer2DModel(
                    in_channels=4,
                    dim=128,
                    n_layers=1,
                    n_refiner_layers=1,
                    n_heads=2,
                    n_kv_heads=2,
                    cap_feat_dim=128,
                    axes_dims=[16, 24, 24],
                    axes_lens=[1024, 512, 512],
                    alignment_padding_mode="learned",
                    multi_frame_output=True,
                )
                .to(device="cuda", dtype=torch.bfloat16)
                .eval()
            )
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "norm" in name and param.ndim == 1:
                        param.fill_(1)
                    else:
                        param.uniform_(-0.02, 0.02)
            x = [torch.randn(4, 1, 16, 16, device="cuda", dtype=torch.bfloat16)]
            t = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
            cap = [torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)]
            ref = torch.randn(1, 4, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
            direct = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)
            calls = 0
            original = z_image_transformer.fused_qk_norm_rope

            def counted_fusion(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            monkeypatch.setattr(z_image_transformer, "fused_qk_norm_rope", counted_fusion)
            with set_forward_context(omni_diffusion_config=OmniDiffusionConfig(model=None)):
                set_forward_context_ref_latent(ref)
                set_forward_context_direct_condition(direct)
                with torch.inference_mode():
                    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
                    eager = model(x, t, cap)[0][0]
                    assert calls == 0
                    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
                    fused = model(x, t, cap)[0][0]
            assert calls > 0
            torch.testing.assert_close(fused, eager, atol=0.05, rtol=0.05)
        finally:
            cleanup_dist_env_and_memory()
