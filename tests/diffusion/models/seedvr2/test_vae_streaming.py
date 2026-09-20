# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal temporal partitions preserve full-frame VAE statistics."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("frames", [5, 6, 9, 10, 13, 17])
@pytest.mark.parametrize("chunk_size", [4, 8])
@torch.inference_mode()
def test_streaming_matches_whole_clip_and_resets(frames, chunk_size):
    torch.manual_seed(7723)
    vae = SeedVR2VAE(channels=(8, 16, 16, 16), groups=4, latent_channels=4).eval()
    video = torch.randn(1, 3, frames, 16, 24)
    full = vae.encode(video).parameters
    streamed = vae.encode(video, chunk_size=chunk_size).parameters
    torch.testing.assert_close(streamed, full, atol=2e-5, rtol=2e-5)
    latent = vae.encode(video).mode()
    expected = vae.decode(latent)
    actual = vae.decode(latent, chunk_size=chunk_size // 4)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    vae.encode(video.flip(2), chunk_size=chunk_size)
    vae.decode(latent.flip(2), chunk_size=chunk_size // 4)
    torch.testing.assert_close(vae.encode(video, chunk_size=chunk_size).parameters, streamed, atol=0, rtol=0)
    torch.testing.assert_close(vae.decode(latent, chunk_size=chunk_size // 4), actual, atol=0, rtol=0)


@torch.inference_mode()
def test_tiling_preserves_short_clip_path():
    vae = SeedVR2VAE(channels=(8, 16, 16, 16), groups=4, latent_channels=4).eval()
    video = torch.randn(1, 3, 5, 16, 24)
    latent = vae.encode(video).mode()
    decoded = vae.decode(latent)
    vae.use_tiling = True
    torch.testing.assert_close(vae.encode(video).mode(), latent, atol=0, rtol=0)
    torch.testing.assert_close(vae.decode(latent), decoded, atol=0, rtol=0)
