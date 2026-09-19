# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporal causality and request isolation for the native whole-clip VAE."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def vae():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7723)
        return SeedVR2VAE(channels=(8, 16, 16, 16), groups=4, latent_channels=4).eval()


@pytest.mark.parametrize("frames", [1, 5, 9])
@torch.inference_mode()
def test_temporal_compression_and_reconstruction(vae, frames):
    video = torch.randn(1, 3, frames, 16, 24)
    posterior = vae.encode(video)
    assert posterior.mean.shape == (1, 4, (frames - 1) // 4 + 1, 2, 3)
    decoded = vae.decode(posterior.mode())
    assert decoded.shape == video.shape
    assert torch.isfinite(decoded).all()


@torch.inference_mode()
def test_future_frames_do_not_change_encoded_prefix(vae):
    video = torch.randn(1, 3, 9, 16, 24)
    changed = video.clone()
    changed[:, :, 5:] += 10
    before = vae.encode(video).parameters[:, :, :2]
    after = vae.encode(changed).parameters[:, :, :2]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@torch.inference_mode()
def test_future_latents_do_not_change_decoded_prefix(vae):
    latent = torch.randn(1, 4, 3, 2, 3)
    changed = latent.clone()
    changed[:, :, 2:] += 10
    before = vae.decode(latent)[:, :, :5]
    after = vae.decode(changed)[:, :, :5]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@torch.inference_mode()
def test_request_generator_and_intervening_clip_are_isolated(vae):
    video = torch.randn(1, 3, 5, 16, 24)
    state = torch.random.get_rng_state()
    first = vae.encode(video).sample(generator=torch.Generator().manual_seed(42))
    vae.decode(vae.encode(torch.zeros(1, 3, 9, 16, 24)).mode())
    second = vae.encode(video).sample(generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())
