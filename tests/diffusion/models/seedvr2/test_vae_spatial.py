# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiled convolution and unequal-height VAE shard parity."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.models.seedvr2.vae_spatial import tiled_convolution

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("stride,padding", [((1, 1, 1), (0, 1, 1)), ((2, 2, 2), (0, 0, 0))])
@torch.inference_mode()
def test_spatial_tiles_match_convolution_at_boundaries(stride, padding):
    torch.manual_seed(42)
    conv = nn.Conv3d(3, 4, 3, stride=stride, padding=padding)
    x = torch.randn(2, 3, 7, 35, 27)
    expected = conv(x)
    actual = tiled_convolution(conv, x, rows=7)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@torch.inference_mode()
def _spatial_worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size)
    torch.manual_seed(7723)
    vae = SeedVR2VAE(channels=(8, 16, 16, 16), groups=4, latent_channels=4).eval()
    for frames, height in [(1, 40), (13, 40), (5, 16)]:
        x = torch.randn(1, 3, frames, height, 24)
        vae.use_tiling = False
        reference = vae.encode(x).parameters
        latent = reference[:, :4]
        expected = vae.decode(latent)
        vae.use_tiling = True
        vae._spatial_group = dist.group.WORLD
        actual = vae.encode(x).parameters
        decoded = vae.decode(latent)
        torch.testing.assert_close(actual, reference, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(decoded, expected, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(vae.encode(x).parameters, actual, atol=0, rtol=0)
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_unequal_shards_preserve_whole_frame_semantics(tmp_path, size):
    mp.spawn(_spatial_worker, args=(size, f"file://{tmp_path / 'rendezvous'}"), nprocs=size, join=True)
