# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical and frame-isolation gates for fused VAE normalization."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.triton_utils import tl, triton

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.seedvr2.frame_norm import _frame_offset
from vllm_omni.diffusion.models.seedvr2.vae import _frame_norm, _frame_norm_silu

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("frames,height,width", [(1, 8, 12), (5, 64, 96), (9, 17, 23)])
@pytest.mark.parametrize("offset", [0, 1000])
@torch.inference_mode()
def test_fused_norm_matches_per_frame_reference(frames, height, width, offset):
    generator = torch.Generator(device="cuda").manual_seed(7723)
    x = torch.randn(2, 32, frames, height, width, device="cuda", dtype=torch.float16, generator=generator) + offset
    norm = nn.GroupNorm(8, 32, eps=1e-6).to(device="cuda", dtype=torch.float16)
    expected = F.silu(_frame_norm(norm, x))
    actual = _frame_norm_silu(norm, x)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
    changed = x.clone()
    changed[:, :, -1] += 10
    after = _frame_norm_silu(norm, changed)
    torch.testing.assert_close(actual[:, :, :-1], after[:, :, :-1], atol=0, rtol=0)


@torch.inference_mode()
def test_constant_noncontiguous_frames_and_affine():
    x = torch.full((2, 5, 32, 8, 12), 3.0, device="cuda", dtype=torch.float16).transpose(1, 2)
    norm = nn.GroupNorm(8, 32, eps=1e-6).to(device="cuda", dtype=torch.float16)
    norm.weight.copy_(torch.linspace(0.5, 2, 32, device="cuda"))
    norm.bias.copy_(torch.linspace(-1, 1, 32, device="cuda"))
    torch.testing.assert_close(_frame_norm_silu(norm, x), F.silu(_frame_norm(norm, x)), atol=0.005, rtol=0.005)


@triton.jit
def _last_frame_offsets(output):
    groups = tl.arange(0, 2) * (109 * 32) + (109 * 32 - 1)
    offsets = _frame_offset(groups, tl.full((2,), 4 * 512 * 896 - 1, tl.int32), 512 * 896, 109, 128, 32)
    tl.store(output + tl.arange(0, 2), offsets)


def test_addressing_beyond_two_billion_elements():
    output = torch.empty(2, device="cuda", dtype=torch.int64)
    _last_frame_offsets[(1,)](output)
    clip_elements = 128 * 109 * 512 * 896
    assert output.tolist() == [clip_elements - 1, 2 * clip_elements - 1]
