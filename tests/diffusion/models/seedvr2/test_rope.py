# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""RoPE geometry and host-synchronization regressions."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.rope import NaMMRotaryEmbedding3d

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("shape,text_len", [((2, 8, 14), 58), ((3, 9, 15), 64), ((1, 1, 1), 1)])
def test_window_positions_without_scalar_reads(shape, text_len):
    rope = NaMMRotaryEmbedding3d(128)
    t, h, w = torch.meshgrid(
        torch.arange(text_len, text_len + shape[0]),
        torch.arange(shape[1]),
        torch.arange(shape[2]),
        indexing="ij",
    )
    positions = torch.stack((t, h, w), dim=-1).reshape(-1, 3)
    expected = (positions[..., None] * rope.freqs).repeat_interleave(2, dim=-1).flatten(1)
    # Exercise cold and warm caches, with a different text length in between.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = rope.window_freqs(shape, text_len, device="cpu", dtype=torch.float32)
        rope.text_freqs(text_len + 1, device="cpu", dtype=torch.float32)
        repeated = rope.window_freqs(shape, text_len, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    assert "aten::_local_scalar_dense" not in {event.key for event in profile.key_averages()}
