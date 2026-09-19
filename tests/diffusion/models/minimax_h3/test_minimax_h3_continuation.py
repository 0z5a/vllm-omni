# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.continuation import (
    diffuse_continuation,
    plan_continuation_windows,
    resolve_continuation,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("total,window", [(1807, 277), (1807, 243), (124, 277)])
def test_continuation_retains_prefix_and_global_stereo_timeline(total, window):
    audio_t = round(total * 40 / 24)
    source = torch.arange(2 * audio_t * 32).reshape(2, audio_t, 32).float()
    calls = []

    def sample(**kw):
        calls.append(kw)
        t = kw["latent_t"]
        # Different constants expose replacement of an old prefix or an
        # accidental append of the hidden overlap.
        video = torch.full((1, 24, t, 2, 2), float(len(calls)))
        audio = kw["locked_audio_rows"].reshape(2, kw["audio_t"], 32).permute(0, 2, 1)
        if len(calls) > 1:
            assert torch.all(kw["visual_condition"] == len(calls) - 1)
            assert kw["ref_blocks"][-1]["kind"] == "latent_guide"
        return video, audio

    kwargs = dict(
        num_frames=total,
        latent_t=(total - 5) // 17 * 5 + 2,
        latent_h=2,
        latent_w=2,
        audio_t=audio_t,
        locked_audio_rows=source.reshape(-1, 32),
    )
    video, audio = diffuse_continuation(sample, kwargs, window_frames=window, overlap_frames=22)
    torch.testing.assert_close(audio, source.permute(0, 2, 1))
    plan = plan_continuation_windows(total, window, 22)
    offset = 0
    for i, part in enumerate(plan):
        new_frames = part.end - part.start - part.overlap
        count = (new_frames - 5) // 17 * 5 + 2 if i == 0 else new_frames // 17 * 5
        assert torch.all(video[:, :, offset : offset + count] == i + 1)
        offset += count
    assert offset == video.shape[2] == kwargs["latent_t"]


def test_latent_guide_shares_target_clock_and_remains_condition_only():
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=12,
        latent_h=4,
        latent_w=4,
        audio_t=68,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "latent_guide", "latent_t": 7, "latent_h": 4, "latent_w": 4, "ref_audio_t": 37},
        ],
    )
    pos = packed["img_position_ids"]
    visual = packed["img_pos"]
    target = visual[packed["update_mask"]]
    guide = visual[~packed["update_mask"]][4:]
    torch.testing.assert_close(pos[guide], pos[target[:28]])
    audio = packed["audio_pos"]
    guide_audio = audio[~packed["audio_update_mask"]].reshape(2, 37)
    target_audio = audio[packed["audio_update_mask"]].reshape(2, 68)
    torch.testing.assert_close(pos[guide_audio], pos[target_audio[:, :37]])
    assert pos[target[0], 0] == 4  # text + original image, without a guide time shift


@pytest.mark.parametrize(
    "extra,step",
    [
        ({"long_video_mode": "bad"}, False),
        ({"long_video_mode": "continuation"}, True),
        ({"long_video_mode": "continuation", "continuation_overlap_frames": 23}, False),
        ({"long_video_mode": "continuation", "continuation_window_frames": True}, False),
        ({"long_video_mode": "continuation", "continuation_overlap_frames": 277}, False),
    ],
)
def test_invalid_continuation_options(extra, step):
    with pytest.raises(OmniClientError):
        resolve_continuation(extra, task="ref2va", step_execution=step)
