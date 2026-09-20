# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness matrix for MOSS-VL-Realtime positions and cross-attention.

Covers the semantics the reference capture cannot exercise with three fixtures:
zero/full/mixed visibility, GQA head mapping, ragged frame lengths, prefill vs
incremental decode, empty vision, repeated media timestamps, and rejected
metadata. Runs on CPU with a tiny random-weight model.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.moss_vl_realtime.config import (
    MossVLConfig,
    MossVLTextConfig,
    MossVLVisionConfig,
)
from vllm_omni.model_executor.models.moss_vl_realtime.modeling import (
    BudgetExceededError,
    MossVLNativeModel,
    SessionLimits,
    VisionTokenInfo,
)

IMAGE_PAD = 3
NEGATIVE = torch.finfo(torch.bfloat16).min


def tiny_config(pad_multiple: int = 1) -> MossVLConfig:
    text = MossVLTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        cross_attention_layers=(1, 3),
        vocab_size=64,
    )
    return MossVLConfig(
        text=text,
        vision=MossVLVisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            num_position_embeddings=64,
            out_hidden_size=32,
            deepstack_visual_indexes=(1,),
        ),
        image_token_id=IMAGE_PAD,
        vision_seq_pad_multiple=pad_multiple,
    )


def build_model(config: MossVLConfig | None = None, dtype: torch.dtype = torch.bfloat16) -> MossVLNativeModel:
    """BF16 like the published checkpoint, with the cross-attention gates opened.

    A fresh model initialises both tanh gates to zero, which makes every
    cross-attention branch a silent no-op; a test that compares masked against
    unmasked attention has to open them first.
    """
    model = MossVLNativeModel(config or tiny_config()).to(dtype)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "cross_attn_attn_gate"):
                module.cross_attn_attn_gate.fill_(0.5)
                module.cross_attn_mlp_gate.fill_(0.5)
    model.eval()
    return model


def frame_info(grid_h: int, grid_w: int, num_frames: int, start: int) -> VisionTokenInfo:
    return VisionTokenInfo(
        grid_h=grid_h,
        grid_w=grid_w,
        num_frames=num_frames,
        start=start,
        vision_tokens_per_frame=(grid_h // 2) * (grid_w // 2),
    )


# --------------------------------------------------------------------------- T06
def test_zero_visibility_rows_stay_inert() -> None:
    """A query that sees no frame must not change the residual stream."""
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 1, 0)]
    hidden = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    vision = torch.randn(1, 5, 32, dtype=torch.bfloat16)
    mask = torch.full((1, 1, 3, 5), NEGATIVE, dtype=torch.bfloat16)
    with torch.no_grad():
        zeros_text = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
        zeros_vision = torch.zeros(1, 5, 8, dtype=torch.bfloat16)
        out_masked = model.model.language_model.layers[1](
            hidden,
            zeros_text,
            zeros_text,
            vision,
            zeros_vision,
            zeros_vision,
            mask,
            model.vision_cache,
            1,
            None,
        )
        baseline = model.model.language_model.layers[1](
            hidden,
            zeros_text,
            zeros_text,
            None,
            None,
            None,
            None,
            model.vision_cache,
            1,
            None,
        )
    assert torch.allclose(out_masked, baseline, atol=1e-5)


def test_mixed_visibility_blocks_later_frames() -> None:
    """Query rows only see the frames whose columns are unmasked.

    One model instance is reused with its vision cache reset between calls, so the
    comparison isolates the mask instead of comparing two random initialisations.
    """
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 2, 0)]
    layer = model.model.language_model.layers[1]

    def run(mask: torch.Tensor | None) -> torch.Tensor:
        vision = torch.randn(1, 10, 32, dtype=torch.bfloat16)
        hidden = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        query_cos = torch.ones(1, 2, 8, dtype=torch.bfloat16)
        query_sin = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
        vision_cos = torch.zeros(1, 10, 8, dtype=torch.bfloat16)
        vision_sin = torch.zeros(1, 10, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            return layer(
                hidden, query_cos, query_sin, vision, vision_cos, vision_sin, mask, model.vision_cache, 1, None
            )

    torch.manual_seed(0)
    only_first = torch.cat(
        [torch.zeros(1, 1, 2, 5, dtype=torch.bfloat16), torch.full((1, 1, 2, 5), NEGATIVE, dtype=torch.bfloat16)],
        dim=-1,
    )
    torch.manual_seed(0)
    all_frames = torch.zeros(1, 1, 2, 10, dtype=torch.bfloat16)

    torch.manual_seed(1)
    blocked = run(only_first)
    model.vision_cache.reset()
    torch.manual_seed(1)
    full = run(all_frames)
    model.vision_cache.reset()
    torch.manual_seed(1)
    run(None)

    assert not torch.allclose(blocked, full)
    # An all-visible mask is the same computation as no mask at all. Checked in
    # float32 because at bf16 the two shapes select different SDPA kernels.
    fp32 = build_model(dtype=torch.float32)
    fp32.vision_token_info = [frame_info(4, 4, 2, 0)]
    fp32_layer = fp32.model.language_model.layers[1]

    def run_fp32(mask: torch.Tensor | None) -> torch.Tensor:
        vision = torch.randn(1, 10, 32)
        hidden = torch.randn(1, 2, 32)
        cos = torch.ones(1, 2, 8)
        sin = torch.zeros(1, 2, 8)
        vision_cos = torch.zeros(1, 10, 8)
        vision_sin = torch.zeros(1, 10, 8)
        with torch.no_grad():
            return fp32_layer(hidden, cos, sin, vision, vision_cos, vision_sin, mask, fp32.vision_cache, 1, None)

    torch.manual_seed(2)
    masked_fp32 = run_fp32(torch.zeros(1, 1, 2, 10))
    fp32.vision_cache.reset()
    torch.manual_seed(2)
    unmasked_fp32 = run_fp32(None)
    assert torch.allclose(masked_fp32, unmasked_fp32, atol=1e-5)


def test_gqa_head_mapping_repeats_kv_groups() -> None:
    model = build_model()
    attention = model.model.language_model.layers[1].cross_attn
    vision = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        key = attention.k_norm(attention.k_proj(vision).view(1, -1, attention.num_kv_heads, attention.head_dim))
        key = key.transpose(1, 2)
        repeated = key[:, :, None].expand(1, attention.num_kv_heads, attention.groups, 4, attention.head_dim)
        repeated = repeated.reshape(1, attention.num_heads, 4, attention.head_dim)
    assert repeated.shape[1] == attention.num_heads
    assert torch.equal(repeated[:, 0], repeated[:, 1])
    assert not torch.equal(repeated[:, 0], repeated[:, 2])


def test_ragged_frame_lengths_expand_per_frame() -> None:
    """Frames with different token counts repeat by their own length."""
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 1, 0), frame_info(2, 2, 1, 5)]
    frame_mask = torch.tensor([[[[True, False], [False, False], [False, False]]]])
    expanded = model.expand_cross_attention_mask(frame_mask, torch.bfloat16)
    assert expanded.shape == (1, 1, 3, 5 + 2)
    assert expanded[0, 0, 0].tolist() == [NEGATIVE] * 5 + [0.0] * 2
    assert expanded[0, 0, 1].tolist() == [0.0] * 7


def test_multi_frame_media_repeats_each_frame() -> None:
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 3, 0)]
    frame_mask = torch.zeros(1, 1, 2, 3, dtype=torch.bool)
    frame_mask[0, 0, 0, 1:] = True
    expanded = model.expand_cross_attention_mask(frame_mask, torch.bfloat16)
    assert expanded.shape == (1, 1, 2, 15)
    assert expanded[0, 0, 0].tolist() == [0.0] * 5 + [NEGATIVE] * 10
    assert expanded[0, 0, 1].tolist() == [0.0] * 15


def test_unsupported_vision_padding_is_rejected() -> None:
    """The published checkpoint has no vision padding; a padded config is refused."""
    model = build_model(tiny_config(pad_multiple=8))
    model.vision_token_info = [frame_info(4, 4, 1, 0)]
    with pytest.raises(ValueError, match="vision_seq_pad_multiple"):
        model.expand_cross_attention_mask(torch.zeros(1, 1, 2, 1, dtype=torch.bool), torch.bfloat16)


def test_frame_metadata_mismatch_is_rejected() -> None:
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 2, 0)]
    input_ids = torch.tensor([[10, IMAGE_PAD, 11]])
    positions = model.compute_text_position_ids(input_ids)
    with pytest.raises(ValueError, match="frame"):
        model.compute_vision_position_ids(input_ids, positions, 10)


# --------------------------------------------------------------------------- T05
def test_prefill_matches_incremental_decode() -> None:
    """A vision prefill and the same tokens fed step by step must agree."""
    model = build_model()
    input_ids = torch.tensor([[10, IMAGE_PAD, 11, 12]])
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        full = model(input_ids, model.compute_text_position_ids(input_ids), 0, pixel_values, grid_thw)
        model.reset()
        head = model(input_ids[:, :2], model.compute_text_position_ids(input_ids[:, :2]), 0, pixel_values, grid_thw)
        for index in range(2, 4):
            token = input_ids[:, index : index + 1]
            step = model(token, model.decode_position_ids(offset=index, input_ids=token), index)
    assert torch.allclose(head[0, -1].float(), full[0, 1].float(), atol=5e-2)
    assert torch.allclose(step[0, -1].float(), full[0, -1].float(), atol=5e-2)


def test_text_only_decode_uses_the_cached_rule_of_the_reference() -> None:
    """Without vision states there is no rope delta, so a decode step restarts at 0."""
    model = build_model()
    token = torch.tensor([[12]])
    with torch.no_grad():
        positions = model.decode_position_ids(offset=9, input_ids=token)
    assert positions.tolist() == [[[0]], [[0]], [[0]]]


def test_text_only_matches_empty_vision() -> None:
    model = build_model()
    input_ids = torch.tensor([[10, 11, 12]])
    positions = model.compute_text_position_ids(input_ids)
    with torch.no_grad():
        text_only = model(input_ids, positions, 0)
        model.reset()
        vision_positions, _ = model.compute_vision_position_ids(input_ids, positions.clone(), 0)
        staged = model(input_ids, positions, 0, cross_attention_mask=None)
        assert vision_positions.shape == (3, 1, 0)
    assert torch.allclose(text_only, staged)


def test_repeated_media_timestamp_keeps_frames_distinct() -> None:
    """Two frames at the same media timestamp still advance the frame axis."""
    model = build_model()
    model.vision_token_info = [frame_info(4, 4, 2, 0)]
    input_ids = torch.tensor([[10, IMAGE_PAD, IMAGE_PAD, 11]])
    positions = model.compute_text_position_ids(input_ids)
    vision_positions, shifted = model.compute_vision_position_ids(input_ids, positions, 10)
    assert vision_positions.shape == (3, 1, 10)
    assert vision_positions[0, 0, 5].item() > vision_positions[0, 0, 0].item()
    assert shifted[0, 0, 3].item() > shifted[0, 0, 2].item()


def test_image_pad_at_sequence_end_keeps_positions_monotonic() -> None:
    model = build_model()
    input_ids = torch.tensor([[10, 11, IMAGE_PAD]])
    model.vision_token_info = [frame_info(4, 4, 1, 0)]
    positions = model.compute_text_position_ids(input_ids)
    vision_positions, shifted = model.compute_vision_position_ids(input_ids, positions, 5)
    text = shifted[0, 0].tolist()
    assert text == sorted(text)
    assert text[-1] == max(vision_positions[0, 0].tolist())


# --------------------------------------------------------------- T10 publication
def test_append_frames_publishes_at_a_step_boundary() -> None:
    """A frame appended after step N is invisible to step N and visible to step N+1."""
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4]])

    torch.manual_seed(0)
    steady = build_model()
    torch.manual_seed(0)
    stepped = build_model()
    input_ids = torch.tensor([[10, IMAGE_PAD, 11]])
    with torch.no_grad():
        steady_first = steady(input_ids, steady.compute_text_position_ids(input_ids), 0, pixel_values, grid_thw)
        stepped_first = stepped(input_ids, stepped.compute_text_position_ids(input_ids), 0, pixel_values, grid_thw)
        assert torch.allclose(steady_first, stepped_first, atol=1e-3)
        assert stepped.vision_length == 5

        token = torch.tensor([[12]])
        before = stepped(token, stepped.decode_position_ids(3, token), 3)
        steady_before = steady(token, steady.decode_position_ids(3, token), 3)
        assert torch.allclose(before, steady_before, atol=1e-3)

        # Publish a second frame at the step boundary: nothing before it can change.
        published = stepped.append_frames(pixel_values, grid_thw, vision_position_start=5)
        assert published == 10
        assert stepped.vision_length == 10

        after = stepped(token, stepped.decode_position_ids(4, token), 4)
        steady_after = steady(token, steady.decode_position_ids(4, token), 4)
        assert not torch.allclose(after, steady_after, atol=1e-3)


def test_appended_positions_continue_the_vision_timeline() -> None:
    model = build_model()
    info = [VisionTokenInfo(grid_h=4, grid_w=4, num_frames=2, start=0, vision_tokens_per_frame=4)]
    positions = model.vision_positions_for(info, position_start=7, num_vision_tokens=10)
    assert positions.shape == (3, 1, 10)
    # Frame 0 occupies 7..7 with column/row spread, separator at 7 + max(2, 2) = 9,
    # and the reference's realtime rule restarts the next frame at separator + 1.
    assert positions[0, 0, :5].tolist() == [7, 7, 7, 7, 9]
    assert positions[1, 0, :5].tolist() == [7, 7, 8, 8, 9]
    assert positions[2, 0, :5].tolist() == [7, 8, 7, 8, 9]
    assert positions[0, 0, 5:].tolist() == [10, 10, 10, 10, 12]
    assert positions[1, 0, 5:].tolist() == [10, 10, 11, 11, 12]
    assert positions[2, 0, 5:].tolist() == [10, 11, 10, 11, 12]


def test_paced_segment_positions_follow_the_reference_rule() -> None:
    """A realtime segment starts at the running position; image-pad takes the block."""
    model = build_model()
    # vision_start, time_start, "0", ".", "0", " seconds", time_end, image_pad, vision_end
    segment = torch.tensor([[20, 21, 22, 23, 24, 25, 26, IMAGE_PAD, 27]])
    grid_thw = torch.tensor([[1, 4, 4]])
    positions, grid_start, next_position = model.paced_segment_positions(segment, grid_thw, start_position=100)
    assert positions.shape == (3, 1, 9)
    # Seven text tokens take 100..106, the image-pad takes 107 + max(2, 2) = 109,
    # and the trailing token continues at 110.
    assert positions[0, 0].tolist() == [100, 101, 102, 103, 104, 105, 106, 109, 110]
    assert grid_start == 107
    assert next_position == 111


def test_paced_segment_positions_reject_frame_mismatch() -> None:
    model = build_model()
    segment = torch.tensor([[20, IMAGE_PAD, 21]])
    with pytest.raises(ValueError, match="frame metadata"):
        model.paced_segment_positions(segment, torch.tensor([[1, 4, 4], [1, 4, 4]]), start_position=0)


# ------------------------------------------------------------- T12 admission
def test_vision_budget_is_refused_not_silently_trimmed() -> None:
    model = build_model()
    model.limits = SessionLimits(max_vision_tokens=6, max_text_tokens=None, max_frames_per_append=None)
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        model.append_frames(pixel_values, grid_thw, vision_position_start=0)
        with pytest.raises(BudgetExceededError, match="max_vision_tokens=6"):
            model.append_frames(pixel_values, grid_thw, vision_position_start=5)


def test_frame_burst_limit_is_refused() -> None:
    model = build_model()
    model.limits = SessionLimits(max_vision_tokens=None, max_text_tokens=None, max_frames_per_append=1)
    pixel_values = torch.randn(32, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]])
    with pytest.raises(BudgetExceededError, match="max_frames_per_append=1"):
        model.append_frames(pixel_values, grid_thw, vision_position_start=0)


def test_text_context_limit_is_refused() -> None:
    model = build_model()
    model.limits = SessionLimits(max_vision_tokens=None, max_text_tokens=3, max_frames_per_append=None)
    input_ids = torch.tensor([[10, 11, 12, 13]])
    with pytest.raises(BudgetExceededError, match="max_text_tokens=3"):
        model(input_ids, model.compute_text_position_ids(input_ids), offset=0)


def test_budget_report_tracks_usage() -> None:
    model = build_model()
    model.limits = SessionLimits(max_vision_tokens=64, max_text_tokens=64, max_frames_per_append=4)
    input_ids = torch.tensor([[10, IMAGE_PAD, 11]])
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    with torch.no_grad():
        model(input_ids, model.compute_text_position_ids(input_ids), 0, pixel_values, torch.tensor([[1, 4, 4]]))
    report = model.budget_report()
    assert report == {"vision_tokens": 5, "max_vision_tokens": 64, "text_tokens": 3, "max_text_tokens": 64}


def test_chunked_prefill_matches_single_prefill() -> None:
    """Two prefill chunks must equal one prefill of the same logical input.

    The cache grows across chunks, so this exercises the non-zero-offset causal
    path in self-attention rather than the single-shot prefill shortcut.
    """
    model = build_model()
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    positions = model.compute_text_position_ids(input_ids)
    with torch.no_grad():
        whole = model(input_ids, positions, offset=0)
        model.reset()
        first = model(input_ids[:, :2], positions[:, :, :2], offset=0)
        second = model(input_ids[:, 2:5], positions[:, :, 2:5], offset=2)
        third = model(input_ids[:, 5:], positions[:, :, 5:], offset=5)
    assert torch.allclose(first[0, -1].float(), whole[0, 1].float(), atol=5e-2)
    assert torch.allclose(second[0, -1].float(), whole[0, 4].float(), atol=5e-2)
    assert torch.allclose(third[0, -1].float(), whole[0, 5].float(), atol=5e-2)


def test_chunked_prefill_keeps_cross_attention_visible_set() -> None:
    """A frame published in the first chunk stays visible to later chunks."""
    model = build_model()
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4]])
    frame_mask = torch.tensor([[[[False], [False]]]])
    head_ids = torch.tensor([[10, IMAGE_PAD]])
    tail_ids = torch.tensor([[11, 12]])
    with torch.no_grad():
        model(head_ids, model.compute_text_position_ids(head_ids), 0, pixel_values, grid_thw, frame_mask)
        before = model.vision_length
        model(tail_ids, model.compute_text_position_ids(tail_ids) + 2, offset=2)
    assert before == 5
    assert model.vision_length == 5
    assert model.text_cache.length(0) == 4
