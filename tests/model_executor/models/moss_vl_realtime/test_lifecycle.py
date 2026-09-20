# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle stress checks for the native session (RFC T12/T13).

The realtime validation plan asks for repeated create/close cycles after a
warm-up rather than a single run, and for a closed session to leave nothing
behind. These run on CPU with a tiny model: the point is state and resource
accounting, not throughput.
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
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMAGE_PAD = 3
WARMUP_ROUNDS = 20
STRESS_ROUNDS = 100


def config() -> MossVLConfig:
    return MossVLConfig(
        text=MossVLTextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            cross_attention_layers=(1,),
            vocab_size=64,
        ),
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
    )


def one_session(model: MossVLNativeModel, device: torch.device) -> None:
    """Prefill a frame, decode two steps, publish another frame, close."""
    input_ids = torch.tensor([[10, IMAGE_PAD, 11]], device=device)
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16, device=device)
    grid_thw = torch.tensor([[1, 4, 4]], device=device)
    with torch.no_grad():
        model(input_ids, model.compute_text_position_ids(input_ids), 0, pixel_values, grid_thw)
        for offset in (3, 4):
            token = torch.tensor([[12]], device=device)
            model(token, model.decode_position_ids(offset, token), offset)
        model.append_frames(pixel_values, grid_thw, model.next_position)
    model.reset()


def test_repeated_create_close_leaves_no_state() -> None:
    """After every cycle the caches, the position cursor and the metadata are empty."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = MossVLNativeModel(config()).to(torch.bfloat16).eval()

    for _ in range(WARMUP_ROUNDS):
        one_session(model, device)

    for round_index in range(STRESS_ROUNDS):
        one_session(model, device)
        assert model.text_cache.length(0) == 0, f"round {round_index}: text cache not released"
        assert model.vision_length == 0, f"round {round_index}: vision cache not released"
        assert model.vision_token_info == [], f"round {round_index}: frame metadata not released"
        assert model.rope_delta is None, f"round {round_index}: rope delta not cleared"
        assert model.next_position == 0, f"round {round_index}: position cursor not reset"
        assert model.memory_report()["text_cache_bytes"] == 0
        assert model.memory_report()["vision_cache_bytes"] == 0


def test_a_closed_session_does_not_accumulate_parameters() -> None:
    """Nothing in a session may add or resize weights."""
    torch.manual_seed(0)
    model = MossVLNativeModel(config()).to(torch.bfloat16).eval()
    before = {name: parameter.shape for name, parameter in model.named_parameters()}
    for _ in range(5):
        one_session(model, torch.device("cpu"))
    after = {name: parameter.shape for name, parameter in model.named_parameters()}
    assert before == after


def test_budget_failure_does_not_corrupt_the_session() -> None:
    """A refused append must leave the published extent usable, not half-written."""
    torch.manual_seed(0)
    model = MossVLNativeModel(config()).to(torch.bfloat16).eval()
    model.limits = SessionLimits(max_vision_tokens=6, max_text_tokens=None, max_frames_per_append=None)
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        model.append_frames(pixel_values, grid_thw, 0)
        published = model.vision_length
        try:
            model.append_frames(pixel_values, grid_thw, published)
        except BudgetExceededError:
            pass
        else:
            raise AssertionError("the second append should have been refused")
    assert model.vision_length == published
    input_ids = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        logits = model(input_ids, model.compute_text_position_ids(input_ids), 11)
    assert logits.shape == (1, 3, 64)
