# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract checks for the model-local session (RFC T09/T10).

The session is the piece the duplex engine will drive: it owns the token stream,
the paced position cursor and frame publication, and it owns no loop of its own.
These checks pin the properties the engine depends on, on CPU with a tiny model.
"""

from __future__ import annotations

from typing import Any

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
from vllm_omni.model_executor.models.moss_vl_realtime.session import MossVLNativeSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMAGE_PAD = 3
SILENCE = 5  # the tiny model's voice for `<|silence|>`
FRAME_TOKENS = (10, IMAGE_PAD, 11)
TOKENS_PER_FRAME = 3


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
        silence_token_id=SILENCE,
    )


def make_model(limits: SessionLimits | None = None) -> MossVLNativeModel:
    torch.manual_seed(0)
    model = MossVLNativeModel(config()).to(torch.bfloat16).eval()
    if limits is not None:
        model.limits = limits
    return model


def encoder(seed: int = 0):
    """A stand-in for the processor: one 4x4 frame per call, deterministic per seed."""
    generator = torch.Generator().manual_seed(seed)

    def encode(frame: Any, media_timestamp_ms: int) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.randn(16, 3 * 1 * 16 * 16, dtype=torch.bfloat16, generator=generator),
            "grid_thw": torch.tensor([[1, 4, 4]]),
            "segment_ids": torch.tensor([list(FRAME_TOKENS)]),
        }

    return encode


def make_session(model: MossVLNativeModel, *, prompts: bool = True, limits: SessionLimits | None = None):
    rendered: list[str] = []

    def prompt_ids_for(prompt: str) -> list[int]:
        rendered.append(prompt)
        return [20, 21]

    session = MossVLNativeSession(
        model,
        frame_encoder=encoder(),
        detokenize=lambda ids: f"<{ids[0]}>",
        limits=limits,
        max_new_tokens_per_turn=2,
    )
    session.start(torch.tensor([[10, 11, 12]]))
    return session, rendered, prompt_ids_for if prompts else None


def force_decision(session: MossVLNativeSession, token_id: int) -> None:
    """Make the next turn's greedy decision land on ``token_id``."""
    logits = torch.full((1, 1, config().text.vocab_size), -10.0)
    logits[0, -1, token_id] = 10.0
    session._logits = logits


def publish(session: MossVLNativeSession, prompt_ids_for, *, prompts: list[str] | None = None, frames: int = 1):
    return session.append(
        frames=[(f"frame{index}.ppm", index * 1000) for index in range(frames)],
        prompts=prompts,
        prompt_ids_for=prompt_ids_for,
    )


# ---- lifecycle ------------------------------------------------------------
def test_append_before_start_is_refused() -> None:
    model = make_model()
    session = MossVLNativeSession(model, frame_encoder=encoder(), detokenize=str)
    with pytest.raises(RuntimeError, match="start\\(\\)"):
        session.append(frames=[("a.ppm", 0)])


def test_empty_append_is_refused() -> None:
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    with pytest.raises(ValueError, match="at least one frame or prompt"):
        session.append(prompt_ids_for=prompt_ids_for)


def test_use_after_close_is_refused() -> None:
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    session.close()
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        publish(session, prompt_ids_for)


def test_close_releases_the_session_state() -> None:
    """A closed session leaves no cache, no cursor and no frame metadata behind."""
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    publish(session, prompt_ids_for, frames=2)
    assert model.vision_length > 0
    report = session.close()
    assert report["frames_accepted"] == 2
    assert model.vision_length == 0
    assert model.text_cache.length(0) == 0
    assert model.vision_token_info == []
    assert model.next_position == 0
    assert model.memory_report()["vision_cache_bytes"] == 0


def test_close_is_reported_in_the_event_trace() -> None:
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    publish(session, prompt_ids_for)
    session.close()
    kinds = [event.kind for event in session.events]
    assert kinds[0] == "session_open"
    assert "append" in kinds
    assert kinds[-1] == "session_closed"


# ---- publication ----------------------------------------------------------
def test_publication_happens_before_the_step_that_consumes_it() -> None:
    """The append result is the extent the following step can attend to."""
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    before = model.vision_length
    result = publish(session, prompt_ids_for, frames=2)
    assert before == 0
    assert result.accepted_frames == 2
    assert result.published_vision_length == model.vision_length > before
    assert result.next_position == model.next_position


def test_frames_advance_the_running_position_by_the_grid_size() -> None:
    """An image-pad token consumes ``max(eh, ew) + 1`` positions; every other token one.

    The frame's own text (vision/time markers) and the segment's leading silence
    marker are ordinary text and each consume a position.
    """
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    before = model.next_position
    result = publish(session, prompt_ids_for, frames=3)
    marker = 1
    frame_text = len(FRAME_TOKENS) - 1
    assert result.next_position == before + marker + 3 * (frame_text + 2 + 1)


def test_a_prompt_only_append_consumes_one_position_per_token() -> None:
    model = make_model()
    session, rendered, prompt_ids_for = make_session(model)
    before = model.next_position
    result = session.append(prompts=["Focus."], prompt_ids_for=prompt_ids_for)
    assert rendered == ["Focus."]
    # the rendered turn (2 tokens) plus the segment's silence marker
    assert result.next_position == before + 3
    assert model.vision_length == 0


def test_the_configured_control_ids_are_the_checkpoint_ids() -> None:
    """The session reads the control ids from the config instead of baking them in."""
    checkpoint = MossVLConfig()
    assert checkpoint.silence_token_id == 151671
    assert checkpoint.response_token_id == 151672


def test_the_frame_mask_spans_every_published_frame() -> None:
    """Visibility is session-wide: the mask carries one column per published frame.

    A per-segment mask would let a late segment see a frame it arrived before.
    """
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    masks: list[Any] = []
    model.register_forward_pre_hook(
        lambda module, args, kwargs: masks.append(kwargs["cross_attention_mask"]), with_kwargs=True
    )
    publish(session, prompt_ids_for, frames=2)
    publish(session, prompt_ids_for, frames=1)
    assert [mask.shape[-1] for mask in masks] == [2, 3]
    # The reference rebuilds visibility from the step's own tokens, so a segment
    # without a frame marker sees no frame as visible; the mask is still applied
    # with the session-wide width rather than dropped.
    session.append(prompts=["Text only."], prompt_ids_for=prompt_ids_for)
    assert masks[-1].shape[-1] == 3
    assert bool(masks[-1].all())


def test_a_prompt_only_append_needs_a_renderer() -> None:
    model = make_model()
    session, _, _ = make_session(model)
    with pytest.raises(ValueError, match="prompt_ids_for"):
        session.append(prompts=["Focus."])


# ---- output ---------------------------------------------------------------
def test_a_turn_is_bounded_by_the_budget() -> None:
    """A turn that never chooses silence still stops at the budget."""
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    session.max_new_tokens_per_turn = 3
    force_decision(session, IMAGE_PAD)  # never the silence token
    text, silent = session.step()
    assert not silent
    assert text.count("<") == 3, text
    assert text.startswith("<3>")


def test_silence_ends_the_turn_without_consuming_a_position() -> None:
    """Silence is a decision carried into the next segment, not a generated token.

    Feeding it here would put the marker in the middle of the stream and shift every
    position after it, so the cursor must not move.
    """
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    force_decision(session, SILENCE)
    before = model.next_position
    cached = model.text_cache.length(0)
    text, silent = session.step()
    assert silent
    assert text == ""
    assert model.next_position == before
    assert model.text_cache.length(0) == cached


def test_silence_is_replayed_at_the_head_of_the_next_segment() -> None:
    """The carried marker is the first token the next append feeds."""
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    force_decision(session, SILENCE)
    assert session.step()[1]
    seen: list[list[int]] = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0][0].tolist()))
    publish(session, prompt_ids_for)
    assert seen and seen[0][0] == SILENCE


def test_generated_text_is_queued_for_polling_in_order() -> None:
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    force_decision(session, IMAGE_PAD)
    session.step()
    text = session.poll_output()
    assert text is not None and text.count("<") == 2, text
    assert session.poll_output() is None


def test_a_silent_turn_queues_nothing() -> None:
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    force_decision(session, SILENCE)
    session.step()
    assert session.poll_output() is None


# ---- budgets --------------------------------------------------------------
def test_a_frame_budget_refusal_keeps_the_session_usable() -> None:
    """Refused work must not half-publish: the extent stays where it was."""
    model = make_model(SessionLimits(max_vision_tokens=None, max_text_tokens=None, max_frames_per_append=1))
    session, _, prompt_ids_for = make_session(model)
    with pytest.raises(BudgetExceededError, match="max_frames_per_append"):
        publish(session, prompt_ids_for, frames=2)
    assert model.vision_length == 0
    publish(session, prompt_ids_for, frames=1)
    assert model.vision_length > 0


def test_a_refused_append_keeps_the_carried_silence_marker() -> None:
    """A refusal may not consume session state, or the retry loses the marker."""
    model = make_model(SessionLimits(max_vision_tokens=None, max_text_tokens=None, max_frames_per_append=1))
    session, _, prompt_ids_for = make_session(model)
    force_decision(session, SILENCE)
    assert session.step()[1]
    with pytest.raises(BudgetExceededError):
        publish(session, prompt_ids_for, frames=2)
    seen: list[list[int]] = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0][0].tolist()))
    publish(session, prompt_ids_for, frames=1)
    assert seen[0][0] == SILENCE


def test_a_refused_append_publishes_nothing() -> None:
    """The frame that would have fit is not published either: the append is all or nothing."""
    model = make_model(SessionLimits(max_vision_tokens=9, max_text_tokens=None, max_frames_per_append=None))
    session, _, prompt_ids_for = make_session(model)
    with pytest.raises(BudgetExceededError, match="max_vision_tokens"):
        publish(session, prompt_ids_for, frames=2)  # 5 + 5 tokens against a budget of 9
    assert model.vision_length == 0
    assert model.budget_report()["vision_tokens"] == 0


def test_a_text_budget_refusal_keeps_the_position_cursor() -> None:
    # 3 prefilled prompt tokens plus one 4-token segment fit; the next segment cannot.
    model = make_model(SessionLimits(max_vision_tokens=None, max_text_tokens=7, max_frames_per_append=None))
    session, _, prompt_ids_for = make_session(model)
    publish(session, prompt_ids_for, frames=1)
    cursor = model.next_position
    cached = model.text_cache.length(0)
    with pytest.raises(BudgetExceededError, match="max_text_tokens"):
        publish(session, prompt_ids_for, frames=1)
    assert model.next_position == cursor
    assert model.text_cache.length(0) == cached


def test_a_vision_budget_refusal_keeps_the_session_usable() -> None:
    model = make_model(SessionLimits(max_vision_tokens=5, max_text_tokens=None, max_frames_per_append=None))
    session, _, prompt_ids_for = make_session(model)
    result = publish(session, prompt_ids_for, frames=1)
    assert result.published_vision_length == 5
    with pytest.raises(BudgetExceededError, match="max_vision_tokens"):
        publish(session, prompt_ids_for, frames=1)
    assert model.vision_length == 5
    assert not session.closed
    session.append(prompts=["Still here."], prompt_ids_for=prompt_ids_for)


def test_limits_can_be_supplied_at_construction() -> None:
    model = make_model()
    limits = SessionLimits(max_vision_tokens=5, max_text_tokens=None, max_frames_per_append=None)
    session = MossVLNativeSession(
        model, frame_encoder=encoder(), detokenize=str, limits=limits, max_new_tokens_per_turn=1
    )
    assert model.limits is limits
    assert session.max_new_tokens_per_turn == 1


# ---- session shape --------------------------------------------------------
def test_a_session_exposes_no_loop_of_its_own() -> None:
    """The engine owns stepping: the session has no generate/run/start_loop entry point."""
    public = {name for name in dir(MossVLNativeSession) if not name.startswith("_")}
    assert public.isdisjoint({"run", "generate", "loop", "serve", "start_loop", "spawn"})


def test_append_reports_the_published_extent_not_the_accepted_count() -> None:
    """What the transport handed over and what the model can see are different promises."""
    model = make_model()
    session, _, prompt_ids_for = make_session(model)
    result = publish(session, prompt_ids_for, frames=1)
    assert result.accepted_frames == 1
    assert result.pending_frames == 0
    assert result.published_vision_length == model.vision_length
