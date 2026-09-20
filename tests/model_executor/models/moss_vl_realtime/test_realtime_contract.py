# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract checks over the captured realtime step fixture.

`steps.jsonl` records every forward the reference realtime session performed:
the tokens it fed, the exact 3-axis positions it used, and its argmax. That makes
the realtime position rule and the segment protocol checkable on CPU, without a
checkpoint and without the reference stack:

* the positions are recomputed from the recorded tokens with the native rule and
  must match the reference's own positions bit for bit;
* every multi-token segment must have the shape the protocol describes — an
  optional user turn, one silence marker, then one frame segment per frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_omni.model_executor.models.moss_vl_realtime.config import (
    MossVLConfig,
    MossVLTextConfig,
    MossVLVisionConfig,
)
from vllm_omni.model_executor.models.moss_vl_realtime.modeling import MossVLNativeModel

REALTIME_DIR = Path(__file__).resolve().parents[3] / "assets" / "moss_vl_realtime" / "reference" / "realtime"

IMAGE_TOKEN_ID = 151655
VISION_START = 151652
VISION_END = 151653
TIME_START = 151669
TIME_END = 151670
SILENCE = 151671
RESPONSE = 151672
IM_END = 151645
IM_START = 151644


def steps() -> list[dict]:
    return [
        json.loads(line) for line in (REALTIME_DIR / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]


def realtime_model() -> MossVLNativeModel:
    """A tiny model whose ids and merge size match the published checkpoint."""
    config = MossVLConfig(
        text=MossVLTextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            cross_attention_layers=(1,),
            vocab_size=200000,
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
        image_token_id=IMAGE_TOKEN_ID,
    )
    return MossVLNativeModel(config)


def frame_grids(segment: list[int]) -> torch.Tensor:
    """The fixture's frames are 4x4-patch images on a 4x4 grid each."""
    count = sum(1 for token in segment if token == IMAGE_TOKEN_ID)
    return torch.tensor([[1, 4, 4]] * count)


def test_realtime_positions_match_the_reference() -> None:
    """Frame-bearing steps are reproduced bit for bit, chaining the position cursor.

    Each segment starts one past the previous segment's last position, so the
    check recomputes the whole timeline instead of being handed each start.
    """
    model = realtime_model()
    cursor = 0
    checked = 0
    for step in steps():
        reference = torch.tensor(step["position_ids"])
        assert reference[0, 0].item() == cursor, (
            f"step {step['step']} starts at {reference[0, 0].item()}, expected {cursor}"
        )
        if IMAGE_TOKEN_ID in step["input_ids"]:
            segment = torch.tensor([step["input_ids"]])
            positions, _, next_position = model.paced_segment_positions(
                segment, frame_grids(step["input_ids"]), start_position=cursor
            )
            assert positions.shape == (3, 1, reference.shape[1])
            assert torch.equal(positions[:, 0], reference), f"step {step['step']} positions differ"
            assert next_position == reference[0, -1].item() + 1
            checked += 1
        cursor = reference[0, -1].item() + 1
    assert checked == 2


def test_segment_protocol_shape() -> None:
    """A segment carries one silence marker and one frame segment per frame."""
    all_steps = steps()
    batch = all_steps[1]["input_ids"]
    frames = batch.count(IMAGE_TOKEN_ID)
    assert frames == 3
    assert batch.count(SILENCE) == 2  # the model's own decision plus the segment marker
    # the user turn is rendered between the two silence markers
    assert batch.index(IM_END) < batch.index(IM_START) < batch.index(SILENCE, batch.index(IM_START))
    chunks = []
    for index, token in enumerate(batch):
        if token == VISION_START:
            chunks.append(batch[index:])
    assert len(chunks) == frames
    for position, chunk in enumerate(chunks):
        stop = len(chunks[position + 1]) if position + 1 < len(chunks) else 0
        piece = chunk[: len(chunk) - stop] if stop else chunk
        assert piece[0] == VISION_START
        assert piece[1] == TIME_START
        assert piece[-3:] == [TIME_END, IMAGE_TOKEN_ID, VISION_END]


def test_decision_sequence_is_recorded() -> None:
    """The trace shows silence while waiting, then a response, then answer tokens."""
    argmaxes = [step["argmax"] for step in steps()]
    assert argmaxes[0] == SILENCE
    assert argmaxes[1] == RESPONSE
    assert argmaxes[2] == RESPONSE
    # the answer starts after the response markers and ends on silence
    assert argmaxes[3] not in {SILENCE, RESPONSE}
    assert argmaxes[-1] == SILENCE
