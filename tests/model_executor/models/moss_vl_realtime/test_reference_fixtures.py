# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract checks over the captured MOSS-VL-Realtime reference fixtures.

Validates the recorded artifacts on CPU: file hashes from the fixture manifest,
the exact integer contract (positions, visibility), and an independent
recomputation of the frame-to-token visibility expansion from ``input_ids``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

REFERENCE = Path(__file__).resolve().parents[3] / "assets" / "moss_vl_realtime" / "reference"
CASES = ("prompt_only", "image", "short_video")


def manifest() -> dict:
    return json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))


def tensors(case: str) -> dict[str, torch.Tensor]:
    with safe_open(REFERENCE / case / "captured_tensors.safetensors", framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


@pytest.mark.parametrize("case", CASES)
def test_fixture_hashes_match_manifest(case: str) -> None:
    for name, entry in manifest()["cases"][case]["files"].items():
        payload = (REFERENCE / case / name).read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("case", CASES)
def test_captured_contract_is_integer_exact(case: str) -> None:
    captured = tensors(case)
    for key in ("text_position_ids", "shifted_text_position_ids", "vision_position_ids"):
        assert captured[key].dtype == torch.int64
    assert captured["text_position_ids"].shape[0] == 3
    negative = torch.finfo(torch.bfloat16).min
    mask = captured["cross_attention_mask_expanded"]
    assert mask.dtype == torch.bfloat16
    assert set(mask.unique().tolist()) <= {0.0, negative}


@pytest.mark.parametrize("case", CASES)
def test_visibility_expansion_recomputed_from_frames(case: str) -> None:
    """Rebuild the token-level visibility from the frame mask without the model."""
    frame_mask = json.loads((REFERENCE / case / "contract-report.json").read_text(encoding="utf-8"))
    captured = tensors(case)
    with safe_open(REFERENCE / case / "inputs.safetensors", framework="pt") as handle:
        inputs = {key: handle.get_tensor(key) for key in handle.keys()}
    coarse = inputs["cross_attention_mask"]
    assert coarse.shape[-1] == frame_mask["cross_attention_frames"]

    expanded = captured["cross_attention_mask_expanded"]
    total_vision = expanded.shape[-1]
    repeats = total_vision // max(1, coarse.shape[-1])
    assert repeats * coarse.shape[-1] == total_vision
    negative = torch.finfo(torch.bfloat16).min
    rebuilt = torch.zeros_like(coarse, dtype=torch.bfloat16).masked_fill_(coarse, negative)
    rebuilt = rebuilt.repeat_interleave(repeats, dim=-1)
    assert torch.equal(rebuilt.reshape(expanded.shape), expanded)


def test_tolerances_cover_every_case() -> None:
    tolerances = json.loads((REFERENCE / "tolerances.json").read_text(encoding="utf-8"))
    assert set(tolerances["measured_abs_max"]) == set(CASES)
    for case in CASES:
        assert tolerances["measured_abs_max"][case]["prefill_logits"] > 0
