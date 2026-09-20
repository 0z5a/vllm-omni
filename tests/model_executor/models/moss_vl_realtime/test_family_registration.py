# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks for the MOSS-VL-Realtime entry in the shared text-output family registry.

Registration here is a prompt/stop-token contract, not a serving claim: it lets a
MOSS-VL checkpoint resolve to its own family in the shared x-to-text example
instead of falling back to the generic one. The config reader is patched by hand
so the checks also run under the standalone runner, which has no fixtures.
"""

from __future__ import annotations

from contextlib import contextmanager

import vllm.transformers_utils.config as vllm_config

import vllm_omni.model_extras.registry as registry
from vllm_omni.model_extras.moss_vl import (
    END_OF_TURN_TOKEN_ID,
    RESPONSE_TOKEN_ID,
    SILENCE_TOKEN_ID,
    build_x_to_text_prompt,
)


@contextmanager
def checkpoint_config(config: dict):
    original = vllm_config.get_hf_file_to_dict
    vllm_config.get_hf_file_to_dict = lambda *args, **kwargs: config
    try:
        yield
    finally:
        vllm_config.get_hf_file_to_dict = original


def test_family_is_resolved_from_the_checkpoint_config() -> None:
    with checkpoint_config({"model_type": "moss_vl"}):
        assert registry.get_x_to_text_model_family("some/checkpoint") == "moss_vl"


def test_architecture_match_resolves_the_family() -> None:
    with checkpoint_config({"model_type": "unknown", "architectures": ["MossVLForConditionalGeneration"]}):
        assert registry.get_x_to_text_model_family("some/checkpoint") == "moss_vl"


def test_other_families_are_not_captured() -> None:
    with checkpoint_config({"model_type": "bagel"}):
        assert registry.get_x_to_text_model_family("some/checkpoint") == "bagel"
    with checkpoint_config({"model_type": "something_else"}):
        assert registry.get_x_to_text_model_family("some/checkpoint") == "generic"


def test_prompt_builder_stops_on_the_turn_end_token() -> None:
    request, stop_ids = build_x_to_text_prompt("OpenMOSS-Team/MOSS-VL-Realtime", "Describe the image.", has_image=True)
    assert request == {"prompt": "Describe the image.", "modalities": ["text"]}
    assert stop_ids == [END_OF_TURN_TOKEN_ID]


def test_realtime_control_tokens_are_not_stop_tokens() -> None:
    """Silence means keep observing; it must never terminate offline decoding."""
    _, stop_ids = build_x_to_text_prompt("m", "p", has_image=False)
    assert SILENCE_TOKEN_ID not in stop_ids
    assert RESPONSE_TOKEN_ID not in stop_ids
    assert {SILENCE_TOKEN_ID, RESPONSE_TOKEN_ID} == {151671, 151672}
