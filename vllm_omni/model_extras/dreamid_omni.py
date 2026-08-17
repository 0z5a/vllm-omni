# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

DREAMID_OMNI_EXTRA_BODY_PARAMS = frozenset({"solver_name", "shift"})
DREAMID_OMNI_INPUT_AUDIO_SAMPLE_RATE = 16000


def _clean_official_prompt(prompt: str) -> str:
    """Remove metadata tags used by DreamID's official prompt fixtures."""
    prompt = re.sub(
        r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(
        r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(r"\[[A-Z_]+\]", "", prompt)
    return re.sub(r"\n\s*\n", "\n", prompt).strip()


def build_x_to_video_audio_prompt(
    prompt: dict[str, Any],
    request_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the canonical X-to-video+audio envelope for DreamID-Omni."""
    result = dict(prompt)
    result["prompt"] = _clean_official_prompt(str(prompt["prompt"]))

    for key in ("video_negative_prompt", "audio_negative_prompt"):
        if request_options.get(key) is not None:
            result[key] = request_options[key]

    media_inputs = prompt.get("multi_modal_data")
    if isinstance(media_inputs, Mapping) and "audio" in media_inputs:
        adapted_media = dict(media_inputs)
        audio_inputs = media_inputs["audio"]
        if not isinstance(audio_inputs, list):
            audio_inputs = [audio_inputs]
        adapted_media["audio"] = [_extract_waveform(item) for item in audio_inputs]
        result["multi_modal_data"] = adapted_media
    return result


def _extract_waveform(audio: Any) -> Any:
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise TypeError("Canonical audio inputs must be (waveform, sample_rate) tuples.")
    waveform, sample_rate = audio
    if sample_rate != DREAMID_OMNI_INPUT_AUDIO_SAMPLE_RATE:
        raise ValueError(
            "DreamID-Omni audio inputs must be resampled to "
            f"{DREAMID_OMNI_INPUT_AUDIO_SAMPLE_RATE} Hz; got {sample_rate}."
        )
    return waveform
