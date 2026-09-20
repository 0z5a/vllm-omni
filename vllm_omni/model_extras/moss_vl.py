# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-VL-Realtime prompt family for the shared text-output path.

The offline path takes a plain user turn through the checkpoint's own chat
template; there is no model-specific prompt scaffolding to reproduce. What *is*
model-specific is the stop token: the checkpoint ends a turn on
``<|im_end|>`` (id 151645, also its EOS).

The realtime control tokens (``<|silence|>`` 151671, ``<|response|>`` 151672)
are deliberately not stop tokens here. They belong to the duplex session
contract, where silence means "keep observing" rather than "stop".
"""

from __future__ import annotations

from typing import Any

#: ``<|im_end|>`` / EOS in the published checkpoint.
END_OF_TURN_TOKEN_ID = 151645
SILENCE_TOKEN_ID = 151671
RESPONSE_TOKEN_ID = 151672


def build_x_to_text_prompt(
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    """Return the request fields and stop ids for a MOSS-VL-Realtime text answer."""
    del model, has_image
    return {"prompt": prompt, "modalities": ["text"]}, [END_OF_TURN_TOKEN_ID]
