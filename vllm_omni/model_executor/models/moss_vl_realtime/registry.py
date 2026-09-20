# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture resolution for MOSS-VL-Realtime checkpoints.

The published checkpoint ships remote code and a converted SGLang variant that
is not weight-equivalent. This module decides, from ``config.json`` alone, which
checkpoints the native executor accepts, and reports why a checkpoint is
refused instead of loading something that would silently not match.

This is the native entry point, not a vLLM worker registration: the current
implementation is a single-device eager executor, and it must not be reachable
through the paged-attention model registry until that port exists (T02 owns the
placement decision for the shared registry).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import MossVLConfig
from .loader import LoadReport, load_native_model
from .modeling import MossVLNativeModel

SUPPORTED_ARCHITECTURES = ("MossVLForConditionalGeneration",)
SUPPORTED_MODEL_TYPE = "moss_vl"


@dataclass(frozen=True)
class CheckpointVerdict:
    checkpoint: Path
    architecture: str
    model_type: str
    supported: bool
    reasons: list[str]


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointVerdict:
    """Decide whether a checkpoint directory can be loaded natively."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{checkpoint} has no config.json")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = tuple(raw.get("architectures") or ())
    model_type = str(raw.get("model_type", ""))
    reasons: list[str] = []

    if model_type != SUPPORTED_MODEL_TYPE:
        reasons.append(f"model_type {model_type!r} is not {SUPPORTED_MODEL_TYPE!r}")
    if not any(arch in SUPPORTED_ARCHITECTURES for arch in architectures):
        reasons.append(f"architectures {architectures} contain none of {SUPPORTED_ARCHITECTURES}")

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if not any(key.endswith("cross_attn_attn_gate") for key in weight_map):
            reasons.append("weights carry no cross-attention gates: this looks like the converted SGLang variant")
    else:
        reasons.append("no model.safetensors.index.json: sharded BF16 weights are required")

    vision_fields = raw.get("vision_config") or {}
    for field in ("deepstack_visual_indexes", "spatial_merge_size", "num_position_embeddings"):
        if field not in vision_fields:
            reasons.append(f"vision_config.{field} is missing")

    return CheckpointVerdict(
        checkpoint=checkpoint,
        architecture=architectures[0] if architectures else "",
        model_type=model_type,
        supported=not reasons,
        reasons=reasons,
    )


def build_native_model(
    checkpoint: str | Path,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MossVLNativeModel, LoadReport, MossVLConfig]:
    """Load a supported checkpoint, refusing anything the executor cannot honour."""
    verdict = inspect_checkpoint(checkpoint)
    if not verdict.supported:
        raise ValueError("unsupported MOSS-VL checkpoint: " + "; ".join(verdict.reasons))
    model, report = load_native_model(checkpoint, torch.device(device), dtype)
    return model, report, model.config
