# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading for the native MOSS-VL-Realtime model.

Weights stream from the sharded safetensors files straight into the module
parameters, so loading never materialises a second full copy of the 22.7 GB
BF16 checkpoint. Every key is accounted for: loaded, missing, unexpected, or an
explicitly recomputed buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from .config import MossVLConfig
from .modeling import MossVLNativeModel

RECOMPUTED_BUFFERS = (
    "model.visual.rotary_pos_emb.inv_freq",
    "model.language_model.rotary_emb.inv_freq",
)


@dataclass
class LoadReport:
    checkpoint_keys: int = 0
    loaded_keys: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    recomputed_buffers: list[str] = field(default_factory=list)
    dtype_mismatches: dict[str, str] = field(default_factory=dict)
    shards: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_keys": self.checkpoint_keys,
            "loaded_keys": self.loaded_keys,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "recomputed_buffers": self.recomputed_buffers,
            "dtype_mismatches": self.dtype_mismatches,
            "shards": self.shards,
            "total_bytes": self.total_bytes,
        }


def load_native_model(
    checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MossVLNativeModel, LoadReport]:
    checkpoint = Path(checkpoint)
    cfg = MossVLConfig.from_json(checkpoint / "config.json")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    with torch.device("meta"):
        model = MossVLNativeModel(cfg)
    model.to(dtype)
    model.to_empty(device=device)
    model.eval()

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    report = LoadReport(
        checkpoint_keys=len(weight_map),
        recomputed_buffers=[key for key in RECOMPUTED_BUFFERS if key not in weight_map],
    )
    report.shards = sorted(set(weight_map.values()))

    with torch.no_grad():
        for shard in report.shards:
            keys = [key for key, shard_name in weight_map.items() if shard_name == shard]
            with safe_open(checkpoint / shard, framework="pt") as handle:
                for key in keys:
                    if key in parameters:
                        target = parameters[key]
                    elif key in buffers:
                        target = buffers[key]
                    else:
                        report.unexpected_keys.append(key)
                        continue
                    tensor = handle.get_tensor(key)
                    if tensor.shape != target.shape:
                        raise ValueError(
                            f"{key}: checkpoint shape {tuple(tensor.shape)} != model shape {tuple(target.shape)}"
                        )
                    if tensor.dtype != target.dtype:
                        report.dtype_mismatches[key] = f"{tensor.dtype} -> {target.dtype}"
                    target.copy_(tensor.to(device=device))
                    report.loaded_keys += 1
                    report.total_bytes += tensor.numel() * tensor.element_size()

    model.model.language_model.rotary_emb.materialize(device)
    model.model.visual.rotary_pos_emb.materialize(device)

    loaded = set(weight_map)
    report.missing_keys = sorted(
        key for key in list(parameters) + list(buffers) if key not in loaded and key not in report.recomputed_buffers
    )
    return model, report


def check_gates(model: MossVLNativeModel) -> dict[str, float]:
    """Cross-attention gates must be non-zero or every gated branch is a no-op."""
    return {
        name: float(parameter.detach().float().abs().mean())
        for name, parameter in model.named_parameters()
        if name.endswith("cross_attn_attn_gate") or name.endswith("cross_attn_mlp_gate")
    }
