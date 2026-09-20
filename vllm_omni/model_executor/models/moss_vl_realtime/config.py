# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint configuration for MOSS-VL-Realtime.

The published checkpoint ships remote code (``configuration_moss_vl.py``).
The native path reads the same ``config.json`` directly so that loading a model
never executes checkpoint-provided Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MossVLTextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True
    cross_attention_layers: tuple[int, ...] = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46)
    vocab_size: int = 151936
    max_position_embeddings: int = 262144

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class MossVLVisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096
    hidden_act: str = "gelu_pytorch_tanh"
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def rotary_dim(self) -> int:
        """Vision RoPE rotates half of each head; the embedding is duplicated."""
        return self.head_dim // 2


@dataclass(frozen=True)
class MossVLConfig:
    text: MossVLTextConfig = field(default_factory=MossVLTextConfig)
    vision: MossVLVisionConfig = field(default_factory=MossVLVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_seq_pad_multiple: int = 1
    tie_word_embeddings: bool = False
    architectures: tuple[str, ...] = ("MossVLForConditionalGeneration",)
    model_type: str = "moss_vl"

    @classmethod
    def from_json(cls, path: str | Path) -> "MossVLConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        text = MossVLTextConfig(
            hidden_size=raw["text_config"]["hidden_size"],
            intermediate_size=raw["text_config"]["intermediate_size"],
            num_hidden_layers=raw["text_config"]["num_hidden_layers"],
            num_attention_heads=raw["text_config"]["num_attention_heads"],
            num_key_value_heads=raw["text_config"]["num_key_value_heads"],
            head_dim=raw["text_config"]["head_dim"],
            rms_norm_eps=raw["text_config"]["rms_norm_eps"],
            rope_theta=float(raw["text_config"]["rope_theta"]),
            mrope_section=tuple(raw["text_config"]["rope_scaling"]["mrope_section"]),
            mrope_interleaved=bool(raw["text_config"]["rope_scaling"].get("mrope_interleaved", False)),
            cross_attention_layers=tuple(raw["text_config"]["cross_attention_layers"]),
            vocab_size=raw["text_config"]["vocab_size"],
            max_position_embeddings=raw["text_config"]["max_position_embeddings"],
        )
        vision_raw = raw["vision_config"]
        vision = MossVLVisionConfig(
            depth=vision_raw["depth"],
            hidden_size=vision_raw["hidden_size"],
            intermediate_size=vision_raw["intermediate_size"],
            num_heads=vision_raw["num_heads"],
            in_channels=vision_raw["in_channels"],
            patch_size=vision_raw["patch_size"],
            spatial_merge_size=vision_raw["spatial_merge_size"],
            temporal_patch_size=vision_raw["temporal_patch_size"],
            num_position_embeddings=vision_raw["num_position_embeddings"],
            out_hidden_size=vision_raw["out_hidden_size"],
            hidden_act=vision_raw["hidden_act"],
            deepstack_visual_indexes=tuple(vision_raw["deepstack_visual_indexes"]),
        )
        return cls(
            text=text,
            vision=vision,
            image_token_id=raw["image_token_id"],
            video_token_id=raw["video_token_id"],
            vision_start_token_id=raw["vision_start_token_id"],
            vision_end_token_id=raw["vision_end_token_id"],
            vision_seq_pad_multiple=raw.get("vision_seq_pad_multiple", 1),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            architectures=tuple(raw.get("architectures", ())),
            model_type=raw.get("model_type", "moss_vl"),
        )
