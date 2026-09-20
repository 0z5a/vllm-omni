# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MOSS-VL-Realtime text stack, positions, and cross-attention.

Single-device eager implementation for bounded BF16 workloads. It reads the
published checkpoint weights with no checkpoint-provided Python and no
Transformers runtime: the only entry points are eager matmuls, softmax
attention, and the checkpoint's mRoPE / visibility semantics.

The 48 text layers are not homogeneous: 36 carry causal self-attention and 12
(the indexes in ``text_config.cross_attention_layers``) carry cross-attention to
vision features plus tanh-gated residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MossVLConfig, MossVLTextConfig
from .vision import MossVLVisionTower


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    batch, heads, seq_len, head_dim = x.shape
    return x[:, :, None].expand(batch, heads, groups, seq_len, head_dim).reshape(batch, heads * groups, seq_len, head_dim)


class MossVLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class MossVLTextRotaryEmbedding(nn.Module):
    """3-axis interleaved mRoPE (``mrope_section`` = 24/20/20 for this checkpoint)."""

    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.theta = cfg.rope_theta
        self.mrope_section = cfg.mrope_section
        self.register_buffer("inv_freq", self.build_inv_freq(), persistent=False)

    def build_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

    def materialize(self, device: torch.device) -> None:
        self.inv_freq = self.build_inv_freq().to(device)

    def interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        out = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[dim, ..., idx]
        return out

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
        inv = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        with torch.autocast(x.device.type, enabled=False):
            freqs = (inv.to(x.device) @ pos.to(x.device)).transpose(2, 3)
            freqs = self.interleave(freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
        return cos, sin


class MossVLTextMLP(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionKVCache:
    """Vision key/value store for cross-attention layers.

    One slot per cross-attention layer index. Frames are appended as they are
    encoded, so a decode step reuses keys that were already rotated once.
    """

    def __init__(self, num_slots: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * num_slots
        self.values: list[torch.Tensor | None] = [None] * num_slots

    def is_empty(self, layer_idx: int) -> bool:
        return self.keys[layer_idx] is None

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = keys, values
        else:
            self.keys[layer_idx] = torch.cat((self.keys[layer_idx], keys), dim=2)
            self.values[layer_idx] = torch.cat((self.values[layer_idx], values), dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        if keys is None or values is None:
            raise ValueError(f"cross-attention layer {layer_idx} has no vision keys")
        return keys, values

    def reset(self) -> None:
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)


class TextKVCache:
    """Contiguous self-attention cache, one slot per text layer."""

    def __init__(self, num_slots: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * num_slots
        self.values: list[torch.Tensor | None] = [None] * num_slots

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = keys, values
        else:
            self.keys[layer_idx] = torch.cat((self.keys[layer_idx], keys), dim=2)
            self.values[layer_idx] = torch.cat((self.values[layer_idx], values), dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def length(self, layer_idx: int) -> int:
        keys = self.keys[layer_idx]
        return 0 if keys is None else keys.shape[2]

    def reset(self) -> None:
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)


class MossVLTextSelfAttention(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_key_value_groups
        self.scaling = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = MossVLTextRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = MossVLTextRMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: TextKVCache,
        layer_idx: int,
        offset: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_norm(self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        key, value = cache.update(layer_idx, key, value)
        key = repeat_kv(key, self.groups)
        value = repeat_kv(value, self.groups)

        if offset == 0 and seq_len > 1:
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=self.scaling)
        elif seq_len == 1:
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scaling)
        else:
            mask = torch.ones(seq_len, key.shape[2], dtype=torch.bool, device=query.device).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))


class MossVLTextCrossAttention(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_key_value_groups
        self.scaling = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = MossVLTextRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = MossVLTextRMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        vision_states: torch.Tensor | None,
        vision_cos: torch.Tensor | None,
        vision_sin: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        cache: VisionKVCache,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_norm(self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        query = apply_rotary(query, query_cos, query_sin)

        if vision_states is not None:
            key = self.k_norm(self.k_proj(vision_states).view(batch, -1, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            value = self.v_proj(vision_states).view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            key = apply_rotary(key, vision_cos, vision_sin)
            key, value = cache.update(layer_idx, key, value)
        else:
            key, value = cache.get(layer_idx)

        key = repeat_kv(key, self.groups)
        value = repeat_kv(value, self.groups)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))


class MossVLSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.self_attn = MossVLTextSelfAttention(cfg)
        self.mlp = MossVLTextMLP(cfg)
        self.input_layernorm = MossVLTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = MossVLTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        text_cache: TextKVCache,
        layer_idx: int,
        offset: int,
        row_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, text_cache, layer_idx, offset)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class MossVLCrossAttentionDecoderLayer(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.cross_attn = MossVLTextCrossAttention(cfg)
        self.mlp = MossVLTextMLP(cfg)
        self.input_layernorm = MossVLTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = MossVLTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        vision_states: torch.Tensor | None,
        vision_cos: torch.Tensor | None,
        vision_sin: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        cache: VisionKVCache,
        layer_idx: int,
        row_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            query_cos,
            query_sin,
            vision_states,
            vision_cos,
            vision_sin,
            attention_mask,
            cache,
            layer_idx,
        )
        if row_mask is not None:
            hidden_states = row_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if row_mask is not None:
            hidden_states = row_mask * hidden_states
        return residual + self.cross_attn_mlp_gate.tanh() * hidden_states


class MossVLTextModel(nn.Module):
    def __init__(self, cfg: MossVLTextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.cross_attention_layers = set(cfg.cross_attention_layers)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            MossVLCrossAttentionDecoderLayer(cfg) if idx in self.cross_attention_layers else MossVLSelfAttentionDecoderLayer(cfg)
            for idx in range(cfg.num_hidden_layers)
        )
        self.norm = MossVLTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = MossVLTextRotaryEmbedding(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        text_cache: TextKVCache,
        vision_cache: VisionKVCache,
        offset: int,
        vision_states: torch.Tensor | None,
        vision_position_ids: torch.Tensor | None,
        cross_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        vision_cos = vision_sin = None
        if vision_states is not None:
            vision_cos, vision_sin = self.rotary_emb(vision_states, vision_position_ids)

        row_mask = None
        if cross_attention_mask is not None:
            negative = torch.finfo(cross_attention_mask.dtype).min
            row_mask = (cross_attention_mask != negative).any(dim=-1, keepdim=True).to(cross_attention_mask.dtype)[:, 0]
            cross_attention_mask = cross_attention_mask * row_mask.unsqueeze(1)

        for idx, layer in enumerate(self.layers):
            if idx in self.cross_attention_layers:
                if vision_states is None and vision_cache.is_empty(idx):
                    continue
                hidden_states = layer(
                    hidden_states,
                    cos,
                    sin,
                    vision_states,
                    vision_cos,
                    vision_sin,
                    cross_attention_mask,
                    vision_cache,
                    idx,
                    row_mask,
                )
            else:
                hidden_states = layer(hidden_states, cos, sin, text_cache, idx, offset, row_mask)
        return self.norm(hidden_states)


@dataclass
class VisionTokenInfo:
    """Per-media vision layout used for position and visibility metadata."""

    grid_h: int
    grid_w: int
    num_frames: int
    start: int
    vision_tokens_per_frame: int


class MossVLNativeModel(nn.Module):
    """Weight-compatible native MOSS-VL-Realtime model."""

    def __init__(self, cfg: MossVLConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = nn.Module()
        self.model.visual = MossVLVisionTower(cfg.vision)
        self.model.language_model = MossVLTextModel(cfg.text)
        self.model.separator_token = nn.Parameter(torch.zeros(cfg.text.hidden_size))
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        self.text_cache = TextKVCache(cfg.text.num_hidden_layers)
        self.vision_cache = VisionKVCache(cfg.text.num_hidden_layers)
        self.rope_delta: torch.Tensor | None = None
        self.vision_token_info: list[VisionTokenInfo] = []

    # ---- vision -----------------------------------------------------------
    def convert_packed_to_batch(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Insert one separator token after every frame's vision tokens (batch size 1)."""
        merge = self.config.vision.spatial_merge_size
        tokens_per_media = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // (merge * merge)
        hidden_size = hidden_states.shape[-1]
        total = sum(tokens_per_media[i].item() + grid_thw[i, 0].item() for i in range(grid_thw.shape[0]))
        batch = hidden_states.new_zeros(1, total, hidden_size)
        separator = self.model.separator_token.to(hidden_states.dtype)

        token_offset = 0
        cursor = 0
        info: list[VisionTokenInfo] = []
        for media_idx in range(grid_thw.shape[0]):
            t, h, w = (int(v) for v in grid_thw[media_idx].tolist())
            num_tokens = int(tokens_per_media[media_idx].item())
            tokens_per_frame = num_tokens // t
            chunk = t * (tokens_per_frame + 1)
            frames = hidden_states[token_offset : token_offset + num_tokens].view(t, tokens_per_frame, hidden_size)
            target = batch[0, cursor : cursor + chunk].view(t, tokens_per_frame + 1, hidden_size)
            target[:, :tokens_per_frame].copy_(frames)
            target[:, tokens_per_frame] = separator
            info.append(VisionTokenInfo(grid_h=h, grid_w=w, num_frames=t, start=cursor, vision_tokens_per_frame=tokens_per_frame))
            cursor += chunk
            token_offset += num_tokens
        self.vision_token_info = info
        return batch

    # ---- positions --------------------------------------------------------
    def compute_text_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_image = input_ids == self.config.image_token_id
        cumulative = (~is_image).long().cumsum(dim=1)
        base = cumulative - (~is_image).long()
        return base.unsqueeze(0).expand(3, -1, -1).clone()

    def compute_vision_position_ids(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, num_vision_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merge = self.config.vision.spatial_merge_size
        image_positions = (input_ids[0] == self.config.image_token_id).nonzero().flatten()
        if image_positions.numel() == 0:
            # The processor injects a blank vision asset for media-less prompts, so
            # vision states can exist while the text carries no image token. The
            # reference leaves text positions untouched and masks every query in
            # that case, which makes the cross-attention layers inert.
            return torch.zeros(3, 1, num_vision_tokens, dtype=torch.long, device=input_ids.device), position_ids
        frames: list[tuple[int, int, int]] = []
        for media in self.vision_token_info:
            effective_h = media.grid_h // merge
            effective_w = media.grid_w // merge
            stride = media.vision_tokens_per_frame + 1
            for frame in range(media.num_frames):
                frames.append((effective_h, effective_w, media.start + frame * stride))
        if len(frames) != image_positions.numel():
            raise ValueError(
                f"frame metadata does not match the text: {len(frames)} frames for "
                f"{image_positions.numel()} image-pad tokens"
            )

        effective_h = torch.tensor([f[0] for f in frames], device=input_ids.device)
        effective_w = torch.tensor([f[1] for f in frames], device=input_ids.device)
        starts = torch.tensor([f[2] for f in frames], device=input_ids.device)
        max_hw = torch.maximum(effective_h, effective_w)
        shifts = max_hw + 1

        shift_map = torch.zeros(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        shift_map[image_positions] = shifts[: image_positions.numel()]
        cumulative = shift_map.cumsum(dim=0)
        t_vals = position_ids[0, 0, image_positions] + cumulative[image_positions] - shifts[: image_positions.numel()]

        position_ids = position_ids + cumulative.unsqueeze(0)
        position_ids[:, 0, image_positions] -= 1

        vision_position_ids = torch.zeros(3, 1, num_vision_tokens, dtype=torch.long, device=input_ids.device)
        for frame_id, (grid_h, grid_w, start) in enumerate(frames):
            t_val = t_vals[frame_id]
            rows = torch.arange(grid_h, device=input_ids.device).view(grid_h, 1).expand(grid_h, grid_w)
            cols = torch.arange(grid_w, device=input_ids.device).view(1, grid_w).expand(grid_h, grid_w)
            flat = slice(start, start + grid_h * grid_w)
            vision_position_ids[0, 0, flat] = t_val
            vision_position_ids[1, 0, flat] = (t_val + rows).reshape(-1)
            vision_position_ids[2, 0, flat] = (t_val + cols).reshape(-1)
            vision_position_ids[:, 0, start + grid_h * grid_w] = t_val + max_hw[frame_id]
        return vision_position_ids, position_ids

    def expand_cross_attention_mask(self, frame_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.vision_token_info:
            return None
        if self.config.vision_seq_pad_multiple != 1:
            raise ValueError(
                "vision_seq_pad_multiple != 1 is not implemented: the published checkpoint sets 1, "
                "and a padded vision sequence needs its padding columns masked explicitly"
            )
        total = sum(m.num_frames * (m.vision_tokens_per_frame + 1) for m in self.vision_token_info)
        repeats = [m.vision_tokens_per_frame + 1 for m in self.vision_token_info for _ in range(m.num_frames)]
        negative = torch.finfo(dtype).min
        frame_mask = frame_mask[:, :, :, : len(repeats)]
        float_mask = torch.zeros_like(frame_mask, dtype=dtype).masked_fill_(frame_mask, negative)
        expanded = torch.full((1, 1, frame_mask.shape[2], total), negative, dtype=dtype, device=frame_mask.device)
        expanded[:, :, :, :] = float_mask.repeat_interleave(torch.tensor(repeats, device=frame_mask.device), dim=-1)
        return expanded

    # ---- forward ----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        offset: int,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One prefill (``offset == 0``) or one decode step (``offset > 0``)."""
        vision_states = None
        vision_position_ids = None
        if pixel_values is not None:
            packed = self.model.visual(pixel_values.to(self.model.visual.patch_embed.proj.weight.dtype), grid_thw)
            vision_states = self.convert_packed_to_batch(packed, grid_thw)
            vision_position_ids, position_ids = self.compute_vision_position_ids(
                input_ids, position_ids, vision_states.shape[1]
            )
            if offset == 0:
                self.rope_delta = position_ids.max() + 1 - input_ids.shape[1]

        expanded_mask = None
        if cross_attention_mask is not None:
            dtype = self.model.language_model.embed_tokens.weight.dtype
            expanded_mask = self.expand_cross_attention_mask(cross_attention_mask, dtype)

        hidden_states = self.model.language_model(
            input_ids,
            position_ids,
            self.text_cache,
            self.vision_cache,
            offset,
            vision_states,
            vision_position_ids,
            expanded_mask,
        )
        return self.lm_head(hidden_states)

    def decode_position_ids(self, offset: int, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Positions for one decode step.

        With vision input the reference caches ``rope_deltas`` at prefill and every
        decode position is ``cache_position + rope_deltas``. A sequence that never
        produced vision states has no cached delta, and the reference then recomputes
        positions from the current token ids, i.e. a one-token decode sits at the
        start of the text position space. That rule is mirrored here instead of being
        silently corrected, so the native path stays comparable with the capture.
        """
        if self.rope_delta is None:
            if input_ids is None:
                raise ValueError("text-only decode needs the token ids that produced no vision states")
            return self.compute_text_position_ids(input_ids)
        position = torch.full((1,), offset, dtype=torch.long, device=self.rope_delta.device) + self.rope_delta
        return position.view(1, 1, 1).expand(3, 1, 1)

    def reset(self) -> None:
        self.text_cache.reset()
        self.vision_cache.reset()
        self.rope_delta = None
        self.vision_token_info = []
