# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MOSS-VL-Realtime vision tower.

Packed-sequence ViT with a 2-D RoPE, per-frame attention segments, deepstack
feature taps, and a 2x2 spatial-merge projector. Parameter names match the
checkpoint keys under ``model.visual.`` exactly, so weights load without a
mapping table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MossVLVisionConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((q * cos) + (rotate_half(q) * sin)).to(q_dtype), ((k * cos) + (rotate_half(k) * sin)).to(k_dtype)


class MossVLVisionPatchEmbed(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.hidden_size
        kernel = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).view(-1, self.embed_dim)


class MossVLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", self.build_inv_freq(), persistent=False)

    def build_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))

    def materialize(self, device: torch.device) -> None:
        self.inv_freq = self.build_inv_freq().to(device)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MossVLVisionMLP(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class MossVLVisionAttention(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self, hidden_states: torch.Tensor, lengths: list[int], cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )
        query, key = apply_rotary_vision(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        # Attention is block-diagonal over frames: the reference splits the same
        # tensors with cu_seqlens and runs one sdpa call per segment.
        outputs = []
        start = 0
        for length in lengths:
            stop = start + length
            segment = F.scaled_dot_product_attention(
                query[:, :, start:stop], key[:, :, start:stop], value[:, :, start:stop], scale=self.scaling
            )
            outputs.append(segment)
            start = stop
        attn_output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class MossVLVisionBlock(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = MossVLVisionAttention(cfg)
        self.mlp = MossVLVisionMLP(cfg)

    def forward(
        self, hidden_states: torch.Tensor, lengths: list[int], cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), lengths, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MossVLVisionPatchMerger(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig, num_deepstack_features: int) -> None:
        super().__init__()
        base_hidden_size = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.input_hidden_size = base_hidden_size * (1 + num_deepstack_features)
        self.hidden_size = cfg.hidden_size
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden_size, eps=1e-6) for _ in range(1 + num_deepstack_features))
        self.linear_fc1 = nn.Linear(self.input_hidden_size, self.input_hidden_size)
        self.linear_fc2 = nn.Linear(self.input_hidden_size, cfg.out_hidden_size)

    def forward(self, last_hidden_state: torch.Tensor, deepstack_features: list[torch.Tensor]) -> torch.Tensor:
        stacked = [self.norms[i](feat) for i, feat in enumerate([last_hidden_state, *deepstack_features])]
        x = torch.cat(stacked, dim=-1).view(-1, self.input_hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class MossVLVisionTower(nn.Module):
    def __init__(self, cfg: MossVLVisionConfig) -> None:
        super().__init__()
        self.config = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.spatial_merge_unit = cfg.spatial_merge_size**2
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = tuple(cfg.deepstack_visual_indexes)
        self.patch_embed = MossVLVisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.rotary_pos_emb = MossVLVisionRotaryEmbedding(cfg.head_dim // 2)
        self.blocks = nn.ModuleList(MossVLVisionBlock(cfg) for _ in range(cfg.depth))
        self.merger = MossVLVisionPatchMerger(cfg, len(self.deepstack_visual_indexes))

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        return freq_table[pos_ids].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for _t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - h_floor, w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        device = self.pos_embed.weight.device
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds.sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])

        merge_size = self.spatial_merge_size
        permuted = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pos_embed)
        return torch.cat(permuted)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: [sum(t*h*w), C*tps*ps*ps]; grid_thw: [num_media, 3] in patch units."""
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        rotary = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # One attention segment per frame, in media order.
        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, lengths, cos, sin)
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(hidden_states)
        return self.merger(hidden_states, deepstack_features)
