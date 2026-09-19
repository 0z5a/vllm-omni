# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import cast

import torch

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


class DistributedAutoencoderKLQwenImageLayered(AutoencoderKLQwenImage, DistributedVaeMixin):
    """Parallel encoding with Layered's vendored VAE numerics and cache semantics."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: object):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.init_distributed()
        return model

    def _encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        tasks: list[TileTask] = []
        for i in range(0, x.shape[3], self.tile_sample_stride_height):
            for j in range(0, x.shape[4], self.tile_sample_stride_width):
                tile = x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                coord = (i // self.tile_sample_stride_height, j // self.tile_sample_stride_width)
                tasks.append(TileTask(len(tasks), coord, tile, tile.numel()))
        return tasks, GridSpec(
            split_dims=(3, 4),
            grid_shape=tuple(c + 1 for c in tasks[-1].grid_coord),
            tile_spec={"height": x.shape[3], "width": x.shape[4]},
            output_dtype=self.dtype,
        )

    def _encode_tile(self, task: TileTask) -> torch.Tensor:
        x = cast(torch.Tensor, task.tensor)
        self.clear_cache()
        chunks: list[torch.Tensor] = []
        # Preserve the vendored encoder's one-frame prefix and four-frame chunks.
        for k in range(1 + (x.shape[2] - 1) // 4):
            self._enc_conv_idx = [0]
            chunk = x[:, :, :1] if k == 0 else x[:, :, 1 + 4 * (k - 1) : 1 + 4 * k]
            chunk = self.encoder(chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            chunks.append(self.quant_conv(chunk))
        self.clear_cache()
        return torch.cat(chunks, dim=2)

    def _encode_tile_merge(self, outputs: dict[tuple[int, ...], torch.Tensor], grid: GridSpec) -> torch.Tensor:
        scale = self.spatial_compression_ratio
        stride_h = self.tile_sample_stride_height // scale
        stride_w = self.tile_sample_stride_width // scale
        blend_h = self.tile_sample_min_height // scale - stride_h
        blend_w = self.tile_sample_min_width // scale - stride_w
        rows: list[torch.Tensor] = []
        for i in range(grid.grid_shape[0]):
            row: list[torch.Tensor] = []
            for j in range(grid.grid_shape[1]):
                tile = outputs[(i, j)]
                if i > 0:
                    tile = self.blend_v(outputs[(i - 1, j)], tile, blend_h)
                if j > 0:
                    tile = self.blend_h(outputs[(i, j - 1)], tile, blend_w)
                row.append(tile[:, :, :, :stride_h, :stride_w])
            rows.append(torch.cat(row, dim=-1))
        return torch.cat(rows, dim=3)[:, :, :, : grid.tile_spec["height"] // scale, : grid.tile_spec["width"] // scale]

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)
        self.clear_cache()
        result = self.distributed_executor.execute(
            x,
            DistributedOperator(self._encode_tile_split, self._encode_tile, self._encode_tile_merge),
            broadcast_result=True,
        )
        self.clear_cache()
        return result
