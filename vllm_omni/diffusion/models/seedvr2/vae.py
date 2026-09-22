# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SeedVR, Copyright 2025 ByteDance Ltd., Apache-2.0.
"""SeedVR2's causal s8/c16/t4 VAE, with released checkpoint names.

This implementation processes a whole clip. Group normalization and attention
operate independently on each frame; only the causal convolutions mix time.
"""

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _frame_norm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    frames = x.shape[2]
    x = norm(rearrange(x, "b c t h w -> (b t) c h w"))
    return rearrange(x, "(b t) c h w -> b c t h w", t=frames)


class CausalConv3d(nn.Conv3d):
    """Replicate the first frame to pad only the past temporal context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, (0, padding[1], padding[2]))
        self.temporal_padding = 2 * padding[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_padding:
            head = x[:, :, :1].expand(-1, -1, self.temporal_padding, -1, -1)
            x = torch.cat((head, x), dim=2)
        return super().forward(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = CausalConv3d(in_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = CausalConv3d(out_channels, out_channels)
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, (1, 1, 1), padding=(0, 0, 0))
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(_frame_norm(self.norm1, x)))
        h = self.conv2(F.silu(_frame_norm(self.norm2, h)))
        return (self.conv_shortcut(x) if self.conv_shortcut is not None else x) + h


class Downsample3d(nn.Module):
    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            channels,
            channels,
            (3 if temporal else 1, 3, 3),
            (2 if temporal else 1, 2, 2),
            (1 if temporal else 0, 0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample3d(nn.Module):
    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.temporal_ratio = 2 if temporal else 1
        self.upscale_conv = nn.Conv3d(channels, channels * 4 * self.temporal_ratio, kernel_size=1)
        self.conv = CausalConv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            self.upscale_conv(x),
            "b (x y z c) t h w -> b c (t z) (h x) (w y)",
            x=2,
            y=2,
            z=self.temporal_ratio,
        )
        if self.temporal_ratio == 2:
            x = torch.cat((x[:, :, :1], x[:, :, 2:]), dim=2)
        return self.conv(x)


class EncoderBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, index: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock3d(in_channels, out_channels, groups), ResnetBlock3d(out_channels, out_channels, groups)]
        )
        self.downsamplers = nn.ModuleList([Downsample3d(out_channels, temporal=index >= 1)] if index < 3 else [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, index: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock3d(in_channels, out_channels, groups)]
            + [ResnetBlock3d(out_channels, out_channels, groups) for _ in range(2)]
        )
        self.upsamplers = nn.ModuleList([Upsample3d(out_channels, temporal=index < 2)] if index < 3 else [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


class MidBlock3d(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3d(channels, channels, groups) for _ in range(2)])
        self.attentions = nn.ModuleList(
            [
                Attention(
                    channels,
                    heads=1,
                    dim_head=channels,
                    eps=1e-6,
                    norm_num_groups=groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[2]
        x = self.resnets[0](x)
        x = self.attentions[0](rearrange(x, "b c t h w -> (b t) c h w"))
        return self.resnets[1](rearrange(x, "(b t) c h w -> b c t h w", t=frames))


class Encoder3d(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], groups: int, latent_channels: int) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(3, channels[0])
        self.down_blocks = nn.ModuleList(
            [EncoderBlock3d(channels[max(0, i - 1)], width, groups, i) for i, width in enumerate(channels)]
        )
        self.mid_block = MidBlock3d(channels[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, channels[-1], eps=1e-6)
        self.conv_out = CausalConv3d(channels[-1], 2 * latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(_frame_norm(self.conv_norm_out, x)))


class Decoder3d(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], groups: int, latent_channels: int) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(latent_channels, channels[-1])
        self.mid_block = MidBlock3d(channels[-1], groups)
        widths = tuple(reversed(channels))
        self.up_blocks = nn.ModuleList(
            [DecoderBlock3d(widths[max(0, i - 1)], width, groups, i) for i, width in enumerate(widths)]
        )
        self.conv_norm_out = nn.GroupNorm(groups, channels[0], eps=1e-6)
        self.conv_out = CausalConv3d(channels[0], 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(_frame_norm(self.conv_norm_out, x)))


class SeedVR2VAE(nn.Module):
    """Whole-clip VAE; sampling uses the caller's request-local generator."""

    spatial_scale_factor = 8
    temporal_scale_factor = 4

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (128, 256, 512, 512),
        groups: int = 32,
        latent_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3d(channels, groups, latent_channels)
        self.decoder = Decoder3d(channels, groups, latent_channels)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
