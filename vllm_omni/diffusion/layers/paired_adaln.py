# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental horizontal fusion of Qwen's independent image/text streams.

Each stream keeps its own native FP32 LayerNorm with its original shape and
epsilon. Only pointwise launches are shared. Outputs own independent storage;
there is no concatenation, cached buffer or mutation of a caller input.
"""

import os

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.layers.gated_residual_adaln import (
    _native_norm_boundary,
    _pointwise_config,
    _supports_pointwise,
)

if HAS_TRITON:
    from vllm_omni.diffusion.layers.gated_residual_adaln import (
        _cast_modulate_kernel,
        _gated_residual_cast_kernel,
    )

    @triton.jit
    def _cast_input_tile(x, out, offset: tl.constexpr, spec: tl.constexpr):
        s, d, stride_b, stride_s, block, tiles = spec
        pid = tl.program_id(0).to(tl.int64) - offset
        row = pid // tiles
        col = pid % tiles * block + tl.arange(0, block)
        value = tl.load(x + row // s * stride_b + row % s * stride_s + col, col < d, other=0)
        tl.store(out + row * d + col, value.to(tl.float32), col < d)

    @triton.jit
    def _pair_cast_input_kernel(x0, x1, out0, out1, split: tl.constexpr, spec0: tl.constexpr, spec1: tl.constexpr):
        if tl.program_id(0) < split:
            _cast_input_tile(x0, out0, 0, spec0)
        else:
            _cast_input_tile(x1, out1, split, spec1)

    @triton.jit
    def _pair_residual_kernel(
        x0,
        branch0,
        gate0,
        out0,
        norm0,
        x1,
        branch1,
        gate1,
        out1,
        norm1,
        split: tl.constexpr,
        spec0: tl.constexpr,
        spec1: tl.constexpr,
        cast_output: tl.constexpr,
    ):
        if tl.program_id(0) < split:
            s, d, xb, xs, bb, bs, gb, block, tiles = spec0
            _gated_residual_cast_kernel(
                x0,
                branch0,
                gate0,
                out0,
                norm0,
                s,
                d,
                xb,
                xs,
                bb,
                bs,
                gb,
                block,
                cast_output,
                tiles,
                0,
            )
        else:
            s, d, xb, xs, bb, bs, gb, block, tiles = spec1
            _gated_residual_cast_kernel(
                x1,
                branch1,
                gate1,
                out1,
                norm1,
                s,
                d,
                xb,
                xs,
                bb,
                bs,
                gb,
                block,
                cast_output,
                tiles,
                split,
            )

    @triton.jit
    def _pair_modulate_kernel(
        norm0,
        scale0,
        shift0,
        out0,
        norm1,
        scale1,
        shift1,
        out1,
        split: tl.constexpr,
        spec0: tl.constexpr,
        spec1: tl.constexpr,
    ):
        if tl.program_id(0) < split:
            s, d, scale_b, shift_b, block, tiles = spec0
            _cast_modulate_kernel(norm0, scale0, shift0, out0, s, d, scale_b, shift_b, block, tiles, 0)
        else:
            s, d, scale_b, shift_b, block, tiles = spec1
            _cast_modulate_kernel(norm1, scale1, shift1, out1, s, d, scale_b, shift_b, block, tiles, split)


def _supports_pair(x0, x1, branch0, branch1, modulation0, modulation1) -> bool:
    return (
        os.environ.get("VLLM_OMNI_QWEN_ADALN_PAIR", "0") == "1"
        and not torch.is_grad_enabled()
        and x0.device == x1.device
        and x0.dtype == x1.dtype
        and _supports_pointwise(x0, branch0, modulation0)
        and _supports_pointwise(x1, branch1, modulation1)
    )


def _launch_shape(x):
    b, s, d = x.shape
    block, _, tiles = _pointwise_config(b, s, d)
    return block, tiles, b * s * tiles


def _cast_spec(x):
    block, tiles, _ = _launch_shape(x)
    return x.shape[1], x.shape[2], x.stride(0), x.stride(1), block, tiles


def _residual_spec(x, branch, gate):
    block, tiles, _ = _launch_shape(x)
    return (
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        block,
        tiles,
    )


def _modulate_spec(x, scale, shift):
    block, tiles, _ = _launch_shape(x)
    return x.shape[1], x.shape[2], scale.stride(0), shift.stride(0), block, tiles


def _empty(x, dtype=None):
    return torch.empty(x.shape, dtype=x.dtype if dtype is None else dtype, device=x.device)


def _modulate_pair(normalized0, normalized1, x0, x1, scale0, shift0, scale1, shift1):
    out0, out1 = _empty(x0), _empty(x1)
    _, _, split = _launch_shape(x0)
    _, _, right = _launch_shape(x1)
    _pair_modulate_kernel[(split + right,)](
        normalized0,
        scale0,
        shift0,
        out0,
        normalized1,
        scale1,
        shift1,
        out1,
        split,
        _modulate_spec(x0, scale0, shift0),
        _modulate_spec(x1, scale1, shift1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out0, out1


def try_paired_native_adaln(
    x0: torch.Tensor,
    x1: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    eps0: float,
    eps1: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pair norm1 casts/modulation; the two native reductions remain separate."""
    if not _supports_pair(x0, x1, None, None, (scale0, shift0), (scale1, shift1)):
        return None
    if x0.dtype == torch.float32:
        norm0, norm1 = x0, x1
    else:
        norm0, norm1 = _empty(x0, torch.float32), _empty(x1, torch.float32)
        _, _, split = _launch_shape(x0)
        _, _, right = _launch_shape(x1)
        _pair_cast_input_kernel[(split + right,)](
            x0,
            x1,
            norm0,
            norm1,
            split,
            _cast_spec(x0),
            _cast_spec(x1),
            num_warps=4,
        )
    norm0, norm1 = _native_norm_boundary(norm0, eps0), _native_norm_boundary(norm1, eps1)
    return _modulate_pair(norm0, norm1, x0, x1, scale0, shift0, scale1, shift1)


def try_paired_gated_residual_adaln(
    left: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    eps0: float,
    eps1: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return image residual/norm2 and text residual/norm2, with no aliases."""
    x0, branch0, gate0, scale0, shift0 = left
    x1, branch1, gate1, scale1, shift1 = right
    if not _supports_pair(x0, x1, branch0, branch1, (gate0, scale0, shift0), (gate1, scale1, shift1)):
        return None
    out0, out1 = _empty(x0), _empty(x1)
    # Distinct write arguments are required for Inductor functionalization.
    norm0, norm1 = _empty(x0, torch.float32), _empty(x1, torch.float32)
    _, _, split = _launch_shape(x0)
    _, _, count1 = _launch_shape(x1)
    _pair_residual_kernel[(split + count1,)](
        x0,
        branch0,
        gate0,
        out0,
        norm0,
        x1,
        branch1,
        gate1,
        out1,
        norm1,
        split,
        _residual_spec(x0, branch0, gate0),
        _residual_spec(x1, branch1, gate1),
        True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    norm0, norm1 = _native_norm_boundary(norm0, eps0), _native_norm_boundary(norm1, eps1)
    y0, y1 = _modulate_pair(norm0, norm1, x0, x1, scale0, shift0, scale1, shift1)
    return out0, y0, out1, y1


def try_paired_gated_residual(
    x0: torch.Tensor,
    branch0: torch.Tensor,
    gate0: torch.Tensor,
    x1: torch.Tensor,
    branch1: torch.Tensor,
    gate1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One pointwise launch for the two final MLP residuals."""
    if not _supports_pair(x0, x1, branch0, branch1, (gate0,), (gate1,)):
        return None
    out0, out1 = _empty(x0), _empty(x1)
    _, _, split = _launch_shape(x0)
    _, _, right = _launch_shape(x1)
    _pair_residual_kernel[(split + right,)](
        x0,
        branch0,
        gate0,
        out0,
        None,
        x1,
        branch1,
        gate1,
        out1,
        None,
        split,
        _residual_spec(x0, branch0, gate0),
        _residual_spec(x1, branch1, gate1),
        False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out0, out1
