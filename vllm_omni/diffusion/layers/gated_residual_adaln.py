# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen-Image's attention residual followed by non-affine AdaLayerNorm.

Both the updated residual and the MLP input are live outputs. FP32 reduction
matches the layer's ``layer_norm(x.float()).to(x.dtype)`` contract. Low precision
multiplication, residual addition, norm output, ``1 + scale``, modulation
multiplication, and shift addition each retain their eager rounding boundary.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    # Explicit PTX keeps both rounding and subnormals. CUDA libdevice rn
    # intrinsics inherit Triton's FTZ mode, including when traced by Inductor.
    @triton.jit
    def _add_rn(x, y):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn(x, y):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _gated_residual_adaln_kernel(
        x_ptr,
        branch_ptr,
        gate_ptr,
        scale_ptr,
        shift_ptr,
        residual_out_ptr,
        modulated_out_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        branch_stride_b: tl.constexpr,
        branch_stride_s: tl.constexpr,
        gate_stride_b: tl.constexpr,
        scale_stride_b: tl.constexpr,
        shift_stride_b: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        batch, token = row // seq_len, row % seq_len
        d = tl.arange(0, block_size)
        mask = d < hidden_size
        dtype = x_ptr.dtype.element_ty
        x = tl.load(x_ptr + batch * x_stride_b + token * x_stride_s + d, mask, other=0).to(tl.float32)
        branch = tl.load(branch_ptr + batch * branch_stride_b + token * branch_stride_s + d, mask, other=0).to(
            tl.float32
        )
        gate = tl.load(gate_ptr + batch * gate_stride_b + d, mask, other=0).to(tl.float32)
        # Keep separate IEEE operations even when Inductor rebuilds launch
        # options and drops enable_fp_fusion=False.
        product = _mul_rn(gate, branch).to(dtype).to(tl.float32)
        residual = _add_rn(x, product).to(dtype)
        r = residual.to(tl.float32)
        mean = tl.sum(tl.where(mask, r, 0), 0) / hidden_size
        centered = _sub_rn(r, mean)
        variance = tl.sum(tl.where(mask, _mul_rn(centered, centered), 0), 0) / hidden_size
        inv_std = tl.rsqrt(_add_rn(variance, tl.full((), eps, tl.float32)))
        normalized = _mul_rn(centered, inv_std).to(dtype).to(tl.float32)
        scale = tl.load(scale_ptr + batch * scale_stride_b + d, mask, other=0).to(tl.float32)
        shift = tl.load(shift_ptr + batch * shift_stride_b + d, mask, other=0).to(tl.float32)
        factor = _add_rn(tl.full((), 1.0, tl.float32), scale).to(dtype).to(tl.float32)
        modulated = _mul_rn(normalized, factor).to(dtype).to(tl.float32)
        y = _add_rn(modulated, shift).to(dtype)
        tl.store(residual_out_ptr + row * hidden_size + d, residual, mask)
        tl.store(modulated_out_ptr + row * hidden_size + d, y, mask)


def try_fused_gated_residual_adaln(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return two new outputs, or None so the caller uses its original layers.

    Supports CUDA FP32/BF16 [B,S,D] rows with unit channel stride, D <= 8192,
    and per-batch [B,1,D] modulation (including chunk/unsqueeze batch gaps).
    Autograd retains the caller's original expression; torch.compile can trace
    the functional Triton launch without an opaque custom-op boundary.
    This helper never catches errors after selecting its supported kernel.
    """
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or not residual.is_cuda
        or residual.ndim != 3
        or residual.dtype not in (torch.float32, torch.bfloat16)
        or residual.numel() == 0
        or not 0 < residual.shape[-1] <= 8192
    ):
        return None
    if branch.shape != residual.shape:
        return None
    b, s, d = residual.shape
    tensors = (residual, branch, gate, scale, shift)
    if any(t.device != residual.device or t.dtype != residual.dtype or t.layout != torch.strided for t in tensors):
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    if residual.stride(-1) != 1 or branch.stride(-1) != 1:
        return None
    if any(t.shape != (b, 1, d) or t.stride(-1) != 1 for t in (gate, scale, shift)):
        return None
    r = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    y = torch.empty_like(r)
    _gated_residual_adaln_kernel[(b * s,)](
        residual,
        branch,
        gate,
        scale,
        shift,
        r,
        y,
        s,
        d,
        residual.stride(0),
        residual.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        scale.stride(0),
        shift.stride(0),
        eps,
        triton.next_power_of_2(d),
        num_warps=4 if d <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return r, y
