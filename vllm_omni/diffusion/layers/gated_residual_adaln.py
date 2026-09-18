# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen-Image's attention residual followed by non-affine AdaLayerNorm.

Both the updated residual and the MLP input are live outputs. Two pointwise
kernels fuse the residual/FP32 cast and the norm-output cast/modulation around
PyTorch's native FP32 LayerNorm. Keeping the native reduction matters: even a
single FP32 ULP can change BF16 rounding and accumulate across denoising steps.
Every low precision operation retains its eager rounding boundary.
"""

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _native_layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def _native_layer_norm_fake(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


# A compile boundary prevents Inductor from replacing the native reduction with
# a different summation order. The pointwise Triton launches remain traceable.
_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "gated_residual_native_layer_norm"):
    direct_register_custom_op(
        op_name="gated_residual_native_layer_norm",
        op_func=_native_layer_norm,
        fake_impl=_native_layer_norm_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


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
    def _gated_residual_cast_kernel(
        x_ptr,
        branch_ptr,
        gate_ptr,
        residual_out_ptr,
        norm_input_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        branch_stride_b: tl.constexpr,
        branch_stride_s: tl.constexpr,
        gate_stride_b: tl.constexpr,
        block_size: tl.constexpr,
        cast_output: tl.constexpr,
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
        tl.store(residual_out_ptr + row * hidden_size + d, residual, mask)
        if cast_output:
            tl.store(norm_input_ptr + row * hidden_size + d, residual.to(tl.float32), mask)

    @triton.jit
    def _cast_modulate_kernel(
        normalized_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        scale_stride_b: tl.constexpr,
        shift_stride_b: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        batch = row // seq_len
        d = tl.arange(0, block_size)
        mask = d < hidden_size
        dtype = out_ptr.dtype.element_ty
        normalized = tl.load(normalized_ptr + row * hidden_size + d, mask, other=0).to(dtype).to(tl.float32)
        scale = tl.load(scale_ptr + batch * scale_stride_b + d, mask, other=0).to(tl.float32)
        shift = tl.load(shift_ptr + batch * shift_stride_b + d, mask, other=0).to(tl.float32)
        factor = _add_rn(tl.full((), 1.0, tl.float32), scale).to(dtype).to(tl.float32)
        modulated = _mul_rn(normalized, factor).to(dtype).to(tl.float32)
        y = _add_rn(modulated, shift).to(dtype)
        tl.store(out_ptr + row * hidden_size + d, y, mask)


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
    Autograd retains the caller's original expression. Two pointwise Triton
    kernels surround native FP32 LayerNorm; torch.compile preserves that
    reduction through a functional custom op. There is no single-kernel
    reduction fast path or error-catching fallback after selection.
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
    norm_input = torch.empty_like(r, dtype=torch.float32) if residual.dtype != torch.float32 else r
    y = torch.empty_like(r)
    block_size = triton.next_power_of_2(d)
    num_warps = 4 if d <= 2048 else 8
    _gated_residual_cast_kernel[(b * s,)](
        residual,
        branch,
        gate,
        r,
        norm_input,
        s,
        d,
        residual.stride(0),
        residual.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        block_size,
        residual.dtype != torch.float32,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    if torch.compiler.is_compiling():
        normalized = torch.ops.vllm_omni.gated_residual_native_layer_norm(norm_input, eps)
    else:
        normalized = _native_layer_norm(norm_input, eps)
    _cast_modulate_kernel[(b * s,)](
        normalized,
        scale,
        shift,
        y,
        s,
        d,
        scale.stride(0),
        shift.stride(0),
        block_size,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return r, y
