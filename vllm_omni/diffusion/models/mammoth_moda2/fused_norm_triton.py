# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only implementation; imported lazily by the guarded dispatcher."""
# ruff: noqa: N803

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_norm_kernel(
    x_ptr,
    weight_ptr,
    condition_ptr,
    residual_ptr,
    output_ptr,
    sequence_length,
    hidden_size,
    eps,
    x_batch_stride,
    x_sequence_stride,
    condition_batch_stride,
    residual_batch_stride,
    residual_sequence_stride,
    GATED_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // sequence_length
    token = row % sequence_length
    column = tl.arange(0, BLOCK)
    mask = column < hidden_size
    dtype = x_ptr.dtype.element_ty
    x = tl.load(x_ptr + batch * x_batch_stride + token * x_sequence_stride + column, mask, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    normalized = (x * tl.rsqrt(variance + eps)).to(dtype).to(tl.float32)
    weight = tl.load(weight_ptr + column, mask, other=0).to(tl.float32)
    # Qwen2RMSNorm casts normalized activations BEFORE multiplying the weight.
    affine = (normalized * weight).to(dtype).to(tl.float32)
    condition = tl.load(condition_ptr + batch * condition_batch_stride + column, mask, other=0).to(tl.float32)
    if GATED_RESIDUAL:
        residual = tl.load(
            residual_ptr + batch * residual_batch_stride + token * residual_sequence_stride + column, mask, other=0
        ).to(tl.float32)
        branch = (condition * affine).to(dtype).to(tl.float32)
        output = residual + branch
    else:
        scale = (1.0 + condition).to(dtype).to(tl.float32)
        output = affine * scale
    tl.store(output_ptr + row * hidden_size + column, output, mask)


def launch_fused_norm(x, weight, condition, eps, *, residual=None):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _fused_norm_kernel[(x.shape[0] * x.shape[1],)](
        x,
        weight,
        condition,
        residual if residual is not None else x,
        output,
        x.shape[1],
        x.shape[2],
        eps,
        x.stride(0),
        x.stride(1),
        condition.stride(0),
        residual.stride(0) if residual is not None else 0,
        residual.stride(1) if residual is not None else 0,
        GATED_RESIDUAL=residual is not None,
        BLOCK=triton.next_power_of_2(x.shape[2]),
        num_warps=4 if x.shape[2] <= 4096 else 8,
        enable_fp_fusion=False,
    )
    return output
