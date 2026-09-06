# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Mammoth RMSNorm epilogues with the original PyTorch fallback.

The CUDA path preserves the Qwen2 RMSNorm cast-before-affine convention and
each BF16 pointwise rounding boundary. Reduction ordering can still differ;
this is not a bitwise-equivalence or an end-to-end performance claim.
"""

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


def _can_fuse(norm: nn.Module, x: torch.Tensor, condition: torch.Tensor, residual: torch.Tensor | None = None) -> bool:
    # Do not bypass replacement norm implementations or a user-installed hook.
    if type(norm) is not Qwen2RMSNorm or norm._forward_hooks or norm._forward_pre_hooks:
        return False
    if not x.is_cuda or torch.version.hip is not None or x.dtype not in (torch.float32, torch.bfloat16):
        return False
    if x.ndim != 3 or x.numel() == 0 or not 0 < x.shape[-1] <= 8192 or x.stride(-1) != 1:
        return False
    tensors = [x, norm.weight, condition]
    if condition.shape != (x.shape[0], x.shape[2]) or condition.stride(-1) != 1:
        return False
    if norm.weight.shape != (x.shape[2],) or not norm.weight.is_contiguous():
        return False
    if residual is not None:
        if residual.shape != x.shape or residual.stride(-1) != 1:
            return False
        tensors.append(residual)
    if any(tensor.dtype != x.dtype or tensor.device != x.device for tensor in tensors):
        return False
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        return False
    return True


def rms_norm_modulation(
    norm: nn.Module, x: torch.Tensor, scale: torch.Tensor, *, enabled: bool = False
) -> torch.Tensor:
    """Compute ``norm(x) * (1 + scale[:, None])`` without changing weights."""
    if not enabled or not _can_fuse(norm, x, scale):
        return norm(x) * (1 + scale[:, None])
    from .fused_norm_triton import launch_fused_norm

    return launch_fused_norm(x, norm.weight, scale, norm.variance_epsilon)


def rms_norm_gated_residual(
    norm: nn.Module,
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    """Compute ``residual + tanh(gate[:, None]) * norm(x)`` out of place."""
    if not enabled or not _can_fuse(norm, x, gate, residual):
        return residual + gate[:, None].tanh() * norm(x)
    from .fused_norm_triton import launch_fused_norm

    # Keep PyTorch's tanh and its dtype rounding. This tiny B*H tensor avoids
    # substituting a different transcendental approximation in every token.
    return launch_fused_norm(x, norm.weight, gate.tanh(), norm.variance_epsilon, residual=residual)
