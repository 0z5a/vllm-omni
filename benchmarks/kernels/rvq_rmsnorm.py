# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in dual-output RVQ/first-RMSNorm compiled benchmark primitive.

One program owns the complete hidden vector of one frame. Q is reduced before
the mandatory raw dtype rounding; H is reduced only after that rounding. This
launch layout has just one CTA at B=T=1 and can increase register pressure.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_omni.platforms import current_omni_platform


def native_region(codes, weight, offsets, gamma, eps):
    import torch
    import torch.nn.functional as F

    raw = F.embedding(codes + offsets, weight).mean(1)
    input_dtype = raw.dtype
    hidden_states = raw.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized = gamma * hidden_states.to(input_dtype)
    return raw, normalized


@triton.jit
def _rvq_rmsnorm_dual_kernel(
    codes,
    weight,
    offsets,
    gamma,
    raw_output,
    normalized_output,
    stride_b: tl.constexpr,
    stride_q: tl.constexpr,
    stride_t: tl.constexpr,
    stride_offset: tl.constexpr,
    stride_gamma: tl.constexpr,
    time: tl.constexpr,
    hidden: tl.constexpr,
    entries: tl.constexpr,
    block_h: tl.constexpr,
    eps: tl.constexpr,
):
    frame = tl.program_id(0).to(tl.int64)
    batch, token = frame // time, frame % time
    quantizer = tl.arange(0, 16).to(tl.int64)
    channel = tl.arange(0, block_h).to(tl.int64)
    code = tl.load(codes + batch * stride_b + quantizer * stride_q + token * stride_t)
    offset = tl.load(offsets + quantizer * stride_offset)
    # Preserve native int32/int64 addition promotion/overflow before widening
    # the combined row for table address arithmetic.
    row = (code + offset).to(tl.int64)
    valid_row = (row >= 0) & (row < entries)
    tl.device_assert(valid_row, "RVQ embedding index out of range")
    embedding = tl.load(
        weight + row[:, None] * hidden + channel[None, :], valid_row[:, None] & (channel[None, :] < hidden), 0
    ).to(tl.float32)
    mean = tl.sum(embedding, axis=0) * (1.0 / 16.0)
    # The residual's rounding is also the input boundary for RMS statistics.
    raw_low = mean.to(raw_output.dtype.element_ty)
    raw_float = raw_low.to(tl.float32)
    squares = tl.where(channel < hidden, raw_float * raw_float, 0.0)
    variance = tl.sum(squares, axis=0) / hidden
    epsilon_float = tl.full((), eps, tl.float32)
    inverse_rms = tl.rsqrt(variance + epsilon_float)
    normalized_low = (raw_float * inverse_rms).to(raw_output.dtype.element_ty)
    scale = tl.load(gamma + channel * stride_gamma, channel < hidden, 0).to(tl.float32)
    # HF casts before gamma multiplication, rather than only at the end.
    normalized = normalized_low.to(tl.float32) * scale
    output_offset = frame * hidden + channel
    tl.store(raw_output + output_offset, raw_low, channel < hidden)
    tl.store(normalized_output + output_offset, normalized, channel < hidden)


def supported_epsilon(eps):
    # PyTorch converts Python integers directly to its scalar representation.
    # Above 2**53, this wrapper's later int->Python-float conversion can double
    # round before fp32, even below int64's limit. Larger ints stay native.
    return (type(eps) is int and 0 <= eps <= 2**53) or (type(eps) is float and 0 <= eps <= 3.4028234663852886e38)


def inspect_inputs(codes, weight, offsets, gamma, eps):
    """Validate live metadata once and reuse only immutable values downstream."""
    tensors = (codes, weight, offsets, gamma)
    if not torch.compiler.is_compiling() and any(tensor.is_neg() or tensor.is_conj() for tensor in tensors):
        return None
    if (
        torch.is_grad_enabled()
        or not all(
            type(tensor) in (torch.Tensor, torch.nn.Parameter)
            and tensor.layout == torch.strided
            and not tensor.is_nested
            for tensor in tensors
        )
        or not codes.is_cuda
        or torch.version.hip is not None
        or torch.is_autocast_enabled("cuda")
        or not supported_epsilon(eps)
    ):
        return None
    device = codes.device
    if (
        device != weight.device
        or device != offsets.device
        or device != gamma.device
        or device.index != current_omni_platform.current_device()
    ):
        return None
    code_shape, weight_shape, offset_shape, gamma_shape = codes.shape, weight.shape, offsets.shape, gamma.shape
    if (
        len(code_shape) != 3
        or len(weight_shape) != 2
        or len(offset_shape) != 3
        or len(gamma_shape) != 1
        or code_shape[1] != 16
        or offset_shape != (1, 16, 1)
    ):
        return None
    code_dtype, weight_dtype, offset_dtype, gamma_dtype = codes.dtype, weight.dtype, offsets.dtype, gamma.dtype
    if (
        code_dtype not in (torch.int32, torch.int64)
        or offset_dtype not in (torch.int32, torch.int64)
        or weight_dtype not in (torch.float16, torch.bfloat16)
        or gamma_dtype != weight_dtype
        or not weight.is_contiguous()
        or weight_shape[0] <= 0
        or not 0 < weight_shape[1] <= 2048
        or gamma_shape[0] != weight_shape[1]
    ):
        return None
    code_stride, offset_stride, gamma_stride = codes.stride(), offsets.stride(), gamma.stride()
    if any(value < 0 for values in (code_stride, offset_stride, gamma_stride) for value in values):
        return None
    batch, _, time = code_shape
    entries, hidden = weight_shape
    constants = (
        *code_stride,
        offset_stride[1],
        gamma_stride[0],
        time,
        hidden,
        entries,
        triton.next_power_of_2(hidden),
        float(eps),
    )
    # Outputs are allocated with exactly the checked weight dtype and device.
    signature = (device.index, code_dtype, weight_dtype, offset_dtype, gamma_dtype)
    return batch, time, hidden, device, weight_dtype, constants, signature


def can_use_variant(codes, weight, offsets, gamma, eps):
    return inspect_inputs(codes, weight, offsets, gamma, eps) is not None


def make_variant(num_warps):
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be four or eight")

    def entry(codes, weight, offsets, gamma, eps):
        metadata = inspect_inputs(codes, weight, offsets, gamma, eps)
        if metadata is None:
            return native_region(codes, weight, offsets, gamma, eps)
        batch, time, hidden, device, dtype, constants, signature = metadata
        raw = torch.empty((batch, time, hidden), dtype=dtype, device=device)
        normalized = torch.empty_like(raw)
        if batch != 0 and time != 0:
            _rvq_rmsnorm_dual_kernel[(batch * time, 1, 1)](
                codes, weight, offsets, gamma, raw, normalized, *constants, num_warps=num_warps, debug=True
            )
        return raw, normalized

    # Both benchmark arms use fullgraph=True and dynamic=False. Dynamo guards
    # live shape/stride/dtype/device/epsilon and Inductor owns normal dispatch.
    # No task-managed cache retains tensor pointers or streams.
    compiled = torch.compile(entry, fullgraph=True, dynamic=False)
    compiled.__name__ = f"rvq_rmsnorm_w{num_warps}"

    def call(codes, weight, offsets, gamma, eps):
        # Dynamo 2.13 cannot trace is_neg/is_conj booleans. Check these four
        # live tensors outside its graph; never erase or resolve a lazy view.
        if any(tensor.is_neg() or tensor.is_conj() for tensor in (codes, weight, offsets, gamma)):
            return native_region(codes, weight, offsets, gamma, eps)
        return compiled(codes, weight, offsets, gamma, eps)

    call.__name__ = compiled.__name__
    return call
