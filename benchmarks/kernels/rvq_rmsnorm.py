# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in tiled RVQ and native-order first RMSNorm benchmark primitive.

RVQ programs own 128-channel tiles and preserve the native Q16 arithmetic.
A second kernel reads the rounded residual and preserves native RMS statistics.
Both outputs remain observable; no cross-program synchronization is assumed.
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
def _rvq_row(
    codes,
    weight,
    offsets,
    batch,
    token,
    channel,
    stride_b: tl.constexpr,
    stride_q: tl.constexpr,
    stride_t: tl.constexpr,
    stride_offset: tl.constexpr,
    hidden: tl.constexpr,
    entries: tl.constexpr,
    quantizer: tl.constexpr,
):
    code = tl.load(codes + batch * stride_b + quantizer * stride_q + token * stride_t)
    offset = tl.load(offsets + quantizer * stride_offset)
    row = (code + offset).to(tl.int64)
    valid = (row >= 0) & (row < entries)
    tl.device_assert(valid, "RVQ embedding index out of range")
    return tl.load(weight + row * hidden + channel, valid & (channel < hidden), 0).to(tl.float32)


@triton.jit
def _rvq_native_mean_kernel(
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
    native_width: tl.constexpr,
):
    frame = tl.program_id(0).to(tl.int64)
    batch, token = frame // time, frame % time
    channel = (tl.program_id(1) * block_h + tl.arange(0, block_h)).to(tl.int64)
    if time * hidden > 1:
        acc0 = tl.full((block_h,), 0, tl.float32)
        acc1 = tl.full((block_h,), 0, tl.float32)
        acc2 = tl.full((block_h,), 0, tl.float32)
        acc3 = tl.full((block_h,), 0, tl.float32)
        for step in tl.static_range(4):
            acc0 = acc0 + _rvq_row(
                codes,
                weight,
                offsets,
                batch,
                token,
                channel,
                stride_b,
                stride_q,
                stride_t,
                stride_offset,
                hidden,
                entries,
                step * 4,
            )
            acc1 = acc1 + _rvq_row(
                codes,
                weight,
                offsets,
                batch,
                token,
                channel,
                stride_b,
                stride_q,
                stride_t,
                stride_offset,
                hidden,
                entries,
                step * 4 + 1,
            )
            acc2 = acc2 + _rvq_row(
                codes,
                weight,
                offsets,
                batch,
                token,
                channel,
                stride_b,
                stride_q,
                stride_t,
                stride_offset,
                hidden,
                entries,
                step * 4 + 2,
            )
            acc3 = acc3 + _rvq_row(
                codes,
                weight,
                offsets,
                batch,
                token,
                channel,
                stride_b,
                stride_q,
                stride_t,
                stride_offset,
                hidden,
                entries,
                step * 4 + 3,
            )
        mean = (((acc0 + acc1) + acc2) + acc3) * (1.0 / 16.0)
    else:
        quantizer = tl.arange(0, 16).to(tl.int64)
        code = tl.load(codes + batch * stride_b + quantizer * stride_q + token * stride_t)
        offset = tl.load(offsets + quantizer * stride_offset)
        row = (code + offset).to(tl.int64)
        valid_row = (row >= 0) & (row < entries)
        tl.device_assert(valid_row, "RVQ embedding index out of range")
        embedding = tl.load(
            weight + row[:, None] * hidden + channel[None, :], valid_row[:, None] & (channel[None, :] < hidden), 0
        ).to(tl.float32)
        mean = tl.sum(embedding, axis=0) * (1.0 / 16.0)
    tl.store(raw_output + frame * hidden + channel, mean, channel < hidden)


@triton.jit
def _native_first_rmsnorm_kernel(
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
    native_width: tl.constexpr,
):
    frame = tl.program_id(0).to(tl.int64)
    batch, token = frame // time, frame % time  # noqa: F841 - preserve validated Triton AST
    channel = tl.arange(0, block_h).to(tl.int64)
    raw_low = tl.load(raw_output + frame * hidden + channel, channel < hidden, 0)
    raw_float = raw_low.to(tl.float32)
    if hidden == 1024:
        # Load each native reduction lane directly from the rounded residual.
        # This avoids redistributing the full H vector through repeated gathers
        # while keeping Torch Reduce.cuh arithmetic and cast boundaries exact.
        lane = tl.arange(0, native_width)
        acc0 = tl.full((native_width,), 0, tl.float32)
        acc1 = tl.full((native_width,), 0, tl.float32)
        acc2 = tl.full((native_width,), 0, tl.float32)
        acc3 = tl.full((native_width,), 0, tl.float32)
        for step in tl.static_range(1024 // (native_width * 4)):
            base = lane * 4 + step * native_width * 4
            value0 = tl.load(raw_output + frame * hidden + base).to(tl.float32)
            acc0 = acc0 + value0 * value0
            value1 = tl.load(raw_output + frame * hidden + base + 1).to(tl.float32)
            acc1 = acc1 + value1 * value1
            value2 = tl.load(raw_output + frame * hidden + base + 2).to(tl.float32)
            acc2 = acc2 + value2 * value2
            value3 = tl.load(raw_output + frame * hidden + base + 3).to(tl.float32)
            acc3 = acc3 + value3 * value3
        partial = ((acc0 + acc1) + acc2) + acc3
        variance = tl.sum(partial, axis=0) * (1.0 / 1024.0)
    else:
        squares = tl.where(channel < hidden, raw_float * raw_float, 0.0)
        variance = tl.sum(squares, axis=0) / hidden
    epsilon_float = tl.full((), eps, tl.float32)
    inverse_rms = tl.rsqrt(variance + epsilon_float)
    normalized_low = (raw_float * inverse_rms).to(raw_output.dtype.element_ty)
    scale = tl.load(gamma + channel * stride_gamma, channel < hidden, 0).to(tl.float32)
    # HF casts before gamma multiplication, rather than only at the end.
    normalized = normalized_low.to(tl.float32) * scale
    output_offset = frame * hidden + channel
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
        min(
            256,
            512
            // min(
                16,
                (
                    16
                    if batch * time >= 16
                    else 8
                    if batch * time >= 8
                    else 4
                    if batch * time >= 4
                    else 2
                    if batch * time >= 2
                    else 1
                ),
            ),
        ),
    )
    # Outputs are allocated with exactly the checked weight dtype and device.
    signature = (device.index, code_dtype, weight_dtype, offset_dtype, gamma_dtype)
    return batch, time, hidden, device, weight_dtype, constants, signature


def can_use_variant(codes, weight, offsets, gamma, eps):
    return inspect_inputs(codes, weight, offsets, gamma, eps) is not None


def make_variant(num_warps):
    if num_warps not in (4, 8):
        raise ValueError("Supported launch widths are four or eight warps")

    def entry(codes, weight, offsets, gamma, eps):
        metadata = inspect_inputs(codes, weight, offsets, gamma, eps)
        if metadata is None:
            return native_region(codes, weight, offsets, gamma, eps)
        batch, time, hidden, device, dtype, constants, signature = metadata
        raw = torch.empty((batch, time, hidden), dtype=dtype, device=device)
        normalized = torch.empty_like(raw)
        if batch != 0 and time != 0:
            mean_constants = (*constants[:8], 128, *constants[9:])
            _rvq_native_mean_kernel[(batch * time, triton.cdiv(hidden, 128), 1)](
                codes,
                weight,
                offsets,
                gamma,
                raw,
                normalized,
                *mean_constants,
                num_warps=num_warps,
                debug=True,
                enable_fp_fusion=False,
            )
            _native_first_rmsnorm_kernel[(batch * time, 1, 1)](
                codes,
                weight,
                offsets,
                gamma,
                raw,
                normalized,
                *constants,
                num_warps=num_warps,
                debug=True,
                enable_fp_fusion=False,
            )
        return raw, normalized

    # Both benchmark arms use fullgraph=True and dynamic=False. Dynamo guards
    # live shape/stride/dtype/device/epsilon and Inductor owns normal dispatch.
    # No task-managed cache retains tensor pointers or streams.
    compiled = torch.compile(entry, fullgraph=True, dynamic=False)
    compiled.__name__ = f"rvq_rmsnorm_tiled_native_w{num_warps}"

    def call(codes, weight, offsets, gamma, eps):
        # Dynamo 2.13 cannot trace is_neg/is_conj booleans. Check these four
        # live tensors outside its graph; never erase or resolve a lazy view.
        if any(tensor.is_neg() or tensor.is_conj() for tensor in (codes, weight, offsets, gamma)):
            return native_region(codes, weight, offsets, gamma, eps)
        return compiled(codes, weight, offsets, gamma, eps)

    call.__name__ = compiled.__name__
    return call
