# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness tests for the FP8 online activation-quantization fast path.

The reference is the kernel vLLM-Omni calls today
(``torch.ops._C.dynamic_per_token_scaled_fp8_quant``), not a torch cast: the two
differ on exact midpoints because PTX ``cvt.rn`` rounds ties away from zero
while torch's fp8 cast rounds them to even.  The fast path must match the
*kernel*, so this module compares payload bytes and scale bits exactly.

These tests need a CUDA device with compute capability >= 8.0.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.quantization import fp8_online

vllm = pytest.importorskip("vllm")
pytest.importorskip("vllm._custom_ops")

FP8_MAX = 448.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the FP8 online fast path requires a CUDA device with sm_80+",
)

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# Shapes that exercise both mappings and their crossover.
SHAPES = [
    (1, 4096), (2, 3072), (7, 128), (8, 512), (33, 2048), (64, 512),
    (128, 4096), (256, 2048), (512, 1024), (1024, 4096), (2048, 3072),
    (5, 8), (377, 1024), (4096, 128),
]


def _reference(x: torch.Tensor, scale_ub: torch.Tensor | None = None):
    m, n = x.shape
    out = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    getattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant")(out, x, scale, scale_ub)
    return out, scale


def _fast(x: torch.Tensor, scale_ub: torch.Tensor | None = None):
    m, n = x.shape
    out = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    fp8_online.per_token(out, x, scale, scale_ub)
    return out, scale


def _fixture(dtype, m, n, kind, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "normal":
        return torch.randn(m, n, device="cuda", dtype=dtype, generator=g)
    if kind == "outlier":
        x = torch.randn(m, n, device="cuda", dtype=dtype, generator=g) * 0.01
        x[:, min(7, n - 1)] = 3000.0
        return x
    if kind == "zero":
        return torch.zeros(m, n, device="cuda", dtype=dtype)
    if kind == "tiny":
        return torch.full((m, n), 2.0**-14, device="cuda", dtype=dtype)
    if kind == "spread":
        e = torch.randint(-12, 9, (m, n), device="cuda", generator=g)
        sign = torch.where(torch.rand(m, n, device="cuda", generator=g) < 0.5, -1.0, 1.0)
        return (torch.rand(m, n, device="cuda", generator=g) * 1.5 + 0.5) * torch.pow(
            2.0, e.float()
        ) * sign
    raise ValueError(kind)


def test_compiled_extension_available():
    assert fp8_online.compiled_ok(), (
        "the CUDA extension failed to build; set "
        "VLLM_OMNI_FP8_ONLINE_VERBOSE=1 for the build log"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("kind", ["normal", "spread", "outlier", "zero", "tiny"])
def test_per_token_matches_reference_bitwise(dtype, m, n, kind):
    x = _fixture(dtype, m, n, kind, seed=m * 31 + n + len(kind))
    if not fp8_online.supported(x):
        pytest.skip("shape outside the fast path's measured envelope")
    ref_o, ref_s = _reference(x)
    got_o, got_s = _fast(x)
    assert torch.equal(ref_o.view(torch.uint8), got_o.view(torch.uint8)), (
        f"payload mismatch for {dtype} M={m} K={n} {kind}"
    )
    assert torch.equal(ref_s, got_s), f"scale mismatch for {dtype} M={m} K={n} {kind}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ub", [0.01, 0.5, 4.0, 100.0])
def test_per_token_scale_ub_matches_reference(dtype, ub):
    x = _fixture(dtype, 128, 4096, "spread", seed=int(ub * 1000))
    scale_ub = torch.tensor([ub], device="cuda", dtype=torch.float32)
    ref_o, ref_s = _reference(x, scale_ub)
    got_o, got_s = _fast(x, scale_ub)
    assert torch.equal(ref_o.view(torch.uint8), got_o.view(torch.uint8))
    assert torch.equal(ref_s, got_s)


def test_tie_breaking_matches_ptx_not_torch_cast():
    """Exact midpoints are where torch's cast and the kernel disagree."""
    ties = []
    for e in range(-6, 8):
        base = 2.0**e
        for k in (1, 3, 5, 7, 9, 11, 13, 15):
            ties.append(base * (1.0 + k / 16.0))
    x = torch.tensor(ties, device="cuda", dtype=torch.float32).reshape(1, -1)
    # amax == 448 makes the scale exactly 1.0, so the ties reach the converter
    # unscaled and the rounding rule is isolated from the scale arithmetic.
    x = torch.cat([x, torch.full((1, 4), FP8_MAX, device="cuda")], dim=1)
    ref_o, ref_s = _reference(x)
    got_o, got_s = _fast(x)
    assert torch.equal(ref_o.view(torch.uint8), got_o.view(torch.uint8))
    assert torch.equal(ref_s, got_s)


@pytest.mark.parametrize(
    "tensor_kwargs",
    [
        {"shape": (16, 4096), "dtype": torch.bfloat16, "noncontig": True},
        {"shape": (16, 4095), "dtype": torch.bfloat16, "noncontig": False},
        {"shape": (16, 4096), "dtype": torch.float64, "noncontig": False},
    ],
)
def test_unsupported_inputs_are_declined(tensor_kwargs):
    shape = tensor_kwargs["shape"]
    dtype = tensor_kwargs["dtype"]
    x = torch.randn(*shape, device="cuda", dtype=dtype if dtype != torch.float64 else torch.float32)
    x = x.to(dtype)
    if tensor_kwargs["noncontig"]:
        x = x[:, ::2]
    assert not fp8_online.supported(x)


def test_large_row_count_is_declined():
    """Above the measured crossover the reference kernel is faster."""
    x = torch.randn(
        fp8_online.FASTPATH_MAX_ROWS + 1, 4096, device="cuda", dtype=torch.bfloat16
    )
    assert not fp8_online.supported(x)


def test_scaled_fp8_quant_falls_back_for_static_scale():
    x = _fixture(torch.bfloat16, 64, 512, "normal")
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    out, got_s = fp8_online.scaled_fp8_quant(x, scale)
    from vllm import _custom_ops as ops

    ref_out, ref_s = ops.scaled_fp8_quant(x, scale)
    assert torch.equal(ref_out.view(torch.uint8), out.view(torch.uint8))
    assert torch.equal(ref_s, got_s)


def test_per_tensor_matches_reference():
    x = _fixture(torch.bfloat16, 256, 4096, "spread")
    ref = torch.empty(1, device="cuda", dtype=torch.float32)
    getattr(torch.ops._C, "dynamic_scaled_fp8_quant")(
        torch.empty_like(x, dtype=torch.float8_e4m3fn), x, ref
    )
    ref_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    getattr(torch.ops._C, "dynamic_scaled_fp8_quant")(ref_out, x, ref)
    got_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    got = torch.empty(1, device="cuda", dtype=torch.float32)
    fp8_online.per_tensor(got_out, x, got)
    assert torch.equal(ref_out.view(torch.uint8), got_out.view(torch.uint8))
    assert torch.equal(ref, got)
