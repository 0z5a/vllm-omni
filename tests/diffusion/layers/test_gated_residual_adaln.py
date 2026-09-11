# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.gated_residual_adaln import try_fused_gated_residual_adaln

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _inputs(b, s, d, dtype, device):
    generator = torch.Generator(device=device).manual_seed(7382)
    residual = torch.randn(b, s, d, dtype=dtype, device=device, generator=generator)
    # Attention output and modulation both have realistic slice strides.
    branch = torch.randn(b, s, 2 * d, dtype=dtype, device=device, generator=generator)[..., :d]
    modulation = torch.randn(b, 6 * d, dtype=dtype, device=device, generator=generator)
    shift, scale, gate, _, _, _ = [t.unsqueeze(1) for t in modulation.chunk(6, -1)]
    return residual, branch, gate, scale, shift


def _reference(residual, branch, gate, scale, shift, eps=1e-6):
    r = residual + gate * branch
    return r, AdaLayerNorm(r.shape[-1], eps=eps)(r, scale, shift)


def _checked_call(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], eps=1e-6):
    before = [x.clone() for x in inputs]
    expected = _reference(*inputs, eps=eps)
    result = try_fused_gated_residual_adaln(*inputs, eps)
    assert result is not None
    r, y = result
    # Residual arithmetic has no reduction and preserves every rounding step.
    torch.testing.assert_close(r, expected[0], rtol=0, atol=0, equal_nan=True)
    # BF16 allows one ULP near 1; the FP32 reduction uses a parallel sum.
    atol, rtol = (0.03125, 0.016) if y.dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(y, expected[1], atol=atol, rtol=rtol, equal_nan=True)
    assert torch.equal(torch.isnan(y), torch.isnan(expected[1]))
    assert torch.equal(torch.isinf(y), torch.isinf(expected[1]))
    assert y.shape == r.shape == inputs[0].shape
    assert r.is_contiguous() and y.is_contiguous()
    assert r.data_ptr() != y.data_ptr()
    for x, original in zip(inputs, before):
        torch.testing.assert_close(x, original, atol=0, rtol=0, equal_nan=True)
        assert r.data_ptr() != x.data_ptr() and y.data_ptr() != x.data_ptr()
    return result


@pytest.mark.cpu
def test_cpu_declines_without_mutation():
    inputs = _inputs(2, 3, 128, torch.float32, "cpu")
    before = [x.clone() for x in inputs]
    assert try_fused_gated_residual_adaln(*inputs, 1e-6) is None
    for x, original in zip(inputs, before):
        assert torch.equal(x, original)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("b,s,d", [(1, 1, 3072), (2, 3, 3072), (2, 129, 3072), (2, 5, 257), (1, 7, 8192)])
def test_two_outputs_and_strided_modulation(dtype, b, s, d):
    _checked_call(_inputs(b, s, d, dtype, "cuda"))


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["zero_gate", "minus_one_scale", "constant", "tiny_variance", "small", "large", "nan", "inf"]
)
def test_numerical_boundaries(dtype, case):
    inputs = _inputs(2, 3, 3072, dtype, "cuda")
    residual, branch, gate, scale, shift = inputs
    if case == "zero_gate":
        gate.zero_()
    elif case == "minus_one_scale":
        scale.fill_(-1)
    elif case in ("constant", "tiny_variance"):
        residual.fill_(1)
        branch.zero_()
        if case == "tiny_variance":
            residual[..., 0] += torch.finfo(dtype).eps
    elif case == "small":
        residual.mul_(1e-15)
        branch.mul_(1e-15)
    elif case == "large":
        residual.mul_(1e8)
        branch.mul_(1e8)
    elif case == "nan":
        residual[0, 0, 0] = float("nan")
    elif case == "inf":
        residual[0, 0, 0] = float("inf")
    _checked_call(inputs)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fallback_and_compile(dtype):
    inputs = _inputs(2, 3, 3072, dtype, "cuda")
    r, branch, gate, scale, shift = inputs
    assert try_fused_gated_residual_adaln(r.requires_grad_(), branch, gate, scale, shift, 1e-6) is None
    r.requires_grad_(False)
    assert try_fused_gated_residual_adaln(r, branch, gate.double(), scale, shift, 1e-6) is None
    assert (
        try_fused_gated_residual_adaln(
            r[..., ::2], branch[..., ::2], gate[..., ::2], scale[..., ::2], shift[..., ::2], 1e-6
        )
        is None
    )
    assert try_fused_gated_residual_adaln(r, branch, gate.expand_as(r), scale, shift, 1e-6) is None

    def caller(r, branch, gate, scale, shift):
        fused = try_fused_gated_residual_adaln(r, branch, gate, scale, shift, 1e-6)
        if fused is not None:
            return fused
        residual = r + gate * branch
        normed = torch.nn.functional.layer_norm(residual.float(), (r.shape[-1],), eps=1e-6).to(r.dtype)
        return residual, normed * (1 + scale) + shift

    expected = torch.compile(caller, fullgraph=True)(*inputs)
    # fullgraph compilation includes the functional fused Triton call.
    reference = _reference(*inputs)
    torch.testing.assert_close(expected[0], reference[0], rtol=0, atol=0)
    atol, rtol = (0.03125, 0.016) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(expected[1], reference[1], rtol=rtol, atol=atol)
    torch.testing.assert_close(expected, caller(*inputs), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"})
def test_cuda_graph_replays_new_inputs():
    inputs = _inputs(2, 5, 3072, torch.bfloat16, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _checked_call(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = try_fused_gated_residual_adaln(*inputs, 1e-6)
    inputs[0].add_(0.5)
    graph.replay()
    assert result is not None
    expected = _reference(*inputs)
    torch.testing.assert_close(result[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected[1], rtol=0.016, atol=0.03125)
