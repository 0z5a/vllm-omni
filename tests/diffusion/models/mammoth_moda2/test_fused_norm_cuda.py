# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA arithmetic checks; no checkpoint or E2E speedup assertion."""

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from vllm_omni.diffusion.models.mammoth_moda2.fused_norm import rms_norm_gated_residual, rms_norm_modulation

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [126, 2520])
@pytest.mark.parametrize("shape", [(1, 1), (2, 17), (1, 1025)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_fused_norm_cuda_matches_native_and_runs_kernel(monkeypatch, dtype, hidden, shape, strided, eps):
    from vllm_omni.diffusion.models.mammoth_moda2 import fused_norm_triton

    torch.manual_seed(42)
    batch, sequence = shape
    norm = Qwen2RMSNorm(hidden, eps=eps).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    x = torch.randn(batch, sequence * (2 if strided else 1), hidden, device="cuda", dtype=dtype)
    if strided:
        x = x[:, ::2]
    residual = torch.randn_like(x)
    scale, gate, _, _ = torch.randn(batch, 4 * hidden, device="cuda", dtype=dtype).chunk(4, dim=1)
    originals = tuple(t.clone() for t in (x, residual, scale, gate, norm.weight))
    real_launch = fused_norm_triton.launch_fused_norm
    calls = []

    def observed_launch(*args, **kwargs):
        calls.append(kwargs.get("residual") is not None)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(fused_norm_triton, "launch_fused_norm", observed_launch)
    with torch.no_grad():
        pairs = (
            (rms_norm_modulation(norm, x, scale, enabled=True), norm(x) * (1 + scale[:, None])),
            (
                rms_norm_gated_residual(norm, x, gate, residual, enabled=True),
                residual + gate[:, None].tanh() * norm(x),
            ),
        )
        torch.accelerator.synchronize()
    assert calls == [False, True], "a silent fallback must not pass a CUDA kernel test"
    for actual, expected in pairs:
        assert actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        if dtype is torch.float32:
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        else:
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            relative_rms = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
            assert relative_rms.item() < 0.003
    for original, after in zip(originals, (x, residual, scale, gate, norm.weight), strict=True):
        torch.testing.assert_close(original, after, rtol=0, atol=0)


def test_fused_norm_cuda_keeps_mixed_dtype_and_autograd_on_native_path(monkeypatch):
    from vllm_omni.diffusion.models.mammoth_moda2 import fused_norm_triton

    def forbidden_launch(*args, **kwargs):
        pytest.fail("unsupported dtype/autograd must retain native arithmetic")

    monkeypatch.setattr(fused_norm_triton, "launch_fused_norm", forbidden_launch)
    norm = Qwen2RMSNorm(2520, eps=1e-5).cuda()
    x = torch.randn(1, 3, 2520, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 2520, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(rms_norm_modulation(norm, x, scale, enabled=True), norm(x) * (1 + scale[:, None]))
    norm = norm.to(dtype=torch.bfloat16)
    rms_norm_modulation(norm, x, scale, enabled=True).sum().backward()
    assert norm.weight.grad is not None
