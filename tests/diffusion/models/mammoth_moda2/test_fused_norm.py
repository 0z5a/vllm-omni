# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from tests.diffusion.models.mammoth_moda2.test_pipeline_sp import _config, _pipeline, _request
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.mammoth_moda2.fused_norm import rms_norm_gated_residual, rms_norm_modulation

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_native_diffusion_extras_default_is_an_independent_dict():
    first, second = _config(), _config()
    assert first.extras == second.extras == {}
    first.extras["mammoth_fused_norm"] = True
    assert second.extras == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence", [0, 1, 17])
@pytest.mark.parametrize("hidden", [126, 2520])
def test_cpu_fallback_preserves_native_dtype_and_rounding(dtype, weight_dtype, sequence, hidden):
    torch.manual_seed(42)
    norm = Qwen2RMSNorm(hidden, eps=1e-5).to(dtype=weight_dtype)
    x = torch.randn(2, sequence, hidden, dtype=dtype)
    residual = torch.randn_like(x)
    # The real adaptive norm uses chunk(), whose batch stride is 4*hidden.
    scale, gate, _, _ = torch.randn(2, 4 * hidden, dtype=dtype).chunk(4, dim=1)
    before = tuple(t.clone() for t in (x, residual, scale, gate, norm.weight))
    with torch.no_grad():
        expected_scale = norm(x) * (1 + scale[:, None])
        expected_residual = residual + gate[:, None].tanh() * norm(x)
        for enabled in (False, True):
            torch.testing.assert_close(
                rms_norm_modulation(norm, x, scale, enabled=enabled), expected_scale, rtol=0, atol=0
            )
            torch.testing.assert_close(
                rms_norm_gated_residual(norm, x, gate, residual, enabled=enabled), expected_residual, rtol=0, atol=0
            )
    for original, after in zip(before, (x, residual, scale, gate, norm.weight), strict=True):
        torch.testing.assert_close(original, after, rtol=0, atol=0)


def test_fallback_preserves_autograd_and_norm_hooks():
    norm = Qwen2RMSNorm(16, eps=1e-5)
    x = torch.randn(1, 3, 16, requires_grad=True)
    scale = torch.randn(1, 16, requires_grad=True)
    calls = []
    handle = norm.register_forward_hook(lambda module, args, output: calls.append(output))
    try:
        rms_norm_modulation(norm, x, scale, enabled=True).sum().backward()
    finally:
        handle.remove()
    assert len(calls) == 1
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, scale, norm.weight))


def test_fusion_flag_changes_no_checkpoint_keys_or_cpu_pipeline_outputs(monkeypatch):
    config = _config()
    baseline = _pipeline(config, monkeypatch)
    enabled = _pipeline(replace(config, extras={"mammoth_fused_norm": True}), monkeypatch)
    assert baseline.state_dict().keys() == enabled.state_dict().keys()
    enabled.load_state_dict(baseline.state_dict(), strict=True)
    assert all(block.fused_norm and block.norm1.fused_norm for block in enabled.gen_transformer.layers)
    assert all(not block.fused_norm for block in enabled.gen_transformer.noise_refiner)
    with set_forward_context(omni_diffusion_config=config):
        torch.testing.assert_close(enabled(_request(3, 4.0)).output, baseline(_request(3, 4.0)).output, rtol=0, atol=0)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_fusion_flag_requires_boolean(monkeypatch, value):
    with pytest.raises(ValueError, match="mammoth_fused_norm must be a bool"):
        _pipeline(replace(_config(), extras={"mammoth_fused_norm": value}), monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [("enforce_eager", False), ("cache_backend", "cache_dit"), ("lora_path", "/unqualified/lora")],
)
def test_fusion_rejects_unqualified_feature_combinations(monkeypatch, field, value):
    config = replace(_config(), extras={"mammoth_fused_norm": True}, **{field: value})
    with pytest.raises(ValueError, match="requires eager execution"):
        _pipeline(config, monkeypatch)


def test_fusion_rejects_unqualified_dev_variant(monkeypatch):
    config = replace(_config(), extras={"mammoth_fused_norm": True})
    config.tf_model_config.params["llm_config"]["model_type"] = "mammothmoda2_qwen3_vl"
    with pytest.raises(ValueError, match="limited to Preview"):
        _pipeline(config, monkeypatch)
