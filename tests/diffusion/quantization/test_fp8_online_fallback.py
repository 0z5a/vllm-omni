# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dispatch contracts; CUDA arithmetic is covered by test_fp8_online_quant."""

import sys
from types import ModuleType

import pytest
import torch

from vllm_omni.quantization import fp8_online

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def quantizer(monkeypatch):
    class Quantizer:
        static = False
        is_group_quant = False
        use_per_token_if_dynamic = True
        num_token_padding = None

        def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
            return "cuda", x

        def forward_native(self, x, scale=None, scale_ub=None, use_triton=False):
            return "torch", x

    module = ModuleType("vllm.model_executor.layers.quantization.input_quant_fp8")
    module.QuantFP8 = Quantizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fp8_online, "_ENABLED", False)
    monkeypatch.setattr(fp8_online, "_DISABLED", False)
    monkeypatch.setattr(fp8_online, "_FORCE_TORCH", False)
    return Quantizer


def test_default_does_not_patch(quantizer):
    original = quantizer.forward_cuda
    fp8_online.install_fp8_online_quant_patch()
    assert quantizer.forward_cuda is original


def test_opt_in_routes_only_supported_eager_cuda(quantizer, monkeypatch):
    original_native = quantizer.forward_native
    monkeypatch.setattr(fp8_online, "_ENABLED", True)
    monkeypatch.setattr(fp8_online, "supported", lambda x: True)
    monkeypatch.setattr(fp8_online, "scaled_fp8_quant", lambda x, *args, **kwargs: ("hot", x))
    fp8_online.install_fp8_online_quant_patch()
    first = quantizer.forward_cuda
    fp8_online.install_fp8_online_quant_patch()
    assert quantizer.forward_cuda is first
    assert quantizer.forward_native is original_native
    x = torch.ones(2, 4)
    assert quantizer().forward_cuda(x)[0] == "hot"
    assert quantizer().forward_native(x)[0] == "torch"
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert quantizer().forward_cuda(x)[0] == "cuda"


@pytest.mark.parametrize("case", ["unsupported", "disabled", "static", "group", "per_tensor", "triton"])
def test_original_cuda_fallback(quantizer, monkeypatch, case):
    monkeypatch.setattr(fp8_online, "_ENABLED", True)
    monkeypatch.setattr(fp8_online, "supported", lambda x: case != "unsupported")
    monkeypatch.setattr(fp8_online, "scaled_fp8_quant", lambda *args, **kwargs: pytest.fail("unexpected hot path"))
    fp8_online.install_fp8_online_quant_patch()
    q = quantizer()
    if case == "disabled":
        monkeypatch.setattr(fp8_online, "_DISABLED", True)
    if case == "static":
        q.static = True
    if case == "group":
        q.is_group_quant = True
    if case == "per_tensor":
        q.use_per_token_if_dynamic = False
    assert q.forward_cuda(torch.ones(2, 4), use_triton=case == "triton")[0] == "cuda"


def test_force_torch_takes_precedence(quantizer, monkeypatch):
    monkeypatch.setattr(fp8_online, "_ENABLED", True)
    monkeypatch.setattr(fp8_online, "_FORCE_TORCH", True)
    monkeypatch.setattr(fp8_online, "supported", lambda *args: pytest.fail("must not inspect CUDA path"))
    fp8_online.install_fp8_online_quant_patch()
    assert not fp8_online.hot_path_enabled()
    assert quantizer().forward_cuda(torch.ones(2, 4))[0] == "torch"


def test_unavailable_extension_calls_reference(monkeypatch):
    monkeypatch.setattr(fp8_online, "supported", lambda x: True)
    monkeypatch.setattr(fp8_online, "_load_extension", lambda: None)
    calls = []
    monkeypatch.setattr(fp8_online, "_reference_per_token", lambda *args: calls.append(args))
    x, out, scale = torch.ones(2, 4), torch.empty(2, 4), torch.empty(2, 1)
    fp8_online.per_token(out, x, scale)
    assert len(calls) == 1
    assert calls[0][0] is x and calls[0][1] is out and calls[0][2] is scale
