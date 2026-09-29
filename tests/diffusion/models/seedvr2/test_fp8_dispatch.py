# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dispatch selection without a CUDA device or model weights."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2 import quantization as seed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("mode", ["default", "hot", "disabled", "torch"])
def test_seedvr2_activation_dispatch_preserves_weight_route(monkeypatch, mode):
    calls = []

    def reference(x, **kwargs):
        calls.append("reference")
        return x, torch.ones(x.shape[0], 1)

    def hot(x, **kwargs):
        calls.append("hot")
        return x, torch.ones(x.shape[0], 1)

    class Native:
        def __init__(self, **kwargs):
            pass

        def forward_native(self, x):
            calls.append("torch")
            return x, torch.ones(x.shape[0], 1)

    module = ModuleType("vllm.model_executor.layers.quantization.input_quant_fp8")
    module.QuantFP8 = Native
    monkeypatch.setitem(sys.modules, module.__name__, module)
    utils = ModuleType("vllm.model_executor.layers.quantization.utils.quant_utils")
    utils.GroupShape = SimpleNamespace(PER_TOKEN="per-token")
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(seed.fp8_online, "_ENABLED", mode != "default")
    monkeypatch.setattr(seed.fp8_online, "_DISABLED", mode == "disabled")
    monkeypatch.setattr(seed.fp8_online, "_FORCE_TORCH", mode == "torch")
    monkeypatch.setattr(seed.fp8_online, "scaled_fp8_quant", hot)
    monkeypatch.setattr(seed.ops, "scaled_fp8_quant", reference)
    monkeypatch.setattr(seed.ops, "cutlass_scaled_mm", lambda x, *args: x)
    layer = seed.SeedVR2W8A8Linear(torch.nn.Linear(4, 4), "fp8")
    assert calls == ["reference"]
    layer(torch.ones(2, 4))
    assert calls == ["reference", {"hot": "hot", "torch": "torch"}.get(mode, "reference")]
