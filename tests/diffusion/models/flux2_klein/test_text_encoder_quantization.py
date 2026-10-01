# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.flux2_klein import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def encoder():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=1024,
    )
    return Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_klein_requires_explicit_component(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_flux2_klein_text_encoder_fp8(encoder, config, torch.device("cpu")) == 0
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_klein_rejects_unsupported_config_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_flux2_klein_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": config}), torch.device("cpu")
        )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_klein_conversion_preserves_embedding_lm_head_norms_and_hidden_states(encoder, monkeypatch):
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = ids.ne(0)
    original = dict(encoder.named_parameters())
    with torch.inference_mode():
        expected = encoder.model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    prefixes = []

    class LoadedLinear(nn.Linear):
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert disable_tp and not return_bias
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            self.quant_method = object()
            self.weight.weight_loader = lambda parameter, weight: parameter.copy_(weight)
            prefixes.append(prefix)

    monkeypatch.setattr(quantization, "ReplicatedLinear", LoadedLinear)
    assert (
        quantization.prepare_flux2_klein_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 14
    )
    assert all(prefix.startswith("text_encoder.model.layers.") for prefix in prefixes)
    for name, parameter in encoder.named_parameters():
        if not name.startswith("model.layers.") or name.endswith("norm.weight"):
            assert parameter is original[name]
    with torch.inference_mode():
        actual = encoder.model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_klein_ignored_layers_retain_original_parameters(encoder, monkeypatch):
    original = dict(encoder.named_parameters())

    class IgnoredLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.quant_method = quantization.UnquantizedLinearMethod()

    monkeypatch.setattr(quantization, "ReplicatedLinear", IgnoredLinear)
    assert (
        quantization.prepare_flux2_klein_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 0
    )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_klein_rejects_float32_weights(encoder):
    with pytest.raises(ValueError, match="BF16 or FP16"):
        quantization.prepare_flux2_klein_text_encoder_fp8(
            encoder.float(), ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
