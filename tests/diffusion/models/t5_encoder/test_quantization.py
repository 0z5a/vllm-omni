# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.t5_encoder import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(params=[(T5Config, T5EncoderModel), (UMT5Config, UMT5EncoderModel)])
def encoder(request):
    config_type, model_type = request.param
    config = config_type(
        vocab_size=32,
        d_model=128,
        d_ff=256,
        d_kv=64,
        num_heads=2,
        num_layers=2,
        feed_forward_proj="gated-gelu",
        dropout_rate=0,
    )
    return model_type(config).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_t5_requires_explicit_component(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_t5_fp8(encoder, config, "text_encoder_2") == 0
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_t5_rejects_unsupported_config_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_t5_fp8(encoder, ComponentQuantizationConfig({"text_encoder_2": config}), "text_encoder_2")
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_t5_conversion_preserves_wo_norms_embeddings_and_forward(encoder, monkeypatch):
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = ids.ne(0)
    original = dict(encoder.named_parameters())
    with torch.inference_mode():
        expected = encoder(ids, attention_mask=mask).last_hidden_state
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

    monkeypatch.setattr(quantization, "_T5Fp8Linear", LoadedLinear)
    assert (
        quantization.prepare_t5_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder_2": Fp8Config()}), "text_encoder_2"
        )
        == 12
    )
    assert all(prefix.startswith("text_encoder_2.encoder.block.") for prefix in prefixes)
    assert not any(prefix.endswith(".wo") for prefix in prefixes)
    for name, parameter in encoder.named_parameters():
        if name not in {prefix.removeprefix("text_encoder_2.") + ".weight" for prefix in prefixes}:
            assert parameter is original[name]
    with torch.inference_mode():
        torch.testing.assert_close(encoder(ids, attention_mask=mask).last_hidden_state, expected, atol=0, rtol=0)


def test_t5_ignored_layers_retain_original_parameters(encoder, monkeypatch):
    original = dict(encoder.named_parameters())

    class IgnoredLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.quant_method = quantization.UnquantizedLinearMethod()

    monkeypatch.setattr(quantization, "_T5Fp8Linear", IgnoredLinear)
    assert (
        quantization.prepare_t5_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder_2": Fp8Config()}), "text_encoder_2"
        )
        == 0
    )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())
