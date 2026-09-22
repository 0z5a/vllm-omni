# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.configuration_utils import FrozenDict
from torch import nn
from transformers import Mistral3Config, Mistral3Model
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.ernie_image import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def encoder():
    config = Mistral3Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=64,
        ),
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            head_dim=64,
            image_size=28,
            patch_size=14,
        ),
    )
    return Mistral3Model(config).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_encoder_requires_component_opt_in(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_ernie_text_encoder_fp8(encoder, config, torch.device("cpu")) == 0
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_encoder_rejects_unsupported_fp8_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_ernie_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": config}), torch.device("cpu")
        )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_text_projection_conversion_preserves_other_weights_and_hidden_states(encoder, monkeypatch):
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = ids.ne(0)
    original = dict(encoder.named_parameters())
    with torch.inference_mode():
        reference = encoder(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    prefixes = []

    class LoadedLinear(nn.Linear):
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert disable_tp and not return_bias
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            self.quant_method = object()
            prefixes.append(prefix)
            self.weight.weight_loader = lambda parameter, weight: parameter.copy_(weight)

    monkeypatch.setattr(quantization, "ReplicatedLinear", LoadedLinear)
    assert (
        quantization.prepare_ernie_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 14
    )
    assert all(prefix.startswith("text_encoder.language_model.layers.") for prefix in prefixes)
    for name, parameter in encoder.named_parameters():
        if not name.startswith("language_model.layers.") or name.endswith("norm.weight"):
            assert parameter is original[name]
    with torch.inference_mode():
        actual = encoder(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_encoder_rejects_float32_weights(encoder):
    encoder.float()
    with pytest.raises(ValueError, match="BF16 or FP16"):
        quantization.prepare_ernie_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )


def test_ignored_projections_keep_original_parameters(encoder, monkeypatch):
    original = dict(encoder.named_parameters())

    class IgnoredLinear(nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            self.quant_method = quantization.UnquantizedLinearMethod()

    monkeypatch.setattr(quantization, "ReplicatedLinear", IgnoredLinear)
    assert (
        quantization.prepare_ernie_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 0
    )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize("quantize_transformer", [False, True])
def test_pipeline_routes_transformer_config(encoder, monkeypatch, tmp_path, quantize_transformer):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.ernie_image import pipeline_ernie_image as pipeline

    text_config = Fp8Config()
    transformer_config = Fp8Config() if quantize_transformer else None
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        quantization_config=ComponentQuantizationConfig(
            {"text_encoder": text_config, "transformer": transformer_config}
        ),
    )
    monkeypatch.setattr(pipeline, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline.FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.AutoModel, "from_pretrained", lambda *a, **k: encoder)
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    vae = nn.Module()
    vae.config = FrozenDict(block_out_channels=[1, 2, 4, 4])
    monkeypatch.setattr(pipeline.AutoencoderKLFlux2, "from_pretrained", lambda *a, **k: vae)
    monkeypatch.setattr(pipeline, "get_transformer_config_kwargs", lambda *a: {})
    calls = []
    monkeypatch.setattr(pipeline, "prepare_ernie_text_encoder_fp8", lambda *args: calls.append(args))

    def make_transformer(*, quant_config):
        assert quant_config is transformer_config
        return nn.Module()

    monkeypatch.setattr(pipeline, "ErnieImageTransformer2DModel", make_transformer)
    result = pipeline.ErnieImagePipeline(od_config=config)
    assert result.text_encoder is encoder
    assert calls == [(encoder, config.quantization_config, torch.device("cpu"))]
