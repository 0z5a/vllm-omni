# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import T5Config, T5EncoderModel
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.flux import pipeline_flux_kontext as kontext
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_kontext_rejects_unsupported_encoder_fp8_before_mutation(monkeypatch, config):
    encoder = T5EncoderModel(T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_heads=2))
    original = dict(encoder.named_parameters())
    monkeypatch.setattr(kontext, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(kontext.FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(kontext.CLIPTextModel, "from_pretrained", lambda *a, **k: nn.Linear(2, 2))
    monkeypatch.setattr(kontext.T5EncoderModel, "from_pretrained", lambda *a, **k: encoder)
    od_config = SimpleNamespace(
        model="unused-model-path",
        quantization_config=ComponentQuantizationConfig({"text_encoder_2": config}),
    )
    with pytest.raises(ValueError, match="text_encoder_2 supports dynamic online FP8"):
        kontext.FluxKontextPipeline(od_config=od_config)
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in original.items())


def _stub_vae() -> nn.Module:
    vae = nn.Identity()
    vae.config = SimpleNamespace(block_out_channels=(4, 8, 8, 8), latent_channels=16)
    return vae


@pytest.mark.parametrize("component_scope", [True, False])
def test_kontext_narrows_transformer_quant_config(monkeypatch, component_scope):
    """A component config must not reach the DiT: vLLM rejects a config with no quant method."""
    fp8 = Fp8Config()
    quantization_config = ComponentQuantizationConfig({"text_encoder_2": fp8}) if component_scope else fp8
    expected = None if component_scope else fp8
    encoder = T5EncoderModel(T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_heads=2))
    tokenizer = SimpleNamespace(model_max_length=77)
    captured: dict[str, object] = {}

    monkeypatch.setattr(kontext, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(kontext.FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(kontext.CLIPTextModel, "from_pretrained", lambda *a, **k: nn.Linear(2, 2))
    monkeypatch.setattr(kontext.T5EncoderModel, "from_pretrained", lambda *a, **k: encoder)
    monkeypatch.setattr(kontext.CLIPTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(kontext.T5TokenizerFast, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(kontext.AutoencoderKL, "from_pretrained", lambda *a, **k: _stub_vae())
    monkeypatch.setattr(kontext, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(kontext, "prepare_t5_fp8", lambda *a, **k: 0)
    monkeypatch.setattr(
        kontext,
        "FluxKontextTransformer2DModel",
        lambda **kwargs: captured.update(kwargs) or nn.Identity(),
    )

    kontext.FluxKontextPipeline(
        od_config=SimpleNamespace(
            model="unused-model-path",
            quantization_config=quantization_config,
            tf_model_config=SimpleNamespace(),
            enable_diffusion_pipeline_profiler=False,
        )
    )

    assert captured["quant_config"] is expected
