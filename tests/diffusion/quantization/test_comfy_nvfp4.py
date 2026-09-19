# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

from vllm_omni.diffusion.models.flux2.nvfp4_checkpoint import (
    map_bfl_name,
    map_bfl_weight,
    quantized_layer_names,
)
from vllm_omni.quantization.comfy_nvfp4_config import ComfyNvfp4Config, ComfyNvfp4LinearMethod
from vllm_omni.quantization.factory import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("double_blocks.0.img_attn.qkv.weight_scale", "transformer_blocks.0.attn.to_qkv.weight_scale"),
        ("double_blocks.7.txt_attn.norm.query_norm.scale", "transformer_blocks.7.attn.norm_added_q.weight"),
        ("single_blocks.47.linear1.input_scale", "single_transformer_blocks.47.attn.to_qkv_mlp_proj.input_scale"),
        ("single_blocks.2.linear2.weight", "single_transformer_blocks.2.attn.to_out.weight"),
        ("img_in.weight", "x_embedder.weight"),
    ],
)
def test_checkpoint_name_mapping(source, target):
    assert map_bfl_name(source) == target


def test_final_modulation_swaps_scale_and_shift():
    source = torch.arange(24).reshape(6, 4)
    name, mapped = map_bfl_weight("final_layer.adaLN_modulation.1.weight", source)
    assert name == "norm_out.linear.weight"
    torch.testing.assert_close(mapped, torch.cat((source[3:], source[:3])))
    name, preserved = map_bfl_weight("double_blocks.0.img_attn.qkv.weight", source)
    assert preserved is source


def test_metadata_keeps_unquantized_text_attention():
    metadata = {"layers": {"double_blocks.0.img_attn.qkv": {"format": "nvfp4"}}}
    config = ComfyNvfp4Config(list(quantized_layer_names(metadata)))
    layer = object.__new__(ReplicatedLinear)
    nn.Module.__init__(layer)
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.0.attn.to_qkv"), ComfyNvfp4LinearMethod)
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.0.attn.add_kv_proj"), UnquantizedLinearMethod)
    assert isinstance(
        config.get_quant_method(layer, "transformer.transformer_blocks.0.attn.to_qkv"), ComfyNvfp4LinearMethod
    )


def test_non_nvfp4_metadata_is_rejected():
    with pytest.raises(ValueError, match="uniform NVFP4"):
        list(quantized_layer_names({"layers": {"img_in": {"format": "float8"}}}))


def test_serialized_layout_loads_packed_weights_and_scales():
    layer = nn.Module()
    method = ComfyNvfp4LinearMethod()
    method.create_weights(layer, 32, [16, 16, 16], 32, 48, torch.bfloat16)
    packed = torch.arange(48 * 16).reshape(48, 16).to(torch.uint8)
    scales = torch.ones(48, 2, dtype=torch.float8_e4m3fn)
    for parameter, value in (
        (layer.weight, packed),
        (layer.weight_scale, scales),
        (layer.weight_scale_2, torch.tensor(0.125)),
        (layer.input_scale, torch.tensor(0.5)),
    ):
        parameter.weight_loader(parameter, value)
    assert torch.equal(layer.weight, packed)
    assert torch.equal(layer.weight_scale.view(torch.uint8), scales.view(torch.uint8))
    assert layer.weight_scale_2.item() == 0.125
    assert layer.input_scale.item() == 0.5


def test_tensor_parallel_is_rejected_before_allocating():
    layer = nn.Module()
    with pytest.raises(ValueError, match="parallel size 1"):
        ComfyNvfp4LinearMethod().create_weights(layer, 16, [16], 32, 32, torch.bfloat16)
    assert not list(layer.parameters())


def test_factory_routes_explicit_checkpoint_layers():
    config = build_quant_config({"method": "comfy_nvfp4", "quantized_layers": ["transformer_blocks.0.attn.to_qkv"]})
    assert isinstance(config, ComfyNvfp4Config)
