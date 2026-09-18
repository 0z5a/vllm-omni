# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)

from vllm_omni.quantization import svdquant_config
from vllm_omni.quantization.factory import (
    SUPPORTED_QUANTIZATION_METHODS,
    build_quant_config,
)
from vllm_omni.quantization.svdquant_config import (
    DiffusionSVDQuantConfig,
    DiffusionSVDQuantLinearMethod,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_config_from_checkpoint_dict() -> None:
    config = DiffusionSVDQuantConfig.from_config(
        {
            "rank": 32,
            "precision": "nvfp4",
            "act_unsigned": False,
            "modules_to_not_convert": ["blocks.0.adaln_proj.linear"],
        }
    )

    assert config.rank == 32
    assert config.precision == "nvfp4"
    assert config.modules_to_not_convert == ["blocks.0.adaln_proj.linear"]
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0}, "rank must be positive"),
        ({"precision": "int4"}, "NVFP4 checkpoints only"),
        ({"act_unsigned": True}, "unsigned activations"),
    ],
)
def test_config_rejects_unsupported_phase1_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DiffusionSVDQuantConfig(**kwargs)


def test_factory_builds_svdquant_with_extra_checkpoint_fields() -> None:
    config = build_quant_config(
        {
            "quant_method": "svdquant",
            "rank": 16,
            "precision": "nvfp4",
            "w4a16_modules": ["adaln_proj.linear"],
            "w4a16_group_size": 64,
        }
    )

    assert isinstance(config, DiffusionSVDQuantConfig)
    assert config.rank == 16
    assert "svdquant" in SUPPORTED_QUANTIZATION_METHODS


def test_skipped_and_non_linear_modules_remain_unquantized() -> None:
    config = DiffusionSVDQuantConfig(modules_to_not_convert=["blocks.0.adaln_proj.linear"])
    linear = Mock(spec=LinearBase)

    assert isinstance(
        config.get_quant_method(linear, "blocks.0.adaln_proj.linear"),
        UnquantizedLinearMethod,
    )
    assert config.get_quant_method(torch.nn.ReLU(), "blocks.0.act") is None


def test_linear_method_creates_canonical_partitioned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    method = DiffusionSVDQuantLinearMethod(DiffusionSVDQuantConfig(rank=32))
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[64, 64],
        input_size=128,
        output_size=128,
        params_dtype=torch.bfloat16,
    )

    assert layer.qweight.shape == (128, 64)
    assert layer.qweight.dtype == torch.int8
    assert layer.wscales.shape == (8, 128)
    assert layer.wscales.dtype == torch.float8_e4m3fn
    assert layer.proj_down.shape == (128, 32)
    assert layer.proj_up.shape == (128, 32)
    assert layer.smooth_factor.shape == (128,)
    assert layer.wcscales.shape == (128,)
    assert layer.wtscale.shape == (1,)

    assert layer.qweight.input_dim == 1
    assert layer.qweight.output_dim == 0
    assert layer.wscales.input_dim == 0
    assert layer.wscales.output_dim == 1
    assert layer.proj_down.input_dim == 0
    assert layer.proj_up.output_dim == 0
    assert layer.wcscales.output_dim == 0


def test_linear_method_rejects_misaligned_input_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    method = DiffusionSVDQuantLinearMethod(DiffusionSVDQuantConfig())

    with pytest.raises(ValueError, match="divisible by the block size 16"):
        method.create_weights(
            torch.nn.Module(),
            input_size_per_partition=24,
            output_partition_sizes=[32],
            input_size=24,
            output_size=32,
            params_dtype=torch.bfloat16,
        )


def test_active_linear_uses_svdquant_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    config = DiffusionSVDQuantConfig()
    linear = Mock(spec=LinearBase)

    method = config.get_quant_method(linear, "blocks.0.attn.out_proj")

    assert isinstance(method, DiffusionSVDQuantLinearMethod)

def test_native_options_default_to_previous_behaviour() -> None:
    config = DiffusionSVDQuantConfig(activation_bits=4, linear_backend="flashinfer")

    # Unset means "keep the library/architecture default", so an existing
    # checkpoint behaves exactly as before these options existed.
    assert config.native_backend is None
    assert config.native_enable_pdl is None
    # The measured-slow geometry keeps the compatibility implementation by
    # default; an empty list in the checkpoint config disables the fallback.
    assert config.native_falls_back(14336, 5376)
    assert not config.native_falls_back(5376, 21504)
    assert not DiffusionSVDQuantConfig(
        activation_bits=4, linear_backend="flashinfer", native_fallback_shapes=[]
    ).native_falls_back(14336, 5376)


def test_native_options_round_trip_through_the_serialized_config() -> None:
    config = DiffusionSVDQuantConfig.from_config(
        {
            "quant_method": "svdquant",
            "rank": 32,
            "precision": "nvfp4",
            "activation_bits": 4,
            "linear_backend": "flashinfer",
            "native_backend": "cute-dsl",
            "native_enable_pdl": True,
            "native_fallback_shapes": [[14336, 5376]],
        }
    )

    assert config.native_backend == "cute-dsl"
    assert config.native_enable_pdl is True
    assert config.native_falls_back(14336, 5376)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"native_backend": "cute-dsl"}, "require linear_backend"),
        ({"native_enable_pdl": True}, "require linear_backend"),
        ({"native_fallback_shapes": [[14336, 5376]]}, "requires linear_backend"),
        ({"native_backend": "triton"}, "native_backend must be one of"),
        ({"native_enable_pdl": 1}, "must be a bool or None"),
        ({"native_fallback_shapes": [[14336]]}, "must be \\[input_size, output_size\\]"),
        ({"native_fallback_shapes": [[14336, -1]]}, "must be \\[input_size, output_size\\]"),
    ],
)
def test_native_options_reject_invalid_combinations(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DiffusionSVDQuantConfig(activation_bits=4, linear_backend="flashinfer", **kwargs)


def test_native_routing_falls_back_for_the_measured_slow_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    calls: list[str] = []

    def fake_compatibility(self, layer: torch.nn.Module) -> None:
        calls.append("compatibility")

    def fake_native(self, layer: torch.nn.Module) -> None:
        calls.append("native")

    monkeypatch.setattr(DiffusionSVDQuantLinearMethod, "_prepare_compatibility_weights", fake_compatibility)
    monkeypatch.setattr(DiffusionSVDQuantLinearMethod, "_prepare_native_weights", fake_native)

    def routed_path(input_size: int, output_size: int, **kwargs: object) -> str:
        method = DiffusionSVDQuantLinearMethod(
            DiffusionSVDQuantConfig(activation_bits=4, linear_backend="flashinfer", **kwargs)
        )
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            torch.zeros(output_size, input_size // 2, dtype=torch.int8), requires_grad=False
        )
        calls.clear()
        method.process_weights_after_loading(layer)
        return calls[-1]

    assert routed_path(14336, 5376) == "compatibility"
    assert routed_path(5376, 21504) == "native"
    assert routed_path(14336, 5376, native_fallback_shapes=[]) == "native"
