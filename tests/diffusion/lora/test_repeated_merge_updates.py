# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Repeated merge-on-load switches on the #5059 foundation."""

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights

from tests.diffusion.lora.helpers import FakeLinearBase
from tests.diffusion.lora.test_merge_on_load import (
    IN_DIM,
    _make_lora,
    _make_manager,
    _MergeableLoRALayer,
    _register,
    _StubLoRAModel,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _expected(base: torch.Tensor, lora: LoRALayerWeights) -> torch.Tensor:
    return base + (lora.lora_b.float() @ lora.lora_a.float()).to(base.dtype)


def _two_layers() -> tuple[DiffusionLoRAManager, _MergeableLoRALayer, _MergeableLoRALayer]:
    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = FakeLinearBase()
    pipeline.transformer.bar = FakeLinearBase()
    manager = DiffusionLoRAManager(
        pipeline, torch.device("cpu"), torch.float32, max_cached_adapters=2, merge_on_load=True
    )
    foo = _MergeableLoRALayer((6,))
    bar = _MergeableLoRALayer((5,))
    manager._lora_modules = {"transformer.foo": foo, "transformer.bar": bar}
    return manager, foo, bar


def _named_lora(name: str, out_dim: int, seed: int) -> LoRALayerWeights:
    gen = torch.Generator().manual_seed(seed)
    return LoRALayerWeights(
        module_name=name,
        rank=2,
        lora_alpha=2,
        lora_a=torch.randn(2, IN_DIM, generator=gen),
        lora_b=torch.randn(out_dim, 2, generator=gen),
    )


def test_shared_target_is_written_once_per_switch():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(2, 6, 10)
    second = _make_lora(2, 6, 11)
    _register(manager, 1, first)
    _register(manager, 2, second)
    base = layer.base_layer.weight.detach().clone()
    address = layer.base_layer.weight.data_ptr()

    manager._activate_adapter(1, 1.0)
    version = layer.base_layer.weight._version
    manager._activate_adapter(2, 1.0)

    assert torch.equal(layer.base_layer.weight, _expected(base, second))
    assert layer.base_layer.weight._version == version + 1
    assert layer.base_layer.weight.data_ptr() == address
    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, base)


def test_target_change_restores_only_old_target():
    manager, foo, bar = _two_layers()
    first_foo = _named_lora("foo", 6, 20)
    first_bar = _named_lora("bar", 5, 21)
    second_foo = _named_lora("foo", 6, 22)
    manager._registered_adapters[1] = _StubLoRAModel({"transformer.foo": first_foo, "transformer.bar": first_bar})
    manager._registered_adapters[2] = _StubLoRAModel({"transformer.foo": second_foo})
    foo_base = foo.base_layer.weight.detach().clone()
    bar_base = bar.base_layer.weight.detach().clone()

    manager._activate_adapter(1, 1.0)
    manager._activate_adapter(2, 1.0)

    assert torch.equal(foo.base_layer.weight, _expected(foo_base, second_foo))
    assert torch.equal(bar.base_layer.weight, bar_base)
    assert manager._merged_layer_names == {"transformer.foo"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_repeated_switches_do_not_accumulate(dtype: torch.dtype):
    layer = _MergeableLoRALayer((6,))
    layer.base_layer.weight = torch.nn.Parameter(layer.base_layer.weight.detach().to(dtype))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(2, 6, 30)
    second = _make_lora(2, 6, 31)
    _register(manager, 1, first)
    _register(manager, 2, second)
    base = layer.base_layer.weight.detach().clone()

    for update in range(100):
        adapter_id, lora = (1, first) if update % 2 == 0 else (2, second)
        manager._activate_adapter(adapter_id, 1.0)
        assert torch.equal(layer.base_layer.weight, _expected(base, lora))

    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, base)


def test_base_update_after_deactivation_gets_a_fresh_snapshot():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    _register(manager, 1, _make_lora(2, 6, 52))
    second = _make_lora(2, 6, 53)
    _register(manager, 2, second)
    manager._activate_adapter(1, 1.0)
    manager._deactivate_all_adapters()
    with torch.no_grad():
        layer.base_layer.weight.add_(3.0)
    new_base = layer.base_layer.weight.detach().clone()

    manager._activate_adapter(2, 1.0)

    assert torch.equal(layer.base_layer.weight, _expected(new_base, second))
    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, new_base)
