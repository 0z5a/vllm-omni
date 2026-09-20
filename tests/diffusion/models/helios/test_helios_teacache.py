# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _model() -> HeliosTransformer3DModel:
    model = HeliosTransformer3DModel.__new__(HeliosTransformer3DModel)
    torch.nn.Module.__init__(model)
    model._tea_cache_config = TeaCacheConfig(transformer_type="HeliosTransformer3DModel", rel_l1_thresh=0.2)
    model._tea_cache_states = {"positive": TeaCacheState(), "negative": TeaCacheState()}
    model._tea_cache_forward_count = 0
    model.do_true_cfg = False
    return model


def test_teacache_accumulates_identity_relative_l1_and_resets() -> None:
    model = _model()
    state = model._tea_cache_states["positive"]
    baseline = torch.ones(1, 2, 3)

    assert model._teacache_should_compute(state, baseline)
    state.previous_modulated_input = baseline
    state.cnt = 1
    assert not model._teacache_should_compute(state, baseline * 1.1)
    state.previous_modulated_input = baseline * 1.1
    state.cnt = 2
    assert model._teacache_should_compute(state, baseline * 1.21)

    state.previous_residual = baseline
    model.reset_teacache()
    assert state.cnt == 0
    assert state.previous_modulated_input is None
    assert state.previous_residual is None


def test_teacache_keeps_serial_cfg_branches_separate() -> None:
    model = _model()
    model.do_true_cfg = True

    with patch(
        "vllm_omni.diffusion.models.helios.helios_transformer.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        assert model._teacache_state() is model._tea_cache_states["positive"]
        model._tea_cache_forward_count = 1
        assert model._teacache_state() is model._tea_cache_states["negative"]


def test_teacache_sp_decision_uses_local_value_for_one_rank() -> None:
    with patch(
        "vllm_omni.diffusion.models.helios.helios_transformer.get_sequence_parallel_world_size",
        return_value=1,
    ):
        assert HeliosTransformer3DModel._sync_teacache_decision(False, torch.device("cpu")) is False
        assert HeliosTransformer3DModel._sync_teacache_decision(True, torch.device("cpu")) is True


def test_teacache_sp_decision_computes_when_any_rank_requires_it() -> None:
    class SPGroup:
        def all_reduce(self, decision: torch.Tensor, op: torch.distributed.ReduceOp) -> None:
            assert op == torch.distributed.ReduceOp.MAX
            decision.fill_(1)

    with (
        patch(
            "vllm_omni.diffusion.models.helios.helios_transformer.get_sequence_parallel_world_size",
            return_value=2,
        ),
        patch(
            "vllm_omni.diffusion.models.helios.helios_transformer.get_sp_group",
            return_value=SPGroup(),
        ),
    ):
        assert HeliosTransformer3DModel._sync_teacache_decision(False, torch.device("cpu")) is True
