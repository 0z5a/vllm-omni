# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("enabled", [False, True])
def test_cross_attention_precompute_respects_offload_setting(enabled):
    model = HeliosTransformer3DModel.__new__(HeliosTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.eval()
    model.cache_cross_attention = enabled
    model._cross_attn_kv_cache = OrderedDict()
    model._cross_attn_cache_size = 2
    block = Mock()
    hidden = torch.randn(1, 2, 4)
    block.attn2.project_kv.return_value = (hidden, hidden)
    model.blocks = [block]

    with torch.no_grad():
        first = model._get_cross_attn_key_values(hidden)
        second = model._get_cross_attn_key_values(hidden)

    if enabled:
        assert first is second
        block.attn2.project_kv.assert_called_once_with(hidden)
    else:
        assert first is second is None
        block.attn2.project_kv.assert_not_called()
