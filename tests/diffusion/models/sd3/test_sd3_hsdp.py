# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.sd3.sd3_transformer import SD3Transformer2DModel, SD3TransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sd3_hsdp_shards_only_joint_transformer_blocks():
    model = object.__new__(SD3Transformer2DModel)
    nn.Module.__init__(model)
    blocks = [object.__new__(SD3TransformerBlock) for _ in range(2)]
    for block in blocks:
        nn.Module.__init__(block)
    model.transformer_blocks = nn.ModuleList(blocks)
    model.proj_out = nn.Linear(4, 4)

    matches = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in model._hsdp_shard_conditions)
    ]

    assert matches == ["transformer_blocks.0", "transformer_blocks.1"]
