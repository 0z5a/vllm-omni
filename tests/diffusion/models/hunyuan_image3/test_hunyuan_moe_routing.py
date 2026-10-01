# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR and DiT must preserve close FP32 router scores across KV reuse."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.hunyuan_image3 import hunyuan_image3_transformer as dit
from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as ar
from vllm_omni.model_executor.models.hunyuan_image3.moe_routing import pack_hunyuan_topk, unpack_hunyuan_topk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("stage", [ar, dit], ids=["ar", "dit"])
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_close_router_scores_select_the_same_fp32_expert(monkeypatch, stage, autocast):
    weights = torch.tensor([[1.0, 0.0], [1.001, 0.0], [0.0, 0.0]])
    assert weights.bfloat16()[:, 0].argmax().item() == 0
    config = SimpleNamespace(
        hidden_size=2,
        num_experts=3,
        moe_topk=1,
        intermediate_size=2,
        moe_intermediate_size=None,
        use_mixed_mlp_moe=0,
    )
    parallel = SimpleNamespace(enable_expert_parallel=False, eplb_config=SimpleNamespace(num_redundant_experts=0))
    monkeypatch.setattr(stage, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(stage, "get_current_vllm_config", lambda: SimpleNamespace(parallel_config=parallel))
    if stage is ar:
        monkeypatch.setattr(
            ar, "get_ep_group", lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0)
        )

    def gate(input_size, output_size, *, bias, quant_config, params_dtype, prefix):
        assert params_dtype is torch.float32 and quant_config is None
        linear = nn.Linear(input_size, output_size, bias=bias, dtype=params_dtype)
        linear.weight.data.copy_(weights)
        return Gate(linear)

    class Gate(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, value):
            assert value.dtype is torch.float32
            return self.linear(value), None

    class Experts(nn.Module):
        def forward(self, *, hidden_states, router_logits):
            routing_weights, indices = unpack_hunyuan_topk(hidden_states, router_logits, 1, False)
            assert routing_weights.dtype is torch.float32
            return indices.to(hidden_states.dtype).expand_as(hidden_states)

    def experts(**kwargs):
        assert kwargs["custom_routing_function"] is unpack_hunyuan_topk
        assert kwargs["renormalize"] is False
        return Experts()

    monkeypatch.setattr(stage, "ReplicatedLinear", gate)
    monkeypatch.setattr(stage, "SharedFusedMoE" if stage is ar else "FusedMoE", experts)
    cls = ar.HunyuanImage3SparseMoeBlock if stage is ar else dit.HunYuanSparseMoeBlock
    block = cls(config, layer_id=0)
    hidden = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        torch.testing.assert_close(block(hidden), torch.ones_like(hidden), rtol=0, atol=0)


def test_routing_weights_keep_hf_model_dtype_rounding():
    logits = torch.tensor([[0.3, 0.5, 0.9]])
    packed = pack_hunyuan_topk(logits, 2, torch.bfloat16)
    weights, indices = unpack_hunyuan_topk(torch.empty(1, 2), packed, 2, False)
    expected = torch.tensor([[0.59868766, 0.40131234]]).bfloat16().float()
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    assert indices.tolist() == [[2, 1]]
