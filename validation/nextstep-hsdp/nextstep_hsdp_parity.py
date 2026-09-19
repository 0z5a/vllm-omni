"""Tiny real NextStep model: HSDP call coverage and cached decode parity."""
import json
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers.cache_utils import StaticCache
from vllm.config import VllmConfig, set_current_vllm_config
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep import NextStepConfig, NextStepModel


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(cfg_parallel_size=2)
    torch.manual_seed(142)
    torch.set_default_dtype(torch.bfloat16)
    config = NextStepConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                            latent_channels=4, latent_patch_size=1, fm_head_dim=32, fm_head_layers=2,
                            pad_token_id=0, image_placeholder_id=63)
    model = NextStepModel(config).eval().to(rank)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.02)
    reference = NextStepModel(config).eval().to(rank)
    reference.load_state_dict(model.state_dict())
    model = apply_hsdp_to_model(model, HSDPInferenceConfig(enabled=True, hsdp_shard_size=2),
                               target_device=torch.device('cuda', rank))
    assert sum(isinstance(module, FSDPModule) for module in model.modules()) == 7
    with torch.no_grad():
        for prefix_length, cfg_mult in ((4, 1), (7, 2), (6, 3), (4, 1)):
            ids = torch.arange(1, prefix_length + 1, device=rank).unsqueeze(0).repeat(cfg_mult, 1)
            mask = torch.ones_like(ids)
            mask[:, 0] = 0
            expected = reference(inputs_embeds=reference.prepare_inputs_embeds(ids), attention_mask=mask, past_key_values=StaticCache(config=config, max_cache_len=prefix_length + 3), use_cache=True)
            actual = model(input_ids=ids, attention_mask=mask, past_key_values=StaticCache(config=config, max_cache_len=prefix_length + 3), use_cache=True)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
            for step in range(3):
                torch.manual_seed(142 + step)
                expected_token = reference.image_head.sample(reference.image_out_projector(expected.last_hidden_state[:, -1]),
                                                            cfg=2.0 if cfg_mult > 1 else 1.0, cfg_img=1.5 if cfg_mult == 3 else 1.0, cfg_mult=cfg_mult, num_sampling_steps=3)
                expected_rng = torch.cuda.get_rng_state()
                torch.manual_seed(142 + step)
                actual_token = model.image_head.sample(model.image_out_projector(actual.last_hidden_state[:, -1]),
                                                      cfg=2.0 if cfg_mult > 1 else 1.0, cfg_img=1.5 if cfg_mult == 3 else 1.0, cfg_mult=cfg_mult, num_sampling_steps=3)
                torch.testing.assert_close(actual_token, expected_token, rtol=0, atol=0)
                assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
                expected = reference(inputs_embeds=reference.image_in_projector(expected_token[:, None].to(torch.bfloat16)).repeat(cfg_mult, 1, 1),
                                     attention_mask=mask, past_key_values=expected.past_key_values, use_cache=True)
                actual = model(inputs_embeds=model.image_in_projector(actual_token[:, None].to(torch.bfloat16)).repeat(cfg_mult, 1, 1),
                               attention_mask=mask, past_key_values=actual.past_key_values, use_cache=True)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
            assert all(isinstance(p, DTensor) for p in model.parameters())
            print(json.dumps(dict(rank=rank,prefix_length=prefix_length,cfg_mult=cfg_mult,decode_steps=3,exact=True,rng_equal=True)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with set_current_vllm_config(VllmConfig()):
        main()
