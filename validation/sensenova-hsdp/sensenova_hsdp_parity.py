"""Tiny real SenseNova modules: embedding, cached AR and denoising under HSDP."""

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from vllm.config import set_current_vllm_config
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp import (
    HSDPInferenceConfig,
    apply_hsdp_to_model,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1ForCausalLM,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1LLMConfig


def main() -> None:
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(cfg_parallel_size=2)
    torch.manual_seed(142)
    torch.set_default_dtype(torch.bfloat16)
    config = SenseNovaU1LLMConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        max_position_embeddings_hw=128,
    )
    od_config = OmniDiffusionConfig(model="tiny-sensenova", dtype=torch.bfloat16)
    with (
        set_current_vllm_config(create_diffusion_vllm_config(device, od_config)),
        set_current_diffusion_config(od_config),
    ):
        model = SenseNovaU1ForCausalLM(config).eval().to(device)
        reference = SenseNovaU1ForCausalLM(config).eval().to(device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, 0.02)
        reference.load_state_dict(model.state_dict())
        apply_hsdp_to_model(
            model,
            HSDPInferenceConfig(enabled=True, hsdp_shard_size=2),
            target_device=device,
        )
        assert sum(isinstance(module, FSDPModule) for module in model.modules()) == 3
        with torch.no_grad():
            for length, batch in ((4, 1), (7, 2), (6, 3), (4, 1)):
                torch.manual_seed(142 + length)
                ids = torch.arange(1, length + 1, device=device).unsqueeze(0).repeat(batch, 1)
                expected_embed = reference(input_ids=ids, embed_only=True).inputs_embeds
                actual_embed = model(input_ids=ids, embed_only=True).inputs_embeds
                torch.testing.assert_close(actual_embed, expected_embed, rtol=0, atol=0)
                assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
                indexes = torch.stack([torch.arange(length, device=device)] * 3)
                expected = reference(inputs_embeds=expected_embed, indexes=indexes, use_cache=True)
                actual = model(inputs_embeds=actual_embed, indexes=indexes, use_cache=True)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                for step in range(3):
                    next_ids = torch.full((batch, 1), 12 + step, device=device, dtype=torch.long)
                    indexes = torch.full((3, 1), length + step, device=device, dtype=torch.long)
                    expected = reference(
                        input_ids=next_ids,
                        indexes=indexes,
                        past_key_values=expected.past_key_values,
                        use_cache=True,
                    )
                    actual = model(
                        input_ids=next_ids,
                        indexes=indexes,
                        past_key_values=actual.past_key_values,
                        use_cache=True,
                    )
                    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                expected_cache = expected.past_key_values
                actual_cache = actual.past_key_values
                prefix_keys = [layer.keys.clone() for layer in actual_cache.layers]
                for image_tokens in (4, 6, 4):
                    inputs = torch.randn(batch, image_tokens, config.hidden_size, device=device)
                    indexes = torch.stack([torch.arange(image_tokens, device=device)] * 3)
                    options = {
                        "inputs_embeds": inputs,
                        "indexes": indexes,
                        "image_gen_indicators": torch.ones(batch, image_tokens, dtype=torch.bool, device=device),
                        "attention_mask": {"full_attention": None},
                        "update_cache": False,
                        "use_cache": True,
                        "compute_logits": False,
                    }
                    before_rng = torch.cuda.get_rng_state()
                    expected = reference(past_key_values=expected_cache, **options)
                    actual = model(past_key_values=actual_cache, **options)
                    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)
                    assert torch.equal(torch.cuda.get_rng_state(), before_rng)
                    for layer, saved in zip(actual_cache.layers, prefix_keys):
                        torch.testing.assert_close(layer.keys, saved, rtol=0, atol=0)
                    assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "prefix_length": length,
                            "batch": batch,
                            "decode_steps": 3,
                            "generation_lengths": [4, 6, 4],
                            "exact": True,
                            "prefix_unchanged": True,
                        }
                    ),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
