"""Exercise the installed Pattern-3 probe against real CPU cache state."""

import json

import torch
from cache_dit import DBCacheConfig, ForwardPattern
from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from probe import ProbeRunner


class AddBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.125


for no_hit in (False, True):
    manager = CachedContextManager(name="cpu-contract")
    config = DBCacheConfig(
        num_inference_steps=9,
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=2,
        max_cached_steps=0 if no_hit else -1,
        max_continuous_cached_steps=3,
        residual_diff_threshold=0.24,
    )
    manager.new_context(name="request", cache_config=config)
    wrapper = CachedBlocks_Pattern_3_4_5(
        transformer_blocks=torch.nn.ModuleList([AddBlock(), AddBlock(), AddBlock()]),
        forward_pattern=ForwardPattern.Pattern_3,
        check_forward_pattern=False,
        cache_context="request",
        context_manager=manager,
        cache_prefix="cpu",
    )
    probe = ProbeRunner.__new__(ProbeRunner)
    probe.wrapper = wrapper
    wrapper.register_forward_pre_hook(probe.before_step)
    wrapper.transformer_blocks[0].register_forward_pre_hook(probe.count_first)
    wrapper.transformer_blocks[1].register_forward_pre_hook(probe.count_middle)
    for request in range(3):
        manager.new_context(name="request", cache_config=config)
        probe.steps = []
        probe.first_calls = probe.middle_calls = 0
        x = torch.ones(1, 4 + request, 8)
        for _ in range(9):
            torch.testing.assert_close(wrapper(x), x + 0.375, rtol=0, atol=0)
        hits = manager.get_context().cached_steps
        assert probe.first_calls == 9
        assert probe.middle_calls == 9 - len(hits)
        assert bool(hits) != no_hit
        print(
            json.dumps(
                {
                    "no_hit": no_hit,
                    "request": request,
                    "cached_steps": hits,
                    "first_calls": probe.first_calls,
                    "middle_calls": probe.middle_calls,
                }
            ),
            flush=True,
        )
