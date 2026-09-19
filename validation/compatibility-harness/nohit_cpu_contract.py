"""Verify the real TeaCache decision policy computes every no-hit step."""

import torch
from compat_probe import ObservedTeaCacheHook

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

hook = ObservedTeaCacheHook(TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[0.0, 0.0, 0.0, 0.0, 1.0]))
state = TeaCacheState()
for request in range(3):
    state.reset()
    hook.decisions.clear()
    for step in range(9):
        signal = torch.ones(1, 8) * (step + 1)
        assert hook._should_compute_full_transformer(state, signal)
        state.previous_modulated_input = signal
        state.previous_residual = torch.zeros(1, 8)
        state.cnt += 1
    assert len(hook.decisions) == 9
    assert hook.decisions[0] == {"step": 0, "compute": True, "residual_present": False}
    print(f"request={request}: 9/9 compute decisions; reset verified", flush=True)
