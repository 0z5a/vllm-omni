"""K7-H: preserve the existing explicit EP/HSDP rejection."""

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig


@pytest.mark.parametrize("overlay", ({}, {"ulysses_degree": 2}, {"ring_degree": 2}, {"cfg_parallel_size": 2}))
def test_ep_hsdp_rejected(overlay: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="HSDP .* is not compatible with EP"):
        DiffusionParallelConfig(enable_expert_parallel=True, use_hsdp=True, hsdp_shard_size=2, **overlay)


def test_hsdp_without_ep_remains_valid() -> None:
    assert DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2).world_size == 2
