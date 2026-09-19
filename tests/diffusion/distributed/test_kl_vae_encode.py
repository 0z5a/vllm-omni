# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare distributed tile scheduling with native KL encoding."""

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedOperator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class LocalTileExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, image: torch.Tensor, operator: DistributedOperator, broadcast_result: bool) -> torch.Tensor:
        assert broadcast_result
        self.calls += 1
        tasks, grid = operator.split(image)
        # Completion order must not change spatial assembly.
        outputs = {task.grid_coord: operator.exec(task) for task in reversed(tasks)}
        return operator.merge(outputs, grid)


@pytest.mark.parametrize(
    ("native_cls", "distributed_cls"),
    [(AutoencoderKL, DistributedAutoencoderKL), (AutoencoderKLFlux2, DistributedAutoencoderKLFlux2)],
)
@pytest.mark.parametrize("use_quant_conv", [False, True])
def test_tiled_encode_preserves_posterior_and_rng(monkeypatch, native_cls, distributed_cls, use_quant_conv):
    config = dict(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
        use_quant_conv=use_quant_conv,
    )
    torch.manual_seed(7)
    native = native_cls(**config).eval()
    candidate = distributed_cls(**config).eval()
    candidate.load_state_dict(native.state_dict())
    native.enable_tiling()
    candidate.enable_tiling()
    executor = LocalTileExecutor()
    monkeypatch.setattr(candidate, "is_distributed_enabled", lambda: True)
    monkeypatch.setattr(candidate, "distributed_executor", executor, raising=False)
    with torch.inference_mode():
        for height, width in ((16, 16), (24, 32), (32, 24), (16, 16)):
            image = torch.randn(2, 3, height, width)
            rng_before = torch.random.get_rng_state()
            expected = native.encode(image).latent_dist
            actual = candidate.encode(image).latent_dist
            assert torch.equal(torch.random.get_rng_state(), rng_before)
            torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
            torch.testing.assert_close(actual.mode(), expected.mode(), rtol=0, atol=0)
            torch.testing.assert_close(
                actual.sample(generator=torch.Generator().manual_seed(42)),
                expected.sample(generator=torch.Generator().manual_seed(42)),
                rtol=0,
                atol=0,
            )
    assert executor.calls == 2
