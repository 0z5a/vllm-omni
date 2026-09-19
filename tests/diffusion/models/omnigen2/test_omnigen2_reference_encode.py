# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedOperator
from vllm_omni.diffusion.models.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ReferenceExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, image: torch.Tensor, operator: DistributedOperator, broadcast_result: bool) -> torch.Tensor:
        assert broadcast_result
        self.calls += 1
        tasks, grid = operator.split(image)
        return operator.merge({task.grid_coord: operator.exec(task) for task in reversed(tasks)}, grid)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("shapes", [(), ((16, 16),), ((16, 24), (24, 16)), ((8, 16), (24, 16), (16, 8))])
def test_reference_order_and_rng(monkeypatch: pytest.MonkeyPatch, parallel: bool, shapes: tuple) -> None:
    vae = DistributedAutoencoderKL(
        block_out_channels=(8, 8),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
        shift_factor=0.1159,
        scaling_factor=0.3611,
    ).eval()
    pipeline = OmniGen2Pipeline.__new__(OmniGen2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae = vae
    pipeline.vae_scale_factor = 2
    executor = ReferenceExecutor()
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: parallel)
    monkeypatch.setattr(vae, "distributed_executor", executor, raising=False)
    images = [torch.randn(1, 3, height, width) for height, width in shapes]
    with torch.inference_mode():
        torch.manual_seed(142)
        expected = [pipeline.encode_vae(image).squeeze(0) for image in images]
        rng_expected = torch.random.get_rng_state()
        torch.manual_seed(142)
        actual = pipeline._encode_reference_images(images)
        assert torch.equal(torch.random.get_rng_state(), rng_expected)
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert executor.calls == int(parallel and len(images) > 1)
