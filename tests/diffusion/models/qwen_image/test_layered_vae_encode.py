# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage_layered import (
    DistributedAutoencoderKLQwenImageLayered,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedOperator
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ReverseExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, x: torch.Tensor, operator: DistributedOperator, broadcast_result: bool) -> torch.Tensor:
        assert broadcast_result
        self.calls += 1
        tasks, grid = operator.split(x)
        return operator.merge({task.grid_coord: operator.exec(task) for task in reversed(tasks)}, grid)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(("batch", "frames"), [(1, 1), (2, 1), (1, 5)])
def test_vendored_posterior_and_cache_parity(monkeypatch: pytest.MonkeyPatch, parallel: bool, batch: int, frames: int):
    native = AutoencoderKLQwenImage(base_dim=4, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1).eval()
    candidate = DistributedAutoencoderKLQwenImageLayered.from_config(native.config).eval()
    candidate.load_state_dict(native.state_dict())
    executor = ReverseExecutor()
    monkeypatch.setattr(candidate, "is_distributed_enabled", lambda: parallel)
    monkeypatch.setattr(candidate, "distributed_executor", executor, raising=False)
    for model in (native, candidate):
        model.enable_tiling(
            tile_sample_min_height=16, tile_sample_min_width=16, tile_sample_stride_height=8, tile_sample_stride_width=8
        )
    with torch.inference_mode():
        for height, width in ((24, 32), (32, 24), (24, 32)):
            x = torch.randn(batch, 3, frames, height, width)
            rng_before = torch.random.get_rng_state()
            expected = native.encode(x).latent_dist
            actual = candidate.encode(x).latent_dist
            torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
            assert torch.equal(torch.random.get_rng_state(), rng_before)
            torch.testing.assert_close(
                actual.sample(generator=torch.Generator().manual_seed(42)),
                expected.sample(generator=torch.Generator().manual_seed(42)),
                rtol=0,
                atol=0,
            )
    assert executor.calls == (3 if parallel else 0)
    assert all(value is None for value in candidate._enc_feat_map)


def test_pipeline_keeps_vendored_vae(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.qwen_image import pipeline_qwen_image_layered as pipeline

    monkeypatch.setattr(pipeline, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "prefetch_subfolders", Mock())
    monkeypatch.setattr(pipeline.FlowMatchEulerDiscreteScheduler, "from_pretrained", Mock())
    monkeypatch.setattr(
        pipeline.Qwen2_5_VLForConditionalGeneration, "from_pretrained", Mock(return_value=torch.nn.Identity())
    )

    class VaeLoadReachedError(Exception):
        pass

    def check_vae_class(cls: type[AutoencoderKLQwenImage], model: str, **kwargs: object) -> None:
        assert cls is DistributedAutoencoderKLQwenImageLayered
        assert issubclass(cls, AutoencoderKLQwenImage)
        assert model == str(tmp_path)
        assert kwargs["subfolder"] == "vae"
        raise VaeLoadReachedError

    monkeypatch.setattr(AutoencoderKLQwenImage, "from_pretrained", classmethod(check_vae_class))
    with pytest.raises(VaeLoadReachedError):
        pipeline.QwenImageLayeredPipeline(od_config=OmniDiffusionConfig(model=str(tmp_path), dtype=torch.bfloat16))
