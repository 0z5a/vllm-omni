# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.models.ernie_image import pipeline_ernie_image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pipeline_loads_distributed_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch silently ignored VAE parallel settings before loading model weights."""
    monkeypatch.setattr(pipeline_ernie_image, "get_local_device", lambda: torch.device("cpu"))
    for loader in (
        pipeline_ernie_image.FlowMatchEulerDiscreteScheduler,
        pipeline_ernie_image.AutoModel,
        pipeline_ernie_image.AutoTokenizer,
    ):
        monkeypatch.setattr(loader, "from_pretrained", Mock())

    class VaeLoadReachedError(Exception):
        pass

    def check_vae_class(cls: type[AutoencoderKLFlux2], model: str, **kwargs: object) -> None:
        assert issubclass(cls, DistributedVaeMixin)
        assert model == "test/ernie-image"
        assert kwargs["subfolder"] == "vae"
        assert kwargs["torch_dtype"] == torch.bfloat16
        raise VaeLoadReachedError

    monkeypatch.setattr(AutoencoderKLFlux2, "from_pretrained", classmethod(check_vae_class))
    with pytest.raises(VaeLoadReachedError):
        pipeline_ernie_image.ErnieImagePipeline(
            od_config=OmniDiffusionConfig(model="test/ernie-image", dtype=torch.bfloat16)
        )
