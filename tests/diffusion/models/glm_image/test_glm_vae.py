# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKL

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.models.glm_image import pipeline_glm_image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pipeline_loads_distributed_vae(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_glm_image, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_glm_image.FlowMatchEulerDiscreteScheduler, "from_pretrained", Mock())
    monkeypatch.setattr(pipeline_glm_image.T5EncoderModel, "from_pretrained", Mock(return_value=torch.nn.Identity()))
    monkeypatch.setattr(pipeline_glm_image.ByT5Tokenizer, "from_pretrained", Mock())

    class VaeLoadReachedError(Exception):
        pass

    def check_vae_class(cls: type[AutoencoderKL], model: str, **kwargs: object) -> None:
        assert issubclass(cls, DistributedVaeMixin)
        assert model == str(tmp_path)
        assert kwargs["subfolder"] == "vae"
        raise VaeLoadReachedError

    monkeypatch.setattr(AutoencoderKL, "from_pretrained", classmethod(check_vae_class))
    with pytest.raises(VaeLoadReachedError):
        pipeline_glm_image.GlmImagePipeline(od_config=OmniDiffusionConfig(model=str(tmp_path), dtype=torch.bfloat16))
