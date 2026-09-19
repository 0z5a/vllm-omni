# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKL

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.models.omnigen2 import pipeline_omnigen2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pipeline_loads_distributed_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_omnigen2, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_omnigen2, "prefetch_subfolders", Mock())
    monkeypatch.setattr(pipeline_omnigen2.FlowMatchEulerDiscreteScheduler, "from_pretrained", Mock())

    class VaeLoadReachedError(Exception):
        pass

    def check_vae_class(cls: type[AutoencoderKL], model: str, **kwargs: object) -> None:
        assert issubclass(cls, DistributedVaeMixin)
        assert model == "test/omnigen2"
        assert kwargs["subfolder"] == "vae"
        raise VaeLoadReachedError

    monkeypatch.setattr(AutoencoderKL, "from_pretrained", classmethod(check_vae_class))
    with pytest.raises(VaeLoadReachedError):
        pipeline_omnigen2.OmniGen2Pipeline(od_config=OmniDiffusionConfig(model="test/omnigen2", dtype=torch.bfloat16))
