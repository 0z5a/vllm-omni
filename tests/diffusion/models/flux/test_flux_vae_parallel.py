# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
from pathlib import Path

import pytest

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.models.flux import pipeline_flux, pipeline_flux_kontext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("module", [pipeline_flux, pipeline_flux_kontext])
def test_flux_pipeline_uses_distributed_vae(module):
    assert module.DistributedAutoencoderKL is DistributedAutoencoderKL
    assert issubclass(module.DistributedAutoencoderKL, DistributedVaeMixin)


@pytest.mark.parametrize("module", [pipeline_flux, pipeline_flux_kontext])
def test_flux_pipeline_preserves_vae_loading_contract(module):
    tree = ast.parse(Path(module.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DistributedAutoencoderKL"
        and node.func.attr == "from_pretrained"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "model"
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["subfolder"], ast.Constant)
    assert keywords["subfolder"].value == "vae"
    assert isinstance(keywords["local_files_only"], ast.Name)
    assert keywords["local_files_only"].id == "local_files_only"
