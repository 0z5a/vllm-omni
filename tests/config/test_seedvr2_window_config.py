# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.config_factory import StageConfigFactory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("architecture,degree", [("SeedVR2Pipeline", 1), ("SeedVR2Pipeline", 2), ("WanPipeline", 1)])
def test_generic_diffusion_factory_owns_window_parallel_option(architecture, degree):
    config = StageConfigFactory.create_typed_default_diffusion(
        "/tmp/model",
        {"model_class_name": architecture, "window_parallel_size": degree, "num_gpus": degree},
    )
    parallel = config.stage_configs[0].parallel_config
    assert parallel.window_parallel_size == degree
    assert parallel.sequence_parallel_size == degree
