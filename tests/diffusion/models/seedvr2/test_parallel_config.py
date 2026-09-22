# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeedVR2 reuses SP configuration without changing other models' modes."""

import pytest

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.seedvr2.parallel import validate_seedvr2_parallel_config

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.sp,
    pytest.mark.core_model,
    pytest.mark.cpu,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_window_sp_accepts_regular_sp_config(degree):
    parallel = DiffusionParallelConfig(ulysses_degree=degree)
    validate_seedvr2_parallel_config(parallel)
    assert parallel.sequence_parallel_size == degree


@pytest.mark.parametrize(
    "parallel",
    [
        DiffusionParallelConfig(ring_degree=2),
        DiffusionParallelConfig(allgather_degree=2),
        DiffusionParallelConfig(ulysses_degree=2, ulysses_mode="advanced_uaa"),
        DiffusionParallelConfig(ulysses_degree=2, ulysses_a2a_permute=True),
    ],
)
def test_window_sp_rejects_unsupported_attention_modes(parallel):
    with pytest.raises(ValueError, match="SeedVR2 window SP requires pure ulysses_degree"):
        validate_seedvr2_parallel_config(parallel)


def test_no_window_degree_in_framework_config():
    assert "window_parallel_size" not in DiffusionParallelConfig.__dataclass_fields__


@pytest.mark.parametrize("ci", [False, True])
def test_checkpoint_absence_is_not_silently_skipped_in_ci(ci, monkeypatch, tmp_path):
    from tests.diffusion.models.seedvr2 import test_window_sp_gpu

    monkeypatch.setattr(test_window_sp_gpu, "_available_gpus", lambda: 2)
    monkeypatch.setenv("SEEDVR2_CHECKPOINT", str(tmp_path / "missing.safetensors"))
    monkeypatch.setenv("CI", "true" if ci else "")
    monkeypatch.delenv("BUILDKITE", raising=False)
    outcome = pytest.fail.Exception if ci else pytest.skip.Exception
    with pytest.raises(outcome, match="checkpoint"):
        test_window_sp_gpu.test_seedvr2_sp_degree_invariance(2, tmp_path)
