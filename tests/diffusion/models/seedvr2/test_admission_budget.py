# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Admission budget selection and its device-specific overrides."""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import _admission_budget
from vllm_omni.diffusion.models.seedvr2.video import (
    MAX_CLIP_PIXELS,
    MAX_FRAME_PIXELS,
    MAX_SHARDED_CLIP_PIXELS,
    MAX_SHARDED_FRAME_PIXELS,
    _budget_from_env,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
DEFAULT = (MAX_FRAME_PIXELS, MAX_CLIP_PIXELS)
SHARDED = (MAX_SHARDED_FRAME_PIXELS, MAX_SHARDED_CLIP_PIXELS)


def config(*, tiling: bool = True, ulysses: int = 1, vae_patch: int | None = None) -> OmniDiffusionConfig:
    od_config = OmniDiffusionConfig()
    od_config.vae_use_tiling = tiling
    od_config.parallel_config.ulysses_degree = ulysses
    od_config.parallel_config.vae_patch_parallel_size = ulysses if vae_patch is None else vae_patch
    return od_config


@pytest.mark.parametrize("degree", [4, 8, 16])
def test_sharded_profile_admits_the_larger_budget(degree: int) -> None:
    """Degrees above the calibrated four must not fall back to the small budget."""
    assert _admission_budget(config(ulysses=degree)) == SHARDED


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_degrees_below_four_keep_the_default_budget(degree: int) -> None:
    assert _admission_budget(config(ulysses=degree)) == DEFAULT


def test_unsharded_vae_keeps_the_default_budget() -> None:
    """The larger budget assumes VAE activations are sharded across the same ranks."""
    assert _admission_budget(config(ulysses=8, vae_patch=2)) == DEFAULT
    assert _admission_budget(config(ulysses=8, vae_patch=1)) == DEFAULT
    assert _admission_budget(config(tiling=False, ulysses=8)) == DEFAULT


def test_more_ranks_never_admit_less() -> None:
    budgets = [_admission_budget(config(ulysses=degree)) for degree in range(1, 17)]
    assert all(later >= earlier for earlier, later in zip(budgets, budgets[1:]))


def test_override_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEEDVR2_TEST_BUDGET", raising=False)
    assert _budget_from_env("SEEDVR2_TEST_BUDGET", 7723) == 7723


def test_override_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEEDVR2_TEST_BUDGET", "4128768")
    assert _budget_from_env("SEEDVR2_TEST_BUDGET", 7723) == 4128768


@pytest.mark.parametrize("value", ["0", "-5", "1.5", "", "  ", "lots", "1e6"])
def test_override_rejects_values_that_are_not_positive_integers(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEEDVR2_TEST_BUDGET", value)
    with pytest.raises(ValueError, match="SEEDVR2_TEST_BUDGET"):
        _budget_from_env("SEEDVR2_TEST_BUDGET", 7723)
