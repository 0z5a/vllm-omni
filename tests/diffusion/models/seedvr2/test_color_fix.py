# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Colour transfer that realigns SeedVR2 output with its resized input."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from vllm_omni.diffusion.models.seedvr2.color_fix import (
    COLOR_CORRECTION_METHODS,
    DEFAULT_COLOR_CORRECTION_METHOD,
    _lab_to_srgb,
    _srgb_to_lab,
    correct_video_color,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
TRANSFER_METHODS = [method for method in COLOR_CORRECTION_METHODS if method != "none"]
# A visible cast: roughly 26, 10 and -15 of 255 per channel, plus a contrast drop.
TINT = (0.10, 0.04, -0.06)


def _clip(frames: int = 2, height: int = 64, width: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """A reference clip and a restored clip that is tinted but sharper."""
    generator = torch.Generator().manual_seed(7723)
    reference = torch.rand(1, 3, frames, height, width, generator=generator)
    reference = reference * 0.3 + torch.linspace(0.2, 0.8, width).view(1, 1, 1, 1, width)
    detail = torch.rand(1, 3, frames, height, width, generator=generator) * 0.2 - 0.1
    restored = reference * 0.85 + detail + torch.tensor(TINT).view(1, 3, 1, 1, 1)
    return reference.clamp(0, 1), restored.clamp(0, 1)


def _channel_drift(video: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-channel mean difference on the 0-255 scale."""
    return (video.mean((0, 2, 3, 4)) - reference.mean((0, 2, 3, 4))).abs() * 255


def _sharpness(video: torch.Tensor) -> float:
    """Laplacian variance, a standard proxy for retained high-frequency detail."""
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    gray = video.mean(1).flatten(0, 1).unsqueeze(1)
    return F.conv2d(gray, kernel).var().item()


def test_default_method_is_enabled() -> None:
    """Colour correction must be on by default, matching the reference tool."""
    assert DEFAULT_COLOR_CORRECTION_METHOD == "lab"
    assert DEFAULT_COLOR_CORRECTION_METHOD in COLOR_CORRECTION_METHODS
    reference, restored = _clip()
    assert torch.equal(correct_video_color(restored, reference), correct_video_color(restored, reference, method="lab"))


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_removes_the_colour_cast(method: str) -> None:
    reference, restored = _clip()
    corrected = correct_video_color(restored, reference, method=method)
    assert _channel_drift(restored, reference).max() > 10.0, "fixture should start with a visible cast"
    assert _channel_drift(corrected, reference).max() < 5.0


@pytest.mark.parametrize("method", ["lab", "wavelet"])
def test_transfer_keeps_restored_detail(method: str) -> None:
    """The point of restoration is detail, so only colour may change."""
    reference, restored = _clip()
    corrected = correct_video_color(restored, reference, method=method)
    assert _sharpness(corrected) > 0.9 * _sharpness(restored)


def test_none_returns_the_input_unchanged() -> None:
    reference, restored = _clip()
    assert torch.equal(correct_video_color(restored, reference, method="none"), restored)


@pytest.mark.parametrize("method", COLOR_CORRECTION_METHODS)
def test_inputs_are_never_modified(method: str) -> None:
    reference, restored = _clip()
    expected_reference, expected_restored = reference.clone(), restored.clone()
    corrected = correct_video_color(restored, reference, method=method)
    assert torch.equal(reference, expected_reference)
    assert torch.equal(restored, expected_restored)
    assert corrected.shape == restored.shape and corrected.dtype == restored.dtype


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_matching_input_is_a_fixed_point(method: str) -> None:
    """Nothing to transfer when the restoration already matches the reference."""
    reference, _ = _clip()
    corrected = correct_video_color(reference.clone(), reference, method=method)
    assert torch.allclose(corrected, reference, atol=2e-3)


def test_adain_matches_per_channel_statistics() -> None:
    # Confined to mid range so the output clamp cannot perturb the statistics.
    generator = torch.Generator().manual_seed(7723)
    reference = torch.rand(1, 3, 2, 64, 80, generator=generator) * 0.5 + 0.25
    restored = reference * 0.6 + 0.1
    corrected = correct_video_color(restored, reference, method="adain")
    for frame in range(reference.shape[2]):
        target, actual = reference[:, :, frame], corrected[:, :, frame]
        assert torch.allclose(actual.mean((2, 3)), target.mean((2, 3)), atol=1e-5)
        # Population deviation, since the transfer regularizes the variance.
        deviations = (actual.std((2, 3), correction=0), target.std((2, 3), correction=0))
        assert torch.allclose(*deviations, atol=1e-4)


def test_cielab_round_trip_is_lossless() -> None:
    """Guards the sRGB/CIELAB constants the transfer depends on."""
    generator = torch.Generator().manual_seed(7723)
    rgb = torch.rand(2, 3, 32, 32, generator=generator)
    assert torch.allclose(_lab_to_srgb(_srgb_to_lab(rgb)), rgb, atol=1e-5)
    # Neutral grey has no chroma, and white sits at L* = 100.
    lab = _srgb_to_lab(torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]).view(2, 3, 1, 1))
    assert torch.allclose(lab[:, 1:], torch.zeros(2, 2, 1, 1), atol=1e-4)
    assert lab[1, 0].item() == pytest.approx(100.0, abs=1e-3)


def test_unknown_method_is_rejected() -> None:
    reference, restored = _clip()
    with pytest.raises(ValueError, match="color_correction_method"):
        correct_video_color(restored, reference, method="histogram")


def test_mismatched_reference_is_rejected() -> None:
    reference, restored = _clip()
    with pytest.raises(ValueError, match="matching shapes"):
        correct_video_color(restored, reference[:, :, :, :32], method="lab")


@pytest.mark.parametrize("luminance_weight", [-0.1, 1.5])
def test_luminance_weight_bounds(luminance_weight: float) -> None:
    reference, restored = _clip()
    with pytest.raises(ValueError, match="luminance_weight"):
        correct_video_color(restored, reference, luminance_weight=luminance_weight)
