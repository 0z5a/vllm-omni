# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.seedvr2 import pipeline_seedvr2 as pipeline
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def native_pipeline(monkeypatch, tmp_path):
    torch.save(torch.zeros(2, 5120), tmp_path / "pos_emb.pt")
    monkeypatch.setattr(pipeline, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "SeedVR2NaDiT", lambda **kwargs: nn.Linear(2, 2))
    monkeypatch.setattr(pipeline, "SeedVR2VAE", lambda: nn.Linear(2, 2))
    config = OmniDiffusionConfig(model=str(tmp_path), dtype=torch.float16, enforce_eager=True)
    return pipeline.SeedVR2Pipeline(od_config=config)


def test_registry_loads_native_pipeline():
    assert DiffusionModelRegistry._try_load_model_cls("SeedVR2Pipeline") is pipeline.SeedVR2Pipeline


@pytest.mark.parametrize("frames,padded", [(1, 1), (5, 5), (6, 9), (9, 9)])
def test_frame_padding_preserves_input_and_repeats_tail(frames, padded):
    video = torch.linspace(0, 1, frames).reshape(frames, 1, 1, 1).expand(-1, 3, 16, 32).clone()
    original = video.clone()
    result = pipeline.prepare_video(video, 16, 32)
    assert result.shape == (1, 3, padded, 16, 32)
    torch.testing.assert_close(result[0, :, :frames].permute(1, 0, 2, 3), video * 2 - 1)
    torch.testing.assert_close(
        result[:, :, frames - 1 :], result[:, :, frames - 1 : frames].expand(-1, -1, padded - frames + 1, -1, -1)
    )
    assert torch.equal(video, original)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_invalid_rgb_rejected(bad):
    with pytest.raises(ValueError):
        pipeline.prepare_video(torch.full((5, 3, 16, 32), bad), 16, 32)


def test_checkpoint_exact_coverage(native_pipeline):
    weights = {name: torch.full_like(value, 3) for name, value in native_pipeline.state_dict().items()}
    loaded = native_pipeline.load_weights(weights.items())
    assert loaded == set(weights)
    for name, value in native_pipeline.state_dict().items():
        assert torch.equal(value, weights[name])


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "shape"])
def test_checkpoint_rejects_incomplete_or_ambiguous_weights(native_pipeline, failure):
    weights = list(native_pipeline.state_dict().items())
    if failure == "missing":
        weights.pop()
    elif failure == "duplicate":
        weights.append(weights[0])
    elif failure == "unexpected":
        weights.append(("wrong", torch.zeros(1)))
    else:
        weights[0] = (weights[0][0], torch.zeros(7))
    with pytest.raises(ValueError):
        native_pipeline.load_weights(weights)


def test_unsupported_sampler_rejected_before_encoding(native_pipeline):
    request = OmniDiffusionRequest(
        prompt={"prompt": ""},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, guidance_scale=1.0),
        request_id="unsupported-steps",
    )
    with pytest.raises(ValueError, match="one Euler step"):
        native_pipeline(DiffusionRequestBatch(requests=[request]))


def test_preprocessor_preserves_frame_count_and_rejects_prompt_text():
    request = OmniDiffusionRequest(
        prompt={"prompt": " ", "multi_modal_data": {"video": torch.zeros(6, 3, 16, 32)}},
        sampling_params=OmniDiffusionSamplingParams(height=16, width=32, seed=42),
        request_id="six-frame-clip",
    )
    assert pipeline.prepare_request(request) is request
    assert request.prepared_layout.frame_count == 6
    assert request.prepared_layout.sample.shape == (1, 3, 9, 16, 32)
    request.prompt["prompt"] = "an unsupported text override"
    with pytest.raises(ValueError, match="fixed checkpoint conditioning"):
        pipeline.prepare_request(request)


def test_engine_warmup_builds_restoration_input():
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig(model_class_name="SeedVR2Pipeline")
    request = engine._make_dummy_request(height=16, width=32, guidance_scale=5.0, num_inference_steps=2)
    pipeline.prepare_request(request)
    assert request.sampling_params.guidance_scale == 1.0
    assert request.prepared_layout.sample.shape == (1, 3, 1, 16, 32)


def test_noise_preserves_reference_rng_assignment_for_channel_last_view():
    condition = torch.empty(16, 2, 16, 28).permute(1, 2, 3, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7723)
        expected = torch.randn_like(condition)
    actual = pipeline.sample_noise(condition, torch.Generator().manual_seed(7723))
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    contiguous_noise = torch.randn(condition.shape, generator=torch.Generator().manual_seed(7723))
    assert not torch.equal(actual, contiguous_noise)
