# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Full-checkpoint E2E for the MiniCPM-o 4.5 vision encoder CUDA Graph.

The component tests under ``tests/model_executor/models/minicpmo_4_5`` run the
manager against small production encoders and the protocol entry point with
synthetic tensors. This module is the serving counterpart: it loads the real
Stage 0/1/2 pipeline through the offline ``Omni`` entry point and drives greedy
image/video requests through it twice — once with
``compilation_config.cudagraph_mm_encoder`` enabled and once with the flag
absent — then compares the decoded text. The deploy profile is otherwise
shared, so a difference isolates encoder graph replay.

Graph replay requires a vLLM build that advertises the ``capture_axes``
encoder-cudagraph protocol. On the released pin the flag is accepted and the
encoder stays eager, so the graph-path test skips there and only the
shared-deploy assertions run.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
from PIL import Image
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

from tests.helpers.clean import wait_for_gpu_memory_to_clear
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

_MODEL = "openbmb/MiniCPM-o-4_5"
_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")

# One four-query-group budget for the real checkpoint. In-budget items replay a
# captured tier; anything above it is never captured and stays eager.
_ENCODER_TOKEN_BUDGET = 256

# A 1024x1024 image slices into far more vision tokens than the budget above,
# so the manager cannot select a captured tier for it.
_OVERSIZED_IMAGE_PIXELS = 1024

# The three stages share one 44 GiB card in CI. Both engines in this module must
# be shut down before the next one loads, so the release has to be awaited.
_ENGINE_TEARDOWN_MEMORY_RATIO = 0.05


def _supports_encoder_capture_axes() -> bool:
    return "capture_axes" in EncoderCudaGraphConfig.__dataclass_fields__


def _deploy_config(*, encoder_graph: bool) -> str:
    stage0: dict[str, object] = {
        # Bound the shared-card profile so a second engine can load after the
        # first is closed, without changing any modality default.
        "kv_cache_memory_bytes": 2 * 1024**3,
        "max_model_len": 8192,
        "max_num_batched_tokens": 4096,
        "limit_mm_per_prompt": {"image": 2, "audio": 2, "video": 2},
        "media_io_kwargs": {"video": {"fps": 1, "num_frames": 2}},
        "default_sampling_params": {"temperature": 0.0, "max_tokens": 64},
    }
    if encoder_graph:
        stage0["compilation_config"] = {
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": [_ENCODER_TOKEN_BUDGET],
            "encoder_cudagraph_max_vision_items_per_batch": 2,
            "encoder_cudagraph_max_frames_per_batch": 2,
        }
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: stage0,
                1: {
                    "default_sampling_params.max_tokens": 1024,
                    "default_sampling_params.temperature": 0.0,
                },
            },
        },
    )


_GRAPH_DEPLOY = _deploy_config(encoder_graph=True)
_EAGER_DEPLOY = _deploy_config(encoder_graph=False)

# Single graph-engine entry: the eager comparison builds its own runner inside
# the test so the two engines are never resident at the same time.
test_params = [(_MODEL, _GRAPH_DEPLOY, {"trust_remote_code": True})]


def _synthetic_rgb(height: int, width: int, seed: int) -> Image.Image:
    """Deterministic non-uniform image so the encoder sees real patch content."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _small_image() -> Image.Image:
    return _synthetic_rgb(224, 224, seed=11)


def _oversized_image() -> Image.Image:
    return _synthetic_rgb(_OVERSIZED_IMAGE_PIXELS, _OVERSIZED_IMAGE_PIXELS, seed=13)


def _small_video() -> np.ndarray:
    return generate_synthetic_video(112, 112, 8)["np_array"]


def _image_question() -> str:
    return "Describe the colors in this image in one short sentence."


def _video_question() -> str:
    return "Describe what happens in this video in one short sentence."


def _oversized_vision_tokens() -> int:
    """Vision tokens for one 1024x1024 slice on the checkpoint's 14px patch."""
    return ((_OVERSIZED_IMAGE_PIXELS // 14) ** 2) // 4


def _final_text(outputs) -> str:
    text = None
    for stage_output in outputs:
        if getattr(stage_output, "final_output_type", None) == "text":
            text = stage_output.outputs[0].text
    assert text, "request produced no text output"
    return text


def _generate(omni_runner, *, images=None, videos=None) -> str:
    return _final_text(
        omni_runner.generate_multimodal(
            prompts=_image_question() if images is not None else _video_question(),
            images=images,
            videos=videos,
            modalities=["text"],
        )
    )


def _compilation_config_of(stage_config):
    """Resolved vLLM compilation config of one stage, across config generations."""
    for owner in (stage_config, getattr(stage_config, "engine_args", None)):
        if owner is None:
            continue
        value = getattr(owner, "compilation_config", None)
        if value is not None:
            return value
    return None


def _stage0_uses_encoder_graph(omni_runner) -> bool:
    """Whether the loaded engine actually carried the enabled compilation block.

    The flag makes the engine capture encoder graphs; if the deploy-config
    plumbing ever stops forwarding it, the parity test would compare two eager
    engines and quietly stop covering the feature. Asserting on the resolved
    config keeps that from happening.
    """
    for stage_config in omni_runner.omni.engine.stage_configs:
        compilation = _compilation_config_of(stage_config)
        if compilation is None:
            continue
        if isinstance(compilation, dict):
            if compilation.get("cudagraph_mm_encoder"):
                return True
        elif getattr(compilation, "cudagraph_mm_encoder", False):
            return True
    return False


def _close_runner(omni_runner) -> None:
    """Shut an ``OmniRunner`` down and wait until its card memory is released.

    The module-scoped fixture owns the graph engine, so the comparison test has
    to hand it back explicitly before a second engine can be loaded on the same
    device.
    """
    omni_runner.__exit__(None, None, None)
    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=_ENGINE_TEARDOWN_MEMORY_RATIO,
    )


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_and_video_requests_are_served(omni_runner, offline_client) -> None:
    """The graph-enabled profile serves in-budget image and video requests."""
    image_response = offline_client.send_omni_request(
        {"prompts": _image_question(), "images": _small_image(), "modalities": ["text"]}
    )
    assert image_response.success
    assert image_response.text_content

    video_response = offline_client.send_omni_request(
        {"prompts": _video_question(), "videos": _small_video(), "modalities": ["text"]}
    )
    assert video_response.success
    assert video_response.text_content


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_oversized_image_falls_back_without_failing(omni_runner, offline_client) -> None:
    """An image above the capture budget is served, not rejected or clamped."""
    assert _oversized_vision_tokens() > _ENCODER_TOKEN_BUDGET, (
        f"oversized fixture only reaches {_oversized_vision_tokens()} vision tokens per slice; "
        f"it must exceed the {_ENCODER_TOKEN_BUDGET}-token budget to exercise the eager fallback"
    )

    response = offline_client.send_omni_request(
        {"prompts": _image_question(), "images": _oversized_image(), "modalities": ["text"]}
    )
    assert response.success
    assert response.text_content


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
@pytest.mark.xfail(
    reason=(
        "Full-checkpoint graph replay does not reproduce the eager encoder output: on an L20 with "
        "vLLM 0.29.1rc1.dev197+gab35354c2 the graph engine answers the image prompt with a repeated "
        "token while the same request without the flag answers correctly. Graph capture itself "
        "succeeds (12 graphs over the 256-token budget, ~3 s). Remove this marker once the replay "
        "path is fixed; the failure output is the evidence for the divergence."
    ),
    strict=True,
)
def test_encoder_graph_matches_eager_encoder(omni_runner) -> None:
    """Replayed graph encoding returns the same greedy text as the eager encoder.

    Greedy decoding turns any encoder output difference into a token-string
    difference, so equality is a direct statement about the image and video
    encoder paths rather than the surrounding stages. The graph engine is closed
    first because the three stages share one card.

    Recorded result before this test was added: graph ``'validator\\n' * 32`` vs
    eager ``'The image is filled with a multitude of small, multicolored pixels
    creating a noisy, speckled appearance.'`` for the same 224x224 fixture. That
    is a replay-correctness gap, not an admission or capture failure, which is
    why the test is marked ``xfail(strict=True)`` instead of being skipped.
    """
    if not _supports_encoder_capture_axes():
        pytest.skip("installed vLLM does not advertise the capture_axes encoder protocol")

    assert _stage0_uses_encoder_graph(omni_runner), "graph profile did not reach Stage 0 compilation config"

    from tests.helpers.runtime import OmniRunner

    graph_image = _generate(omni_runner, images=_small_image())
    graph_video = _generate(omni_runner, videos=_small_video())

    _close_runner(omni_runner)

    with OmniRunner(_MODEL, deploy_config=_EAGER_DEPLOY, trust_remote_code=True) as eager_runner:
        assert not _stage0_uses_encoder_graph(eager_runner), "eager profile unexpectedly enabled encoder graphs"
        eager_image = _generate(eager_runner, images=_small_image())
        eager_video = _generate(eager_runner, videos=_small_video())

    assert graph_image == eager_image, f"image encoder graph diverged from eager: {graph_image!r} != {eager_image!r}"
    assert graph_video == eager_video, f"video encoder graph diverged from eager: {graph_video!r} != {eager_video!r}"
