# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint acceptance and input-preparation contract checks.

These run without a checkpoint on disk beyond a few small JSON files, and
without Transformers: the processor is a stub that records what it was asked to
render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.moss_vl_realtime.registry import inspect_checkpoint

ASSET_SCRIPTS = Path(__file__).resolve().parents[3] / "assets" / "moss_vl_realtime"


def write_checkpoint(root: Path, *, model_type: str = "moss_vl", architecture: str = "MossVLForConditionalGeneration",
                     gates: bool = True, vision_fields: bool = True, index: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
        },
    }
    if not vision_fields:
        config["vision_config"].pop("spatial_merge_size")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if index:
        weight_map = {"model.language_model.embed_tokens.weight": "model-00001-of-00005.safetensors"}
        if gates:
            weight_map["model.language_model.layers.2.cross_attn_attn_gate"] = "model-00001-of-00005.safetensors"
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return root


def test_published_layout_is_accepted(tmp_path: Path) -> None:
    verdict = inspect_checkpoint(write_checkpoint(tmp_path / "ok"))
    assert verdict.supported
    assert verdict.reasons == []


def test_converted_checkpoint_is_refused(tmp_path: Path) -> None:
    verdict = inspect_checkpoint(write_checkpoint(tmp_path / "converted", gates=False))
    assert not verdict.supported
    assert any("SGLang" in reason for reason in verdict.reasons)


def test_wrong_architecture_is_refused(tmp_path: Path) -> None:
    verdict = inspect_checkpoint(write_checkpoint(tmp_path / "other", model_type="qwen3", architecture="Qwen3ForCausalLM"))
    assert not verdict.supported
    assert len(verdict.reasons) == 2


def test_unsharded_checkpoint_is_refused(tmp_path: Path) -> None:
    verdict = inspect_checkpoint(write_checkpoint(tmp_path / "single", index=False))
    assert not verdict.supported
    assert any("index" in reason for reason in verdict.reasons)


def test_corrupt_config_is_refused(tmp_path: Path) -> None:
    verdict = inspect_checkpoint(write_checkpoint(tmp_path / "vision", vision_fields=False))
    assert not verdict.supported
    assert any("spatial_merge_size" in reason for reason in verdict.reasons)


def test_missing_config_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(tmp_path / "empty")


class StubProcessor:
    """Minimal stand-in for the checkpoint processor: records the rendered text."""

    image_placeholder = "<|image|>"
    video_placeholder = "<|video|>"
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.images: object = None

    def apply_chat_template(self, messages: list[dict], tokenize: bool, add_generation_prompt: bool) -> str:
        self.messages = messages
        return "<rendered>"

    def __call__(self, text: str, images: object, videos: object, return_tensors: str, padding: bool) -> dict:
        self.images = images
        return {"text": text}


@pytest.fixture()
def prepare_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_inputs", ASSET_SCRIPTS / "prepare_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_content_order_and_markers_are_preserved(prepare_module) -> None:
    processor = StubProcessor()
    content = [
        {"type": "text", "text": "Describe the image."},
        {"type": "image", "image": "red.ppm"},
        {"type": "image", "image": "blue.ppm"},
    ]
    assert prepare_module.flatten_content(processor, content) == "Describe the image.<|image|><|image|>"


def test_prompt_placeholders_are_stripped(prepare_module) -> None:
    processor = StubProcessor()
    assert prepare_module.sanitize_prompt_text(processor, "\n<|image|>hello<|image_pad|>") == "hello"


def test_images_travel_with_the_prompt(prepare_module) -> None:
    processor = StubProcessor()
    inputs, text = prepare_module.prepare(processor, "Look.", ["a.ppm", "b.ppm"], [])
    assert text == "<rendered>"
    assert processor.images == ["a.ppm", "b.ppm"]
    assert inputs["text"] == "<rendered>"
    assert processor.messages[0]["content"] == "Look.<|image|><|image|>"


def test_fixture_case_maps_to_prompt_and_frames(prepare_module) -> None:
    cases = json.loads((ASSET_SCRIPTS / "cases.json").read_text(encoding="utf-8"))
    manifest = json.loads((ASSET_SCRIPTS / "manifest.json").read_text(encoding="utf-8"))
    case = next(item for item in cases["cases"] if item["id"] == "short_video")
    prompt, images = prepare_module.case_to_prompt(case, manifest)
    assert prompt == "Describe changes in color."
    assert [Path(path).name for path in images] == ["red.ppm", "blue.ppm"]
