# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract checks for the native MOSS-VL-Realtime implementation.

These run without a checkpoint, without CUDA, and without Transformers. They pin
the structural facts the reference capture depends on: layer typing, the
image-pad position block, frame-to-token visibility expansion, packed vision
geometry, and the prefill/decode cache contract.
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.moss_vl_realtime.config import (
    MossVLConfig,
    MossVLTextConfig,
    MossVLVisionConfig,
)
from vllm_omni.model_executor.models.moss_vl_realtime.modeling import MossVLNativeModel, VisionTokenInfo
from vllm_omni.model_executor.models.moss_vl_realtime.vision import MossVLVisionTower

# The published checkpoint uses 151655; the tiny config keeps ids inside a small vocab.
TINY_IMAGE_PAD = 3


def tiny_vision() -> MossVLVisionConfig:
    return MossVLVisionConfig(
        depth=4,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_position_embeddings=64,
        out_hidden_size=32,
        deepstack_visual_indexes=(1, 2),
    )


def tiny_config() -> MossVLConfig:
    text = MossVLTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        cross_attention_layers=(1, 3),
        vocab_size=64,
    )
    return MossVLConfig(text=text, vision=tiny_vision(), image_token_id=TINY_IMAGE_PAD)


def test_layer_typing_matches_checkpoint_split() -> None:
    model = MossVLNativeModel(tiny_config())
    cross = [idx for idx, layer in enumerate(model.model.language_model.layers) if hasattr(layer, "cross_attn")]
    self_attn = [idx for idx, layer in enumerate(model.model.language_model.layers) if hasattr(layer, "self_attn")]
    assert cross == [1, 3]
    assert self_attn == [0, 2]


def test_vision_tower_packed_geometry() -> None:
    tower = MossVLVisionTower(tiny_vision()).eval()
    with torch.no_grad():
        image = torch.randn(16, 3 * 1 * 16 * 16)
        assert tower(image, torch.tensor([[1, 4, 4]])).shape == (4, 32)
        video = torch.randn(32, 3 * 1 * 16 * 16)
        assert tower(video, torch.tensor([[2, 4, 4]])).shape == (8, 32)


def test_image_pad_consumes_a_position_block() -> None:
    model = MossVLNativeModel(tiny_config())
    input_ids = torch.tensor([[10, 11, TINY_IMAGE_PAD, 12, 13]])
    positions = model.compute_text_position_ids(input_ids)
    assert positions.shape == (3, 1, 5)
    assert positions[0, 0].tolist() == [0, 1, 2, 2, 3]
    assert positions[1, 0].tolist() == positions[2, 0].tolist() == positions[0, 0].tolist()


def test_visibility_expansion_repeats_per_frame() -> None:
    model = MossVLNativeModel(tiny_config())
    model.vision_token_info = [VisionTokenInfo(grid_h=4, grid_w=4, num_frames=2, start=0, vision_tokens_per_frame=4)]
    frame_mask = torch.tensor([[[[True, True], [False, True], [False, False]]]])
    expanded = model.expand_cross_attention_mask(frame_mask, torch.bfloat16)
    assert expanded.shape == (1, 1, 3, 10)
    negative = torch.finfo(torch.bfloat16).min
    assert expanded[0, 0, 0].tolist() == [negative] * 10
    assert expanded[0, 0, 1].tolist() == [0.0] * 5 + [negative] * 5
    assert expanded[0, 0, 2].tolist() == [0.0] * 10


def test_prefill_then_decode_grows_text_cache_only() -> None:
    config = tiny_config()
    model = MossVLNativeModel(config).eval()
    input_ids = torch.tensor([[10, 11, TINY_IMAGE_PAD, 12]])
    pixel_values = torch.randn(16, 3 * 1 * 16 * 16)
    grid_thw = torch.tensor([[1, 4, 4]])
    frame_mask = torch.tensor([[[[False], [False], [True], [True]]]])
    with torch.no_grad():
        logits = model(input_ids, model.compute_text_position_ids(input_ids), 0, pixel_values, grid_thw, frame_mask)
    assert logits.shape == (1, 4, 64)
    cross_layer = config.text.cross_attention_layers[0]
    vision_length = model.vision_cache.keys[cross_layer].shape[2]
    assert vision_length == 5
    assert model.text_cache.length(0) == 4

    with torch.no_grad():
        step = model(torch.tensor([[12]]), model.decode_position_ids(offset=4), 4)
    assert step.shape == (1, 1, 64)
    assert model.text_cache.length(0) == 5
    assert model.vision_cache.keys[cross_layer].shape[2] == vision_length
    assert model.vision_cache.keys[0] is None


def test_text_only_prefill_skips_cross_attention_layers() -> None:
    model = MossVLNativeModel(tiny_config()).eval()
    input_ids = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        model(input_ids, model.compute_text_position_ids(input_ids), 0)
    for idx in tiny_config().text.cross_attention_layers:
        assert model.vision_cache.is_empty(idx)
