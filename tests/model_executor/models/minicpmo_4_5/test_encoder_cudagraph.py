# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field

import pytest
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.whisper.modeling_whisper import WhisperConfig
from vllm.config import CompilationConfig, MultiModalConfig, ParallelConfig
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import (
    _AXIS_KEY,
    _ceiling,
    _MiniCPMO45EncoderCudaGraphMixin,
    _NoEncoderCudaGraph,
    _select_encoder_cudagraph_mixin,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_serving_wrapper_exposes_concrete_thinker_protocol():
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    thinker = _EncoderModel().eval()
    thinker.multimodal_config = MultiModalConfig(media_io_kwargs={"video": {"num_frames": 2}})
    assert not supports_encoder_cudagraph(model)
    bind_minicpmo_encoder_cudagraph(model, thinker)
    assert supports_encoder_cudagraph(model)
    assert model.get_encoder_cudagraph_config() == thinker.get_encoder_cudagraph_config()


def test_unsupported_stage_does_not_advertise_encoder_protocol():
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph

    model = torch.nn.Module()
    bind_minicpmo_encoder_cudagraph(model, torch.nn.Module())
    assert not supports_encoder_cudagraph(model)


@dataclass
class _EncoderConfig:
    query_num: int = 4
    hidden_size: int = 16
    vision_batch_size: int = 16
    audio_chunk_length: int = 0
    audio_pool_step: int = 2


@dataclass
class _EncoderTestModelConfig:
    multimodal_config: MultiModalConfig | None = None
    max_model_len: int = 32


@dataclass
class _EncoderTestSchedulerConfig:
    max_num_batched_tokens: int = 32


@dataclass
class _EncoderTestVllmConfig:
    """Only manager-facing state; not a substitute for serving initialization."""

    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    model_config: _EncoderTestModelConfig = field(default_factory=_EncoderTestModelConfig)
    scheduler_config: _EncoderTestSchedulerConfig = field(default_factory=_EncoderTestSchedulerConfig)


@pytest.mark.parametrize("value,expected", [(1, 1), (3, 4), (8, 8), (9, 9)])
def test_tier_selection_preserves_out_of_range_inputs(value: int, expected: int) -> None:
    assert _ceiling(value, (1, 2, 4, 8)) == expected


def test_non_cuda_platform_does_not_create_encoder_manager(monkeypatch) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    mixin = _select_encoder_cudagraph_mixin()
    assert mixin is _NoEncoderCudaGraph

    class NativeEncoderModel(torch.nn.Module, mixin):
        pass

    model = NativeEncoderModel()
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = model
    runner.compilation_config = CompilationConfig(cudagraph_mm_encoder=True)
    runner.supports_mm_inputs = True
    assert GPUModelRunner._create_encoder_cudagraph_manager(runner) is None


@pytest.mark.parametrize("limit,expected", [(128, (16, 16)), (8, (8, 8))])
def test_default_budget_range_is_bounded(limit: int, expected: tuple[int, int]) -> None:
    model = _EncoderModel()
    config = _EncoderTestVllmConfig(
        scheduler_config=_EncoderTestSchedulerConfig(max_num_batched_tokens=limit),
        model_config=_EncoderTestModelConfig(max_model_len=limit),
    )
    assert model.get_encoder_cudagraph_budget_range(config) == expected


@pytest.mark.parametrize("budget,expected", [(8, (1, 2)), (16, (1, 2, 4)), (64, (1, 2, 4, 8, 16))])
def test_slice_capture_tiers_follow_output_budget(budget, expected) -> None:
    model = _EncoderModel().eval()
    model.vllm_config = _EncoderTestVllmConfig(
        compilation_config=CompilationConfig(encoder_cudagraph_token_budgets=[budget])
    )
    assert model._encoder_slice_caps() == expected


def test_duplex_audio_bypasses_offline_graph_protocol_when_enabled(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

    torch.manual_seed(42)
    model = _EncoderModel().eval()
    model.vllm_config = _EncoderTestVllmConfig(compilation_config=CompilationConfig(cudagraph_mm_encoder=True))
    model.audio_past_key_values = None
    data = {"audio_features": torch.randn(1, 16, 16), "audio_feature_lens": [torch.tensor([16])]}
    with torch.no_grad():
        expected = model.get_audio_embedding_streaming(
            data, use_extra_context=True, prefix_extra_frames=0, suffix_extra_frames=2
        )
    model.audio_past_key_values = None

    def unexpected_protocol(*args, **kwargs):
        pytest.fail("duplex must use the stateful streaming encoder, not the offline graph protocol")

    monkeypatch.setattr(model, "encoder_cudagraph_forward", unexpected_protocol)
    monkeypatch.setattr(model, "encoder_eager_forward", unexpected_protocol)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = model
    runtime.thinker = model
    runtime.device = "cpu"
    with torch.no_grad():
        actual = runtime._stage_audio_embeddings(data)
    torch.testing.assert_close(actual, torch.cat(expected[0]))
    assert model.audio_past_key_values is not None


@pytest.mark.parametrize("modality", ["image", "video"])
def test_empty_encoder_shard_has_no_items(modality: str) -> None:
    model = _EncoderModel().eval()
    prefix = "video_" if modality == "video" else ""
    kwargs = {prefix + "pixel_values": [[torch.randn(3, 2, 12)]], prefix + "tgt_sizes": [torch.tensor([[2, 3]])]}
    selected = model.select_encoder_cudagraph_items(kwargs, [])
    assert model.get_encoder_cudagraph_item_specs(selected) == []


@pytest.mark.parametrize("grids", [((2, 3), (1, 4)), ((3, 2), (2, 3))])
def test_vision_capture_metadata_matches_eager(grids, monkeypatch) -> None:
    torch.manual_seed(42)
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    config._attn_implementation = "eager"
    vision = SiglipVisionTransformer(config).eval()
    resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, kv_dim=16).eval()
    sizes = torch.tensor(grids, dtype=torch.int32)
    patches = int(sizes.prod(-1).max())
    pixels = torch.randn(len(grids), 3, 2, patches * 2)
    mask = (torch.arange(patches)[None, :] < sizes.prod(-1)[:, None]).unsqueeze(1)

    with torch.no_grad():
        expected = resampler(vision(pixels, patch_attention_mask=mask, tgt_sizes=sizes).last_hidden_state, sizes)
        position_ids = vision.embeddings._create_position_ids(mask, sizes, device=torch.device("cpu"))
        attention_mask = _prepare_4d_attention_mask(mask.flatten(1), pixels.dtype) if not mask.all() else None
        positions, padding_mask = resampler.prepare_metadata(sizes, device=pixels.device, dtype=pixels.dtype)

        def unexpected_metadata(*args, **kwargs):
            pytest.fail("layout metadata must not be recomputed in the captured forward")

        monkeypatch.setattr(vision.embeddings, "_create_position_ids", unexpected_metadata)
        monkeypatch.setattr(resampler, "prepare_metadata", unexpected_metadata)
        hidden = vision(
            pixels,
            patch_attention_mask=mask,
            tgt_sizes=sizes,
            position_ids=position_ids,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state
        actual = resampler(hidden, pos_embed=positions, key_padding_mask=padding_mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _EncoderModel(MiniCPMO45OmniLLMForConditionalGeneration, _MiniCPMO45EncoderCudaGraphMixin):
    """Use production encoder entry points without constructing the language model."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = _EncoderConfig()
        vision_config = SiglipVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=8, patch_size=2
        )
        vision_config._attn_implementation = "eager"
        self.vpm = SiglipVisionTransformer(vision_config).eval()
        self.resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, kv_dim=16).eval()
        audio_config = WhisperConfig(
            num_mel_bins=16,
            d_model=16,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            max_source_positions=100,
            dropout=0,
            attention_dropout=0,
        )
        audio_config._attn_implementation = "eager"
        self.apm = MiniCPMWhisperEncoder(audio_config).eval()
        self.audio_projection_layer = torch.nn.Identity()
        self.audio_avg_pooler = torch.nn.AvgPool1d(2, stride=2)
        self.audio_encoder_layer = -1


@pytest.mark.parametrize("modality", ["image", "video"])
def test_protocol_buffers_match_encoder_entry_point(modality: str) -> None:
    torch.manual_seed(42)
    model = _EncoderModel().eval()
    prefix = "video_" if modality == "video" else ""
    kwargs = {
        prefix + "pixel_values": [[torch.randn(3, 2, 12), torch.randn(3, 2, 8)], [torch.randn(3, 2, 12)]],
        prefix + "tgt_sizes": [torch.tensor([[2, 3], [1, 4]]), torch.tensor([[3, 2]])],
    }
    selected = model.select_encoder_cudagraph_items(kwargs, [0, 1])
    axes = selected.pop(_AXIS_KEY)
    assert axes[0][1] == 4
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        128, 2, 2, torch.device("cpu"), torch.float32, "default", axes
    )
    replay = model.prepare_encoder_cudagraph_replay_buffers(selected, 2, 2)
    for key, buffer in capture.values.items():
        buffer.zero_()
        source = replay.values[key]
        buffer[: source.shape[0]].copy_(source)
    with torch.no_grad():
        expected = model.get_multimodal_embeddings(**kwargs)
        output = model.encoder_cudagraph_forward(capture.values)
    specs = model.get_encoder_cudagraph_item_specs(kwargs)
    dest = {}
    model.postprocess_encoder_output(
        {"default": output}, [0, 1], [spec.output_tokens for spec in specs], dest, batch_mm_kwargs=selected
    )
    assert len(dest) == 2
    for index, reference in enumerate(expected):
        torch.testing.assert_close(dest[index], reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_audio_is_not_advertised_for_padded_graph_capture() -> None:
    model = _EncoderModel().eval()
    model.multimodal_config = MultiModalConfig()
    assert "audio" not in model.get_encoder_cudagraph_config().modalities


def test_resampler_metadata_extends_cache_before_forward(monkeypatch) -> None:
    torch.manual_seed(42)
    resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, max_size=(2, 2)).eval()
    sizes = torch.tensor([[3, 2], [1, 4]])
    hidden = torch.randn(2, 6, 16)
    with torch.no_grad():
        expected = resampler(hidden, sizes)
        positions, mask = resampler.prepare_metadata(sizes, device=hidden.device, dtype=hidden.dtype)
        monkeypatch.setattr(resampler, "_adjust_pos_cache", lambda *args, **kwargs: pytest.fail("graph grew cache"))
        actual = resampler(hidden, pos_embed=positions, key_padding_mask=mask)
    assert tuple(resampler.max_size) == (3, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
