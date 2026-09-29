# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.anima.fp8_cuda_graph import install_anima_fp8_cuda_graph


def test_empty_negative_conditioning_cache_respects_prompt_and_mode():
    class Pipeline:
        def __init__(self):
            self.text_encoder = SimpleNamespace(training=False)
            self.text_conditioner = SimpleNamespace(training=False)
            self.qwen_calls = 0
            self.condition_calls = 0

        def diffuse(self, *args, **kwargs):
            return None

        def _get_qwen_prompt_embeds(self, prompt, max_sequence_length, device, dtype):
            self.qwen_calls += 1
            return torch.tensor([self.qwen_calls], dtype=dtype), torch.ones(1)

        def condition_prompt_embeds(
            self,
            qwen_prompt_embeds,
            qwen_attention_mask,
            t5_input_ids,
            t5_attention_mask,
            device=None,
            conditioning_dtype=None,
            output_dtype=None,
        ):
            self.condition_calls += 1
            return qwen_prompt_embeds * 2

    pipeline = Pipeline()
    install_anima_fp8_cuda_graph(pipeline)
    options = {"max_sequence_length": 512, "device": torch.device("cpu"), "dtype": torch.bfloat16}
    blank1 = pipeline._get_qwen_prompt_embeds(prompt=[""], **options)
    blank2 = pipeline._get_qwen_prompt_embeds(prompt=[""], **options)
    assert blank1 is blank2
    args = (blank1[0], blank1[1], torch.ones(1), torch.ones(1))
    assert pipeline.condition_prompt_embeds(*args) is pipeline.condition_prompt_embeds(*args)
    assert (pipeline.qwen_calls, pipeline.condition_calls) == (1, 1)

    other = pipeline._get_qwen_prompt_embeds(prompt=["different"], **options)
    pipeline.condition_prompt_embeds(other[0], other[1], torch.ones(1), torch.ones(1))
    assert (pipeline.qwen_calls, pipeline.condition_calls) == (2, 2)

    pipeline.text_encoder.training = True
    pipeline._get_qwen_prompt_embeds(prompt=[""], **options)
    assert pipeline.qwen_calls == 3

    assert (
        pipeline.diffuse(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 16, 1, 16, 16),
            torch.zeros(1, 1, 128, 128),
            torch.ones(1),
            True,
            4.0,
        )
        is None
    )


def test_explicit_disable_uses_torch_path(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ANIMA_FP8_CUDA_GRAPH", "0")
    expected = object()
    pipeline = SimpleNamespace(
        diffuse=lambda *args: expected,
        _get_qwen_prompt_embeds=lambda *args: None,
        condition_prompt_embeds=lambda *args: None,
    )
    install_anima_fp8_cuda_graph(pipeline)
    assert pipeline.diffuse(None, None, None, None, None, True, 4.0) is expected


def test_capture_failure_restores_forwards_and_falls_back(monkeypatch):
    from contextlib import nullcontext

    class Tensor:
        is_cuda = True
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        def __init__(self, shape):
            self.shape = shape

        def copy_(self, other):
            return self

        def __getitem__(self, item):
            return self

        def expand(self, *args):
            return self

        def to(self, *args):
            return self

        def __truediv__(self, other):
            return self

        def data_ptr(self):
            return id(self)

    class Projection:
        def forward(self, x):
            return Tensor(x.shape)

    projections = [Projection(), Projection()]
    shared = [Projection() for _ in range(8)]
    first = SimpleNamespace(
        norm1=shared[4],
        attn1=shared[5],
        norm2=shared[6],
        attn2=SimpleNamespace(to_q=shared[7], norm_q=Projection(), to_k=projections[0], to_v=projections[1]),
    )
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        if len(calls) == 6:  # Fail after the first graph was captured successfully.
            raise RuntimeError("capture unsupported")
        return (Tensor((1,)),)

    transformer = SimpleNamespace(
        dtype=torch.bfloat16,
        transformer_blocks=[first],
        forward=forward,
        patch_embed=shared[0],
        rope=shared[1],
        time_embed=shared[2],
        learnable_pos_embed=shared[3],
    )
    modules = projections + shared + [first.attn2.norm_q]
    originals = [module.forward for module in modules]
    eager_calls = []
    expected = object()

    def eager(*args):
        assert transformer.forward is forward
        assert [module.forward for module in modules] == originals
        eager_calls.append(args)
        return expected

    pipeline = SimpleNamespace(
        transformer=transformer,
        diffuse=eager,
        _get_qwen_prompt_embeds=lambda *args: None,
        condition_prompt_embeds=lambda *args: None,
    )
    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch, "empty_like", lambda x, **kwargs: Tensor(x.shape))
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    install_anima_fp8_cuda_graph(pipeline)
    args = (
        Tensor((1, 512, 1024)),
        Tensor((1, 512, 1024)),
        Tensor((1, 16, 1, 128, 128)),
        Tensor((1, 1, 128, 128)),
        Tensor((50,)),
        True,
        4.0,
    )
    assert pipeline.diffuse(*args) is expected
    assert pipeline.diffuse(*args) is expected
    assert len(eager_calls) == 2
    assert len(calls) == 6  # No new capture attempt after the failure.
