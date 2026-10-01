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
