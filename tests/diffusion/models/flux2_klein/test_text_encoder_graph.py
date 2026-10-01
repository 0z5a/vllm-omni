# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from vllm_omni.diffusion.models.flux2_klein.quantization import Flux2KleinTextEncoderGraph

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def encoder():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=1024,
    )
    return Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()


def reference(encoder, ids, mask):
    output = encoder.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    return torch.stack([output.hidden_states[k] for k in (0, 1, 2)], dim=1)


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(1, 512), (1, 32), (2, 512)])
def test_cpu_and_unsupported_shape_use_intermediate_states_without_lm_head(encoder, shape):
    def reject_lm_head(*args, **kwargs):
        raise AssertionError("hidden-state extraction must not run the LM head")

    encoder.lm_head.forward = reject_lm_head
    ids = torch.randint(1, 32, shape)
    mask = torch.ones_like(ids)
    graph = Flux2KleinTextEncoderGraph(encoder, (0, 1, 2))
    with torch.inference_mode():
        torch.testing.assert_close(graph(ids, mask), reference(encoder, ids, mask), atol=0, rtol=0)
    assert graph.graph is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_updates_ids_and_masks_and_orders_concurrent_streams(encoder):
    encoder.cuda()
    graph = Flux2KleinTextEncoderGraph(encoder, (0, 1, 2))
    ids = [torch.randint(1, 32, (1, 512), device="cuda") for _ in range(4)]
    masks = [torch.ones_like(value) for value in ids]
    for index, mask in enumerate(masks):
        mask[:, 400 + 20 * index :] = 0
    with torch.inference_mode():
        expected = [reference(encoder, value, mask) for value, mask in zip(ids, masks)]
        first = graph(ids[0], masks[0])
    torch.accelerator.synchronize()
    torch.testing.assert_close(first, expected[0], rtol=0.01, atol=0.01)
    captured = graph.graph
    assert captured is not None

    streams = [torch.cuda.Stream() for _ in ids]

    def replay(index):
        with torch.cuda.stream(streams[index]), torch.inference_mode():
            # Delay this stream to expose reuse before its previous replay completes.
            torch.cuda._sleep(1_000_000)
            return graph(ids[index], masks[index])

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(replay, range(4)))
    torch.accelerator.synchronize()
    for result, value in zip(actual, expected):
        torch.testing.assert_close(result, value, rtol=0.01, atol=0.01)
    assert graph.graph is captured
