# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.parallel import uaa_layout
from vllm_omni.diffusion.attention.parallel.uaa_layout import pad_pack_heads, unpack_unpad_heads
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _reference_pack(x, u, padded_h):
    b, s, h, d = x.shape
    x = F.pad(x, (0, 0, 0, padded_h - h)) if padded_h > h else x
    return x.reshape(b, s, u, padded_h // u, d).permute(2, 1, 0, 3, 4).contiguous().flatten(0, 1)


def _reference_unpack(x, u, s, h):
    _, b, local_h, d = x.shape
    out = x.reshape(u, s, b, local_h, d).permute(2, 1, 0, 3, 4).contiguous()
    return out.reshape(b, s, u * local_h, d)[:, :, :h].contiguous()


def _assert_bits(actual, expected, *, check_stride=True):
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert actual.shape == expected.shape
    if check_stride:
        assert actual.stride() == expected.stride()
    assert torch.equal(actual.contiguous().view(bits), expected.contiguous().view(bits))


@pytest.mark.cpu
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("h,padded_h", [(21, 24), (7, 8), (24, 24), (8, 8)])
def test_coordinate_mapping(b, h, padded_h):
    s, d, u = 3, 5, 2
    x = torch.arange(b * s * h * d, dtype=torch.float32).reshape(b, s, h, d)
    packed = pad_pack_heads(x, u, padded_h).reshape(u, s, b, padded_h // u, d)
    # Independent coordinate oracle catches B=1-only permutation mistakes.
    for rank in range(u):
        for seq in range(s):
            for batch in range(b):
                for local_head in range(padded_h // u):
                    head = rank * (padded_h // u) + local_head
                    expected = x[batch, seq, head] if head < h else torch.zeros(d)
                    assert torch.equal(packed[rank, seq, batch, local_head], expected)
    received = torch.arange(packed.numel(), dtype=x.dtype).reshape_as(packed)
    restored = unpack_unpad_heads(received.flatten(0, 1), u, s, h)
    for batch in range(b):
        for seq in range(s):
            for head in range(h):
                rank, local_head = divmod(head, padded_h // u)
                assert torch.equal(restored[batch, seq, head], received[rank, seq, batch, local_head])


@pytest.mark.cpu
@pytest.mark.parametrize("s,u", [(0, 2), (3, 1), (3, 2)])
def test_reference_empty_no_padding_and_grad(s, u):
    x = torch.randn(2, s, 8, 5, requires_grad=True)
    packed = pad_pack_heads(x, u, 8)
    restored = unpack_unpad_heads(packed, u, s, 8)
    assert torch.equal(restored, x)
    restored.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("s", [1, 3, 129])
@pytest.mark.parametrize("h,padded_h", [(21, 24), (7, 8), (24, 24), (8, 8)])
@pytest.mark.parametrize("layout", ["contiguous", "projection", "channel_slice"])
def test_cuda_bits(dtype, b, s, h, padded_h, layout):
    d = 120
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    shape = (b, s, h * (3 if layout == "projection" else 1), d * (2 if layout == "channel_slice" else 1))
    info = torch.iinfo(bits)
    storage = torch.randint(info.min, info.max, shape, device="cuda", dtype=bits).view(dtype)
    x = storage[:, :, :h, ::2] if layout == "channel_slice" else storage[:, :, :h]
    before = x.clone()
    packed = pad_pack_heads(x, 2, padded_h)
    _assert_bits(packed, _reference_pack(x, 2, padded_h))
    _assert_bits(unpack_unpad_heads(packed, 2, s, h), _reference_unpack(packed, 2, s, h))
    assert torch.equal(x.contiguous().view(bits), before.contiguous().view(bits))


@hardware_test(res={"cuda": "L4"})
def test_cuda_graph_compile_and_grad():
    x = torch.randn(2, 7, 21, 120, device="cuda", dtype=torch.bfloat16)
    expected = _reference_pack(x, 2, 24)
    compiled = torch.compile(pad_pack_heads, fullgraph=True)
    _assert_bits(compiled(x, 2, 24), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pad_pack_heads(x, 2, 24)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pad_pack_heads(x, 2, 24)
        restored = unpack_unpad_heads(out, 2, 7, 21)
    x.add_(1)
    graph.replay()
    _assert_bits(out, _reference_pack(x, 2, 24))
    _assert_bits(restored, x)
    x = x.float().requires_grad_()
    unpack_unpad_heads(pad_pack_heads(x, 2, 24), 2, 7, 21).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def _distributed_worker(rank, init_file):
    from vllm_omni.diffusion.attention.parallel.ulysses import (
        _ulysses_all_to_all_any_o,
        _ulysses_all_to_all_any_qkv,
    )

    current_omni_platform.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    group = dist.group.WORLD
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for b in (1, 2):
                for lens in ([3, 3], [3, 5], [0, 5], [0, 0]):
                    s = lens[rank]
                    for h, hp in ((21, 24), (7, 8), (24, 24)):
                        x = torch.arange(b * s * h * 120, device=f"cuda:{rank}").reshape(b, s, h, 120)
                        x = (x % 31 + rank * 32).to(dtype)
                        original_support = uaa_layout._can_fuse
                        outputs = []
                        for fused in (False, True):
                            uaa_layout._can_fuse = original_support if fused else lambda _: False
                            y, original_h = _ulysses_all_to_all_any_qkv(
                                group, x, seq_lens=lens, use_sync=False, padded_head_cnt=hp
                            )
                            # Distinguish head owners rather than relying only on an identity round trip.
                            z = _ulysses_all_to_all_any_o(
                                group, y + rank, seq_lens=lens, local_seq_len=s, orig_head_cnt=original_h, use_sync=True
                            )
                            outputs.append((y, z))
                        uaa_layout._can_fuse = original_support
                        for actual, expected in zip(outputs[1], outputs[0]):
                            _assert_bits(actual, expected)
                        owner = torch.arange(h, device=x.device) // (hp // 2)
                        # The coordinate oracle does not encode the native
                        # empty-slice strides; those are checked above.
                        _assert_bits(outputs[1][1], (x + owner[None, None, :, None]).to(dtype), check_stride=False)
                # Actual GQA attention surrounded by the same U=2 exchanges.
                lens = [5, 7]
                q = torch.randn(b, lens[rank], 21, 120, device=f"cuda:{rank}", dtype=dtype)
                k = torch.randn(b, lens[rank], 7, 120, device=f"cuda:{rank}", dtype=dtype)
                v = torch.randn_like(k)
                outputs = []
                original_support = uaa_layout._can_fuse
                for fused in (False, True):
                    uaa_layout._can_fuse = original_support if fused else lambda _: False
                    tensors = [
                        _ulysses_all_to_all_any_qkv(group, x, seq_lens=lens, use_sync=False, padded_head_cnt=hp)[0]
                        for x, hp in ((q, 24), (k, 8), (v, 8))
                    ]
                    a = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in tensors), enable_gqa=True)
                    outputs.append(
                        _ulysses_all_to_all_any_o(
                            group,
                            a.transpose(1, 2),
                            seq_lens=lens,
                            local_seq_len=lens[rank],
                            orig_head_cnt=21,
                            use_sync=False,
                        )
                    )
                uaa_layout._can_fuse = original_support
                _assert_bits(outputs[1], outputs[0])
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parallel
def test_nccl_uneven_shards_and_gqa_attention(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "init"),), nprocs=2, join=True)
