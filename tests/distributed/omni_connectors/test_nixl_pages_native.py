# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.connection import Connection

import msgspec
import pytest
import torch
from torch.nn import functional as F

from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import KVPagePool, PageOffer, ReservedKVPages
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.parallel]

LAYER = "model.layers.0.attn"


def inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(123)
    return tuple(torch.randn(19, 2, 64, dtype=torch.float16, device=device, generator=generator) for _ in range(3))


def source_worker(wire: Connection) -> None:
    from tests.diffusion.diffusion_kv.test_paged_attention_adapter_gpu import _native_adapter
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import DiffusionPagedAttentionRow
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device("cuda:0")
    adapter, _, _ = _native_adapter()
    cache = adapter.layers[LAYER].kv_cache
    assert cache is not None
    cache.fill_(7)
    pool = KVPagePool({LAYER: cache}, "native-attention-fixture:fp16", adapter.vllm_config.cache_config.kv_cache_layout)
    connector = NixlConnector({"role": "sender", "host": "127.0.0.1", "zmq_port": 0})
    connector.register_page_pool(pool)
    q, k, v = inputs("cuda:0")
    batch = adapter.prepare_batch([DiffusionPagedAttentionRow("source", sequence_id=0, query_len=17, seq_len=17)])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), adapter.activate(batch):
        context = adapter.prepare_layer_context(LAYER, q[:17], k[:17], v[:17])
        backend = FlashAttentionImpl(num_heads=2, head_size=64, softmax_scale=64**-0.5, num_kv_heads=2)
        backend.forward_paged(context)
        ready = torch.cuda.Event()
        ready.record()
    offer = connector.export_pages("native-pages", pool, (3, 1), 17, expected_readers=2, ready=ready)
    wire.send_bytes(msgspec.msgpack.encode(offer))
    assert wire.poll(60) and wire.recv_bytes() == b"first-done"
    assert connector._pending["native-pages"].claims == {"negative"}
    assert pool.exports and pool.registrations
    wire.send_bytes(b"source-retained")
    assert wire.poll(60) and wire.recv_bytes() == b"second-done"
    assert not pool.exports and not connector._pending
    connector.unregister_page_pool(pool)
    connector.close()
    assert connector._closed
    wire.send_bytes(b"source-clean")
    wire.close()


def target_worker(wire: Connection) -> None:
    from tests.diffusion.diffusion_kv.test_paged_attention_adapter_gpu import _native_adapter
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import DiffusionPagedAttentionRow
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device("cuda:1")
    adapter, _, device = _native_adapter(num_rows=2)
    cache = adapter.layers[LAYER].kv_cache
    assert cache is not None
    cache.fill_(-3)
    torch.accelerator.synchronize(device)
    pool = KVPagePool({LAYER: cache}, "native-attention-fixture:fp16", adapter.vllm_config.cache_config.kv_cache_layout)
    connector = NixlConnector({"role": "receiver", "metadata_query_timeout_ms": 1000})
    connector.register_page_pool(pool)
    assert wire.poll(60)
    offer = msgspec.msgpack.decode(wire.recv_bytes(), type=PageOffer)
    peers = ("positive", "negative")
    for row, blocks in enumerate(((3, 1), (7, 5))):
        claimed = connector.claim_pages(
            "native-pages",
            offer.sender_host,
            offer.sender_zmq_port,
            generation=offer.generation,
            claim_id=peers[row],
            page_claim_ids=peers,
        )
        assert claimed is not None
        reservation = ReservedKVPages(peers[row], 7, pool, blocks)
        read_id = connector.read_into("native-pages", claimed, reservation)
        deadline = time.monotonic() + 30
        while not connector.poll_page_read(read_id):
            assert time.monotonic() < deadline, "Page DMA remains pinned for inspection"
            time.sleep(0.001)
        assert peers[row] in pool.reservations
        if row == 0:
            wire.send_bytes(b"first-done")
            assert wire.poll(60) and wire.recv_bytes() == b"source-retained"
    wire.send_bytes(b"second-done")
    assert wire.poll(60) and wire.recv_bytes() == b"source-clean"

    q, k, v = inputs("cuda:1")
    local_adapter, _, _ = _native_adapter(num_rows=2)
    local_cache = local_adapter.layers[LAYER].kv_cache
    assert local_cache is not None
    local_cache.fill_(7)
    prefix_batch = local_adapter.prepare_batch(
        [DiffusionPagedAttentionRow("reference", sequence_id=row, query_len=17, seq_len=17) for row in range(2)]
    )
    backend = FlashAttentionImpl(num_heads=2, head_size=64, softmax_scale=64**-0.5, num_kv_heads=2)
    with local_adapter.activate(prefix_batch):
        context = local_adapter.prepare_layer_context(
            LAYER,
            q[:17].repeat(2, 1, 1),
            k[:17].repeat(2, 1, 1),
            v[:17].repeat(2, 1, 1),
        )
        backend.forward_paged(context)
    affected = torch.tensor([3, 1, 7, 5], device=device)
    torch.testing.assert_close(
        cache.index_select(pool.block_axis, affected),
        local_cache.index_select(pool.block_axis, affected),
        rtol=0,
        atol=0,
    )
    untouched = torch.tensor([0, 2, 4, 6], device=device)
    assert torch.all(cache.index_select(pool.block_axis, untouched) == -3)

    # Consume the transferred pages through the real diffusion paged kernel.
    # Only the suffix is passed as K/V, so prefix recomputation cannot mask a bad READ.
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow("request", sequence_id=row, query_len=2, seq_len=19, kv_start_pos=17)
            for row in range(2)
        ]
    )
    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            LAYER, q[17:].repeat(2, 1, 1), k[17:].repeat(2, 1, 1), v[17:].repeat(2, 1, 1)
        )
        result = backend.forward_paged(context)
    reference = (
        F.scaled_dot_product_attention(
            q[17:].transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=64**-0.5,
        )
        .squeeze(0)
        .transpose(0, 1)
        .repeat(2, 1, 1)
    )
    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
    torch.accelerator.synchronize(device)
    for row, blocks in enumerate(((3, 1), (7, 5))):
        connector.retire_pages(ReservedKVPages(peers[row], 7, pool, blocks))
    connector.unregister_page_pool(pool)
    connector.close()
    assert connector._closed and not pool.registrations and not pool.reservations
    wire.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2, reason="requires two CUDA GPUs"
)
def test_two_gpu_native_scatter_cfg_and_paged_attention() -> None:
    from vllm.distributed.nixl_utils import NixlWrapper

    assert NixlWrapper is not None, "Native page validation requires NIXL"
    context = mp.get_context("spawn")
    source_wire, target_wire = context.Pipe()
    producer = context.Process(target=source_worker, args=(source_wire,))
    consumer = context.Process(target=target_worker, args=(target_wire,))
    producer.start()
    consumer.start()
    producer.join(90)
    consumer.join(90)
    assert not producer.is_alive(), f"Producer PID {producer.pid} remains running for inspection"
    assert not consumer.is_alive(), f"Consumer PID {consumer.pid} remains running for inspection"
    assert producer.exitcode == consumer.exitcode == 0
