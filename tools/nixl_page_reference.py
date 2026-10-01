# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark reference: ordinary NIXL allocation/copy into the same native pages."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence

import msgspec
import torch

from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector
from vllm_omni.distributed.omni_connectors.connectors.nixl_kv_connector import (
    OmniNixlKVConnector as NativeKVConnector,
)
from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import (
    KVPagePool,
    PageOffer,
    PageReadyEvent,
    ReservedKVPages,
)


class ReferenceOffer(PageOffer, frozen=True):
    ordinary_metadata: bytes = b""


class ReferenceNixlConnector(NixlConnector):
    page_offer_type = ReferenceOffer

    def export_pages(
        self,
        key: str,
        pool: KVPagePool,
        block_ids: Sequence[int],
        num_tokens: int,
        *,
        expected_readers: int,
        ready: PageReadyEvent,
        generation: str | None = None,
    ) -> PageOffer:
        ready.synchronize()
        indexes = torch.tensor(block_ids, dtype=torch.long, device=next(iter(pool.caches.values())).device)
        packed = [tensor.index_select(pool.block_axis, indexes) for tensor in pool.caches.values()]
        packed_ready = torch.cuda.Event()
        packed_ready.record()
        packed_ready.synchronize()
        with self._state_lock:
            ok, _, metadata = self.put("0", "1", key, packed)
            assert ok and metadata is not None
            generation = generation or uuid.uuid4().hex
            metadata["generation"] = generation
            pending = self._pending[key]
            pending.generation, pending.page_pool = generation, pool
            pending.deadline, pending.expected_readers = math.inf, expected_readers
            pool.exports.add(generation)
            offer = ReferenceOffer(
                1,
                "pages",
                generation,
                pool.epoch,
                pool.geometry,
                num_tokens,
                len(block_ids),
                expected_readers,
                pool.regions(block_ids),
                self._agent.get_agent_metadata(),
                str(self.host),
                int(self._zmq_port),
                ordinary_metadata=msgspec.msgpack.encode(metadata),
            )
            self._published[key] = msgspec.structs.asdict(offer)
        return offer

    def read_into(self, key: str, offer: PageOffer, target: ReservedKVPages) -> str:
        assert isinstance(offer, ReferenceOffer)
        read_id, _ = self._reserve_page_read(key, offer, target)
        metadata = msgspec.msgpack.decode(offer.ordinary_metadata)
        # The complete CFG lease was claimed above. Keep it until GPU scatter
        # finishes; the ordinary legacy get must not ACK the packed source early.
        for field in ("generation", "sender_host", "sender_zmq_port"):
            metadata.pop(field)
        received = self.get("0", "1", key, metadata)
        assert received is not None
        tensors, _ = received
        assert isinstance(tensors, list)
        indexes = torch.tensor(
            target.block_ids, dtype=torch.long, device=next(iter(target.pool.caches.values())).device
        )
        for tensor, cache in zip(tensors, target.pool.caches.values(), strict=True):
            assert isinstance(tensor, torch.Tensor)
            cache.index_copy_(target.pool.block_axis, indexes, tensor)
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()
        self._page_reads[read_id].dma_done = True
        return read_id

    def poll_page_read(self, read_id: str) -> bool:
        with self._state_lock:
            read = self._page_reads[read_id]
            if not read.dma_done:
                raise RuntimeError("Ordinary NIXL scatter did not finish; native pages remain pinned")
            if not self._notify_transfer_done(read.key, msgspec.structs.asdict(read.offer)):
                return False
            read.target.pool.reads.remove(read_id)
            del self._page_reads[read_id]
        return True


class OmniNixlKVConnector(NativeKVConnector):
    transport_type = ReferenceNixlConnector
