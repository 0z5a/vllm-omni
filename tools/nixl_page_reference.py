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
    PageGeometry,
    PageOffer,
    PageReadyEvent,
    ReservedKVPages,
)


class ReferenceOffer(PageOffer, frozen=True):
    ordinary_metadata: bytes = b""


class ReferenceNixlConnector(NixlConnector):
    def __init__(self, config: dict[str, object]):
        super().__init__(config)
        self._reference_reads: dict[str, tuple[str, ReferenceOffer, ReservedKVPages]] = {}

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

    def claim_pages(
        self,
        key: str,
        host: str,
        port: int,
        *,
        generation: str,
        claim_id: str,
        page_claim_ids: tuple[str, ...],
        geometry: PageGeometry,
    ) -> PageOffer | None:
        self.update_sender_info(host, port)
        metadata = self._query_metadata_at(
            key,
            host,
            port,
            generation=generation,
            claim_id=claim_id,
            page_claim_ids=page_claim_ids,
            geometry=geometry,
        )
        if metadata is None:
            return None
        offer = msgspec.convert(metadata, type=ReferenceOffer)
        with self._state_lock:
            self._page_claims[(key, offer.generation, offer.pool_epoch, offer.claim_id)] = offer
        return offer

    def read_into(self, key: str, offer: PageOffer, target: ReservedKVPages) -> str:
        assert isinstance(offer, ReferenceOffer)
        target.pool.validate_offer(offer, target.block_ids)
        claim = (key, offer.generation, offer.pool_epoch, offer.claim_id)
        assert self._page_claims.pop(claim) == offer
        metadata = msgspec.msgpack.decode(offer.ordinary_metadata)
        # The complete CFG lease was claimed above. Keep it until GPU scatter
        # finishes; the ordinary legacy get must not ACK the packed source early.
        for field in ("generation", "sender_host", "sender_zmq_port"):
            metadata.pop(field)
        target.pool.reservations[target.request_id] = (target.allocation_generation, target.block_ids)
        read_id = uuid.uuid4().hex
        target.pool.reads.add(read_id)
        self._reference_reads[read_id] = (key, offer, target)
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
        return read_id

    def poll_page_read(self, read_id: str) -> bool:
        key, offer, target = self._reference_reads[read_id]
        if not self._notify_transfer_done(key, msgspec.structs.asdict(offer)):
            return False
        target.pool.reads.remove(read_id)
        del self._reference_reads[read_id]
        return True

    def cancel_page_claim(self, key: str, offer: PageOffer) -> bool:
        if any(current == offer for _, current, _ in self._reference_reads.values()):
            raise RuntimeError("Cannot cancel a submitted ordinary NIXL READ")
        return super().cancel_page_claim(key, offer)


class OmniNixlKVConnector(NativeKVConnector):
    transport_type = ReferenceNixlConnector
