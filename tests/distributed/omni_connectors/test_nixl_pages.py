# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ctypes
import importlib
from dataclasses import replace

import msgspec
import pytest
import torch

from tests.distributed.omni_connectors.test_nixl_connector import (
    _FakeNixlAgent,
    nixl_connector_cls,  # noqa: F401
)
from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector
from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import (
    KVPagePool,
    PageOffer,
    Region,
    ReservedKVPages,
)

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


class Ready:
    def __init__(self):
        self.calls = 0

    def synchronize(self) -> None:
        self.calls += 1


class PageAgent(_FakeNixlAgent):
    def __init__(self, name: str, config: object = None):
        super().__init__(name, config)
        self.dlists: dict[int, tuple[Region, ...]] = {}
        self.handles: dict[int, tuple[int, int]] = {}
        self.states: dict[int, str] = {}
        self.remotes: set[str] = set()
        self.registrations = 0
        self.next_id = 0

    def get_reg_descs(self, regions: list[tuple[int, int, int, str]], memory_type: str) -> object:
        assert all(len(region) == 4 for region in regions)
        return super().get_reg_descs(regions, memory_type)

    def register_memory(self, descs: object, backends: object = None) -> None:
        super().register_memory(descs, backends)
        self.registrations += 1

    def add_remote_agent(self, metadata: bytes) -> str:
        name = metadata.decode()
        assert name not in self.remotes
        self.remotes.add(name)
        return name

    def remove_remote_agent(self, name: str) -> None:
        self.remotes.remove(name)

    def get_xfer_descs(self, regions: list[Region], memory_type: str) -> tuple[Region, ...]:
        return tuple(regions)

    def prep_xfer_dlist(self, agent: str, regions: tuple[Region, ...]) -> int:
        self.next_id += 1
        self.dlists[self.next_id] = regions
        return self.next_id

    def make_prepped_xfer(self, op: str, local: int, lids: list[int], remote: int, rids: list[int]) -> int:
        assert op == "READ" and lids == rids == list(range(len(self.dlists[local])))
        self.next_id += 1
        self.handles[self.next_id] = (local, remote)
        return self.next_id

    def transfer(self, handle: int) -> None:
        local, remote = self.handles[handle]
        for dst, src in zip(self.dlists[local], self.dlists[remote], strict=True):
            assert dst[1] == src[1]
            ctypes.memmove(dst[0], src[0], dst[1])
        self.states[handle] = "PROC"

    def check_xfer_state(self, handle: int) -> str:
        return self.states[handle]

    def release_xfer_handle(self, handle: int) -> None:
        del self.handles[handle]

    def release_dlist_handle(self, dlist: int) -> None:
        del self.dlists[dlist]


@pytest.fixture
def peers(nixl_connector_cls, monkeypatch):  # noqa: F811
    nixl_utils = importlib.import_module("vllm.distributed.nixl_utils")

    monkeypatch.setattr(nixl_utils, "NixlWrapper", PageAgent)
    producer = nixl_connector_cls({"role": "sender", "host": "127.0.0.1", "zmq_port": 0})
    consumer = nixl_connector_cls({"role": "receiver", "metadata_query_timeout_ms": 1000})
    yield producer, consumer
    # These CPU stubs have no outstanding DMA after the explicit DONE checks.
    for connector in (consumer, producer):
        for read_id, read in list(connector._page_reads.items()):
            connector._agent.states[read.handle] = "DONE"
            assert connector.poll_page_read(read_id)
        for pool in connector._page_pools:
            pool.reservations.clear()
        for key, pending in list(connector._pending.items()):
            pending.claims.clear()
            pending.page_claim_ids = frozenset()
            connector.cleanup(key)
        connector.close()


def make_pool(fill: int = 0, *, hnd: bool = False, packed: bool = False) -> KVPagePool:
    caches = {}
    for layer in range(2):
        shape = (8, 2, 16, 8) if packed else (2, 8, 16, 2, 4)
        tensor = torch.full(shape, fill + layer, dtype=torch.bfloat16)
        if packed and not hnd:
            tensor = tensor.transpose(1, 2).contiguous().transpose(1, 2)
        elif not packed and hnd:
            tensor = tensor.transpose(2, 3).contiguous().transpose(2, 3)
        caches[f"layers.{layer}.attn"] = tensor
    layout = ("LBHNC" if hnd else "LBNHC") if packed else ("HND" if hnd else "NHD")
    return KVPagePool(caches, "test-model-revision:bf16:tp1", layout)


def claim(consumer: NixlConnector, key: str, offer: PageOffer, claim_id: str, peers: tuple[str, ...]) -> PageOffer:
    claimed = consumer.claim_pages(
        key,
        offer.sender_host,
        offer.sender_zmq_port,
        generation=offer.generation,
        claim_id=claim_id,
        page_claim_ids=peers,
        geometry=offer.geometry,
    )
    assert claimed is not None
    return claimed


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_cfg_scatter_reuses_registration_and_keeps_compute_ownership(peers, hnd, packed):
    producer, consumer = peers
    source, destination = make_pool(10, hnd=hnd, packed=packed), make_pool(-1, hnd=hnd, packed=packed)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    before = {name: tensor.clone() for name, tensor in destination.caches.items()}
    for iteration in range(3):
        ready = Ready()
        offer = producer.export_pages("cfg", source, (5, 1), 17, expected_readers=2, ready=ready)
        assert ready.calls == 1
        a = claim(consumer, "cfg", offer, "positive", ("positive", "negative"))
        b = claim(consumer, "cfg", offer, "negative", ("positive", "negative"))
        first = ReservedKVPages("positive", iteration, destination, (6, 2))
        second = ReservedKVPages("negative", iteration, destination, (3, 7))
        ra, rb = consumer.read_into("cfg", a, first), consumer.read_into("cfg", b, second)
        assert len(consumer._agent.remotes) == 1
        producer._pending["cfg"].deadline = 0
        producer._cleanup_expired_pending()
        producer.cleanup("cfg")
        assert source.exports and producer._pending["cfg"].claims == {"positive", "negative"}
        consumer._agent.states[consumer._page_reads[ra].handle] = "DONE"
        assert consumer.poll_page_read(ra)
        assert source.exports and producer._pending["cfg"].claims == {"negative"}
        assert len(consumer._agent.remotes) == 1
        with pytest.raises(RuntimeError, match="owned"):
            consumer.unregister_page_pool(destination)
        with pytest.raises(ValueError, match="previous transfer or computation"):
            consumer.read_into("cfg", a, first)
        consumer._agent.states[consumer._page_reads[rb].handle] = "DONE"
        assert consumer.poll_page_read(rb)
        assert not source.exports and "cfg" not in producer._pending
        assert len(consumer._agent.remotes) == 1  # The registered pool reuses its peer.
        for name, tensor in destination.caches.items():
            for blocks in ((6, 2), (3, 7)):
                indexes = torch.tensor(blocks)
                expected = source.caches[name].index_select(source.block_axis, torch.tensor((5, 1)))
                torch.testing.assert_close(
                    tensor.index_select(destination.block_axis, indexes), expected, rtol=0, atol=0
                )
            indexes = torch.tensor((0, 1, 4, 5))
            torch.testing.assert_close(
                tensor.index_select(destination.block_axis, indexes),
                before[name].index_select(destination.block_axis, indexes),
                rtol=0,
                atol=0,
            )
        consumer.retire_pages(first)
        consumer.retire_pages(second)
    assert producer._agent.registrations == consumer._agent.registrations == 1
    assert not destination.reservations
    assert consumer._metrics["page_bytes_completed"] == 2 * producer._metrics["page_bytes_published"]


def test_unknown_dma_late_ack_and_stale_target_generation(peers, monkeypatch):
    producer, consumer = peers
    source, destination = make_pool(2), make_pool(-1)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("late", source, (4,), 7, expected_readers=1, ready=Ready())
    claimed = claim(consumer, "late", offer, "reader", ("reader",))
    target = ReservedKVPages("request", 3, destination, (2,))
    read_id = consumer.read_into("late", claimed, target)
    assert not consumer.poll_page_read(read_id)
    with pytest.raises(RuntimeError, match="unfinished"):
        consumer.retire_pages(target)
    consumer._agent.states[consumer._page_reads[read_id].handle] = "UNKNOWN"
    with pytest.raises(RuntimeError, match="remain pinned"):
        consumer.poll_page_read(read_id)
    assert source.exports and destination.reads and destination.registrations
    consumer.close()
    assert consumer._closing and not consumer._closed
    consumer._agent.states[consumer._page_reads[read_id].handle] = "DONE"
    notify = consumer._notify_transfer_done
    monkeypatch.setattr(consumer, "_notify_transfer_done", lambda key, metadata: False)
    assert not consumer.poll_page_read(read_id)
    assert source.exports and destination.reads and not consumer._agent.handles
    monkeypatch.setattr(consumer, "_notify_transfer_done", notify)
    assert consumer.poll_page_read(read_id)
    with pytest.raises(ValueError, match="stale"):
        consumer.retire_pages(replace(target, allocation_generation=2))
    consumer.retire_pages(target)
    consumer._close_thread.join(2)
    assert consumer._closed and not destination.registrations


def test_cfg_cancel_before_read_is_idempotent_and_preserves_other_reader(peers):
    producer, consumer = peers
    source, destination = make_pool(), make_pool()
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("cancel", source, (1,), 1, expected_readers=2, ready=Ready())
    readers = ("positive", "negative")
    positive = claim(consumer, "cancel", offer, readers[0], readers)
    negative = claim(consumer, "cancel", offer, readers[1], readers)
    assert consumer.cancel_page_claim("cancel", positive)
    assert consumer.cancel_page_claim("cancel", positive)
    assert producer._pending["cancel"].claims == {"negative"} and source.exports
    read_id = consumer.read_into("cancel", negative, ReservedKVPages("negative", 1, destination, (2,)))
    with pytest.raises(RuntimeError, match="submitted"):
        consumer.cancel_page_claim("cancel", negative)
    consumer._agent.states[consumer._page_reads[read_id].handle] = "DONE"
    assert consumer.poll_page_read(read_id)
    assert not source.exports


@pytest.mark.parametrize("field", ["namespace", "layout", "dtype", "device_type", "page_strides", "layers"])
def test_geometry_mismatch_rejected_before_dma(peers, field):
    producer, consumer = peers
    source, destination = make_pool(), make_pool()
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("geometry", source, (1,), 1, expected_readers=1, ready=Ready())
    changes = {
        "namespace": "other",
        "layout": "HND",
        "dtype": "torch.float16",
        "device_type": "cuda",
        "page_strides": (8, 4, 2),
        "layers": ("other",),
    }
    geometry = msgspec.structs.replace(offer.geometry, **{field: changes[field]})
    with pytest.raises(ValueError, match="geometry"):
        consumer.claim_pages(
            "geometry",
            offer.sender_host,
            offer.sender_zmq_port,
            generation=offer.generation,
            claim_id="reader",
            page_claim_ids=("reader",),
            geometry=geometry,
        )
    assert not producer._pending["geometry"].claims
    assert producer._pending["geometry"].page_claim_ids is None
    invalid = msgspec.structs.replace(offer, geometry=geometry, claim_id="reader")
    with pytest.raises(ValueError, match="geometry"):
        consumer.read_into("geometry", invalid, ReservedKVPages("target", 1, destination, (3,)))
    assert not destination.reads and not destination.reservations and not consumer._agent.handles


def test_shared_storage_registered_once_and_strided_pages_have_exact_addresses():
    backing = torch.zeros((2, 2, 8, 16, 2, 4), dtype=torch.bfloat16)
    pool = KVPagePool({"a": backing[0], "b": backing[1]}, "shared", "NHD")
    assert pool.registration_regions() == [(backing.data_ptr(), backing.numel() * 2, 0)]
    regions = pool.regions((7, 2))
    assert len(regions) == 8
    assert regions[4][0] == backing[1, 0, 7].data_ptr()
    assert regions[5][0] == backing[1, 1, 7].data_ptr()
    with pytest.raises(ValueError, match="distinct"):
        pool.regions((2, 2))
    with pytest.raises(ValueError, match="inside"):
        pool.regions((8,))
    with pytest.raises(ValueError, match="dense"):
        KVPagePool({"sliced": backing[0, :, :, :, :, ::2]}, "shared", "NHD")


def test_reused_key_rejects_stale_claim_ack_and_pool_epoch(peers):
    producer, consumer = peers
    source, destination = make_pool(4, packed=True), make_pool(-1, packed=True)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    previous = None
    for generation in range(2):
        offer = producer.export_pages("same-key", source, (4,), 9, expected_readers=1, ready=Ready())
        current = claim(consumer, "same-key", offer, "reader", ("reader",))
        target = ReservedKVPages("target", generation, destination, (2,))
        if previous is not None:
            assert consumer._notify_transfer_done("same-key", msgspec.structs.asdict(previous))
            assert producer._pending["same-key"].claims == {"reader"}
            with pytest.raises(ValueError, match="Stale"):
                consumer.read_into("same-key", previous, target)
            changed_epoch = msgspec.structs.replace(current, pool_epoch="previous-pool")
            with pytest.raises(ValueError, match="Stale"):
                consumer.read_into("same-key", changed_epoch, target)
        read_id = consumer.read_into("same-key", current, target)
        consumer._agent.states[consumer._page_reads[read_id].handle] = "DONE"
        assert consumer.poll_page_read(read_id)
        consumer.retire_pages(target)
        with pytest.raises(ValueError, match="Stale"):
            consumer.read_into("same-key", current, target)
        previous = current
    assert not consumer._page_claims


def test_partial_submission_failure_and_producer_close_keep_both_pools(peers, monkeypatch):
    producer, consumer = peers
    source, destination = make_pool(4), make_pool(-1)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("partial", source, (4,), 9, expected_readers=1, ready=Ready())
    current = claim(consumer, "partial", offer, "reader", ("reader",))

    def submit_partial(handle):
        local, remote = consumer._agent.handles[handle]
        dst, src = consumer._agent.dlists[local][0], consumer._agent.dlists[remote][0]
        ctypes.memmove(dst[0], src[0], dst[1])
        consumer._agent.states[handle] = "PROC"
        raise RuntimeError("partial submission")

    monkeypatch.setattr(consumer._agent, "transfer", submit_partial)
    target = ReservedKVPages("target", 1, destination, (2,))
    with pytest.raises(RuntimeError, match="partial submission"):
        consumer.read_into("partial", current, target)
    producer.close()
    assert producer._closing and not producer._closed
    assert source.exports and destination.reads and destination.reservations
    with pytest.raises(RuntimeError, match="owned"):
        consumer.unregister_page_pool(destination)
    read_id = next(iter(consumer._page_reads))
    assert not consumer.poll_page_read(read_id)
    # Explicit terminal proof from this fake backend permits retirement.
    consumer._agent.states[consumer._page_reads[read_id].handle] = "DONE"
    assert consumer.poll_page_read(read_id)
    consumer.retire_pages(target)
    producer._close_thread.join(2)
    assert producer._closed


def test_close_before_first_claim_drains_published_pages_without_ttl_release(peers):
    producer, consumer = peers
    source, destination = make_pool(3), make_pool(-1)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("closing", source, (4,), 9, expected_readers=1, ready=Ready())
    producer.cleanup("closing")
    assert source.exports and "closing" in producer._pending
    producer.close()
    assert not producer._closed and source.registrations
    current = claim(consumer, "closing", offer, "reader", ("reader",))
    target = ReservedKVPages("target", 1, destination, (2,))
    read_id = consumer.read_into("closing", current, target)
    consumer._agent.states[consumer._page_reads[read_id].handle] = "DONE"
    assert consumer.poll_page_read(read_id)
    consumer.retire_pages(target)
    producer._close_thread.join(2)
    assert producer._closed and not source.registrations
