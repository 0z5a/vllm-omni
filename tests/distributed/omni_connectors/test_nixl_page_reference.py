# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import msgspec
import pytest
import torch

from tests.distributed.omni_connectors.test_nixl_connector import nixl_connector_cls  # noqa: F401
from tests.distributed.omni_connectors.test_nixl_pages import Ready, claim, make_pool, peers  # noqa: F401
from tools.nixl_page_reference import ReferenceNixlConnector, ReferenceOffer
from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import ReservedKVPages

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


def test_reference_reuses_claim_and_holds_native_pages_until_scatter_ack(peers, monkeypatch):  # noqa: F811
    producer, _ = peers
    consumer = ReferenceNixlConnector({"role": "receiver", "metadata_query_timeout_ms": 1000})
    source, destination = make_pool(10), make_pool(-1)
    producer.register_page_pool(source)
    consumer.register_page_pool(destination)
    offer = producer.export_pages("reference", source, (5, 1), 17, expected_readers=2, ready=Ready())
    producer._published["reference"]["ordinary_metadata"] = msgspec.msgpack.encode(
        {"generation": offer.generation, "sender_host": offer.sender_host, "sender_zmq_port": offer.sender_zmq_port}
    )
    readers = ("copy", "cancel")
    copied = claim(consumer, "reference", offer, readers[0], readers)
    cancelled = claim(consumer, "reference", offer, readers[1], readers)
    assert isinstance(copied, ReferenceOffer) and copied.ordinary_metadata
    assert consumer.cancel_page_claim("reference", cancelled)
    assert consumer.cancel_page_claim("reference", cancelled)

    def receive(from_stage, to_stage, key, metadata):
        assert metadata == {}  # The outer page lease owns the ACK.
        return [tensor.index_select(source.block_axis, torch.tensor((5, 1))) for tensor in source.caches.values()], 0

    monkeypatch.setattr(consumer, "get", receive)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None))
    target = ReservedKVPages("copy", 1, destination, (6, 2))
    read_id = consumer.read_into("reference", copied, target)
    for name, tensor in destination.caches.items():
        torch.testing.assert_close(
            tensor.index_select(destination.block_axis, torch.tensor((6, 2))),
            source.caches[name].index_select(source.block_axis, torch.tensor((5, 1))),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="previous transfer or computation"):
        consumer.read_into("reference", copied, target)
    with pytest.raises(RuntimeError, match="unfinished"):
        consumer.retire_pages(target)
    with pytest.raises(RuntimeError, match="submitted"):
        consumer.cancel_page_claim("reference", copied)
    notify = consumer._notify_transfer_done
    monkeypatch.setattr(consumer, "_notify_transfer_done", lambda key, metadata: False)
    assert not consumer.poll_page_read(read_id)
    assert source.exports and destination.reads
    monkeypatch.setattr(consumer, "_notify_transfer_done", notify)
    assert consumer.poll_page_read(read_id)
    assert not source.exports and not destination.reads and destination.reservations
    consumer.retire_pages(target)
    consumer.close()
    assert consumer._closed
