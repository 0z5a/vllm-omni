# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import msgspec
import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from tests.diffusion.diffusion_kv.test_manager import _manager, _request
from tests.distributed.omni_connectors.test_nixl_connector import nixl_connector_cls  # noqa: F401
from tests.distributed.omni_connectors.test_nixl_pages import Ready, peers  # noqa: F401
from vllm_omni.diffusion.diffusion_kv.kv_connector import commit_kv_load, prepare_kv_requests
from vllm_omni.distributed.omni_connectors.connectors.nixl_kv_connector import (
    OmniNixlKVConnector,
    PageTicket,
)
from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import KVPagePool

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


def scheduler(manager, *, producer: bool = False) -> OmniNixlKVConnector:
    config = VllmConfig()
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OmniNixlKVConnector",
        kv_role="kv_producer" if producer else "kv_consumer",
        engine_id="test-ar" if producer else "test-dit",
        kv_ip="127.0.0.1",
        kv_connector_extra_config={"page_namespace": "native-test", "page_port": 8998, "expected_readers": 2},
    )
    return OmniNixlKVConnector(config, KVConnectorRole.SCHEDULER, manager.native_manager.kv_cache_config)


def pool(fill: int) -> KVPagePool:
    cache = torch.full((12, 2, 4, 16), fill, dtype=torch.bfloat16).transpose(1, 2).contiguous().transpose(1, 2)
    return KVPagePool({"layer0": cache}, "native-test", "LBNHC")


@pytest.mark.parametrize("abort_before_read", [False, True])
@pytest.mark.parametrize("token_mismatch", [False, True])
@pytest.mark.parametrize(
    "prefix_lengths,reusable_lengths",
    [((7, 6), (7, 6)), ((7, 0), (7, 0)), ((7, 6), (4, 0))],
)
def test_native_manager_generations_cfg_claims_and_delayed_source_free(
    peers,  # noqa: F811
    abort_before_read,
    token_mismatch,
    prefix_lengths,
    reusable_lengths,
):
    producer, consumer = peers
    source_manager, target_manager = _manager(12), _manager(12)
    source_scheduler, target_scheduler = scheduler(source_manager, producer=True), scheduler(target_manager)
    source_scheduler.extra["page_port"] = producer._zmq_port
    source_worker, target_worker = scheduler(source_manager, producer=True), scheduler(target_manager)
    source_worker._transport, target_worker._transport = producer, consumer
    source_worker._pool, target_worker._pool = pool(11), pool(-1)
    producer.register_page_pool(source_worker._pool)
    consumer.register_page_pool(target_worker._pool)
    source_worker._events = {"layer0": Ready()}
    for iteration in range(2):
        source_request = _request("ar", 0, prefix_len=7, target_len=1, seq_len=8)
        source_manager.reserve_request("ar", (source_request,))
        source_request = Request(source_request.request_id, list(range(7)), SamplingParams(max_tokens=1), None)
        source_request.kv_transfer_params = {"transfer_id": "cfg", "do_remote_decode": True}
        source_request.num_computed_tokens = 7
        source_request.status = RequestStatus.FINISHED_STOPPED
        blocks = source_manager.native_manager.get_block_ids(source_request.request_id)[0]
        delay_free, params = source_scheduler.request_finished(source_request, blocks)
        assert delay_free and params is not None
        assert source_scheduler.has_pending_push_work() and source_scheduler.has_pending_block_frees()
        ticket = msgspec.convert(params, type=PageTicket)
        requests = tuple(
            _request("dit", row, prefix_len=length, target_len=9 - length, seq_len=9, cache_token_ids=range(length))
            for row, length in enumerate(prefix_lengths)
        )
        for request, reusable in zip(requests, reusable_lengths, strict=True):
            request.prompt_token_ids = list(request.cache_token_ids[:reusable])
            if token_mismatch and reusable > 3:
                request.prompt_token_ids[3] = 99
        prepare_kv_requests(requests, params)
        metadata = target_manager.reserve_request("dit", requests)
        assert metadata is not None and metadata.allocation_generation == iteration + 1
        assert all(request.allocation_generation == metadata.allocation_generation for request in requests)
        matched = [target_scheduler.get_num_new_matched_tokens(request, 0)[0] for request in requests]
        assert matched == [min(length, 3) if token_mismatch else length for length in reusable_lengths]
        assert commit_kv_load(target_scheduler, target_manager.native_manager, requests, matched) == {
            request.request_id for request in requests
        }
        source_worker.bind_connector_metadata(source_scheduler.build_connector_meta(None))
        source_worker.start_load_kv(None)
        source_worker.get_finished({source_request.request_id})
        assert ticket.source_generation in source_worker._pool.exports
        target_meta = target_scheduler.build_connector_meta(None)
        claims = tuple(target.claim_id for target in target_meta.targets)
        assert all(target.peer_claims == claims for target in target_meta.targets)
        assert all(target.allocation_generation == iteration + 1 for target in target_meta.targets)
        target_worker.bind_connector_metadata(target_meta)
        target_worker.start_load_kv(None)
        if abort_before_read:
            finished_ids = {request.request_id for request in requests}
            assert target_worker.get_finished(finished_ids) == (set(), finished_ids)
            assert target_worker.get_finished(finished_ids) == (set(), set())
            assert not consumer._page_reads and not target_worker._pool.reservations
        else:
            assert target_worker.get_finished(set()) == (set(), set())
            handles = list(consumer._page_reads.values())
            consumer._agent.states[handles[0].handle] = "DONE"
            _, received = target_worker.get_finished(set())
            assert len(received) == 1 and source_worker.get_finished(set())[0] == set()
            assert source_manager.has_request("ar") and source_worker._pool.exports
            consumer._agent.states[handles[1].handle] = "DONE"
            _, received = target_worker.get_finished(set())
            assert len(received) == 1
        sent, _ = source_worker.get_finished(set())
        assert sent == {source_request.request_id}
        source_scheduler.update_connector_output(KVConnectorOutput(finished_sending=sent))
        assert not source_scheduler.has_pending_push_work()
        source_manager.free_request("ar")
        assert target_manager.has_request("dit")
        assert bool(target_worker._pool.reservations) is not abort_before_read
        target_worker.get_finished({request.request_id for request in requests})
        assert not target_worker._pool.reservations
        target_manager.free_request("dit")


def test_incomplete_cfg_metadata_retains_native_reservations_until_rollback():
    manager = _manager(12)
    connector = scheduler(manager)
    requests = (_request("cfg", 0, cache_token_ids=range(4)), _request("cfg", 1, cache_token_ids=range(4)))
    for request in requests:
        request.prompt_token_ids = list(request.cache_token_ids)
    params = msgspec.structs.asdict(PageTicket("cfg", "generation", "127.0.0.1", 8998, 4, 2, tuple(range(4))))
    prepare_kv_requests(requests, {**params, "do_remote_prefill": True})
    manager.reserve_request("cfg", requests)
    first = requests[0]
    commit_kv_load(connector, manager.native_manager, (first,), [4])
    with pytest.raises(ValueError, match="All CFG reservations"):
        connector.build_connector_meta(None)
    assert manager.has_request("cfg") and first.request_id in connector._targets
    first.status = RequestStatus.FINISHED_ABORTED
    assert connector.request_finished(first, [])[0] is False
    assert not connector._targets
    manager.free_request("cfg")
