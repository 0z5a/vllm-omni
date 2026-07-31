"""Pure-TP KV receive consensus (#5627) — CPU-testable mechanism.

In pure TP (LOCAL role) each rank fetches its own KV shard independently and
``distribute_kv_cache`` is a no-op, so one rank's silent receive timeout used
to send only that rank into the pipeline's no-KV prefill — desyncing TP
collectives until the NCCL watchdog killed the worker. These tests cover the
all-or-nothing verdict exchange: skip guards, both rank roles, KV dropping on
a peer miss, and the wiring from both distributed-receive entry points.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
    _TransferTopoConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeGroup:
    """Scripted stand-in for one rank's view of a GroupCoordinator."""

    def __init__(self, rank_in_group, world_size, peer_flags=None, verdict=None):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self._peer_flags = list(peer_flags or [])
        self._verdict = verdict
        self.sent = []
        self.broadcasts = []

    def recv_object(self, src):
        return self._peer_flags.pop(0)

    def send_object(self, obj, dst):
        self.sent.append((obj, dst))

    def broadcast_object(self, obj=None, src=0):
        self.broadcasts.append(obj)
        return obj if obj is not None else self._verdict


def _topo(role=ReceiveRole.LOCAL, tp_active=True, world=None):
    return _TransferTopoConfig(
        role=role,
        tp_active=tp_active,
        cfg_size=1,
        cfg_rank=0,
        cfg_group=None,
        sp_size=1,
        sp_rank=0,
        sp_group=None,
        world=world,
    )


def _manager(topo, need_recv_cache=True):
    mgr = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=need_recv_cache))
    mgr._topo_config = topo
    return mgr


def _req_with_kv(rid="r1"):
    kv = SimpleNamespace(key_cache=[], value_cache=[])
    return SimpleNamespace(
        request_id=rid,
        past_key_values=kv,
        kv_metadata={"ropes": [3]},
        sampling_params=SimpleNamespace(past_key_values=kv, kv_metadata={"ropes": [3]}),
    )


# --------------------------------------------------------------------------- #
#  Skip guards — no exchange may happen (a one-sided exchange would deadlock)
# --------------------------------------------------------------------------- #


def test_pass_through_outside_pure_tp():
    group = _FakeGroup(0, 2, peer_flags=[False])
    for topo in (
        _topo(role=ReceiveRole.LEADER, tp_active=True, world=group),
        _topo(role=ReceiveRole.LOCAL, tp_active=False, world=group),
        _topo(role=ReceiveRole.LOCAL, tp_active=True, world=None),
    ):
        mgr = _manager(topo)
        assert mgr._tp_local_receive_consensus(_req_with_kv(), True) is True
        assert group.sent == [] and group.broadcasts == []


def test_pass_through_single_rank_group_no_recv_config_and_dummy():
    group = _FakeGroup(0, 1)
    assert _manager(_topo(world=group))._tp_local_receive_consensus(_req_with_kv(), True) is True

    group2 = _FakeGroup(0, 2, peer_flags=[False])
    mgr = _manager(_topo(world=group2), need_recv_cache=False)
    assert mgr._tp_local_receive_consensus(_req_with_kv(), True) is True

    mgr3 = _manager(_topo(world=group2))
    dummy = _req_with_kv(rid=DUMMY_DIFFUSION_REQUEST_ID)
    assert mgr3._tp_local_receive_consensus(dummy, False) is False
    assert group2.sent == [] and group2.broadcasts == []


# --------------------------------------------------------------------------- #
#  Rank 0: collects flags, decides, broadcasts
# --------------------------------------------------------------------------- #


def test_rank0_drops_kv_when_peer_missed():
    group = _FakeGroup(0, 2, peer_flags=[False])
    mgr = _manager(_topo(world=group))
    req = _req_with_kv()

    assert mgr._tp_local_receive_consensus(req, True) is False
    assert group.broadcasts == [False]  # peers were told to fall back too
    assert req.past_key_values is None
    assert req.kv_metadata is None
    assert req.sampling_params.past_key_values is None
    assert req.sampling_params.kv_metadata is None


def test_rank0_keeps_kv_when_all_received():
    group = _FakeGroup(0, 2, peer_flags=[True])
    mgr = _manager(_topo(world=group))
    req = _req_with_kv()

    assert mgr._tp_local_receive_consensus(req, True) is True
    assert group.broadcasts == [True]
    assert req.past_key_values is not None
    assert req.sampling_params.past_key_values is not None


def test_rank0_own_miss_fails_verdict_even_if_peers_received():
    group = _FakeGroup(0, 3, peer_flags=[True, True])
    mgr = _manager(_topo(world=group))
    assert mgr._tp_local_receive_consensus(_req_with_kv(), False) is False
    assert group.broadcasts == [False]


# --------------------------------------------------------------------------- #
#  Rank > 0: reports flag, honors verdict
# --------------------------------------------------------------------------- #


def test_follower_rank_drops_kv_on_negative_verdict():
    group = _FakeGroup(1, 2, verdict=False)
    mgr = _manager(_topo(world=group))
    req = _req_with_kv()

    assert mgr._tp_local_receive_consensus(req, True) is False
    assert group.sent == [(True, 0)]
    assert req.past_key_values is None
    assert req.sampling_params.past_key_values is None


def test_follower_rank_keeps_kv_on_positive_verdict():
    group = _FakeGroup(1, 2, verdict=True)
    mgr = _manager(_topo(world=group))
    req = _req_with_kv()

    assert mgr._tp_local_receive_consensus(req, True) is True
    assert group.sent == [(True, 0)]
    assert req.past_key_values is not None


# --------------------------------------------------------------------------- #
#  Wiring — both distributed-receive entry points run the consensus
# --------------------------------------------------------------------------- #


def _wired_manager(monkeypatch, consensus_calls):
    mgr = _manager(_topo(world=_FakeGroup(0, 2)))

    def fake_consensus(req, received):
        consensus_calls.append(received)
        return False

    monkeypatch.setattr(mgr, "_tp_local_receive_consensus", fake_consensus)
    monkeypatch.setattr(mgr, "receive_multi_kv_cache", lambda *a, **k: True)
    return mgr


def test_receive_multi_kv_cache_distributed_runs_consensus(monkeypatch):
    calls = []
    mgr = _wired_manager(monkeypatch, calls)
    assert mgr.receive_multi_kv_cache_distributed(_req_with_kv()) is False
    assert calls == [True]


def test_consume_and_distribute_kv_cache_runs_consensus(monkeypatch):
    calls = []
    mgr = _wired_manager(monkeypatch, calls)
    assert mgr.consume_and_distribute_kv_cache(_req_with_kv()) is False
    assert calls == [True]
