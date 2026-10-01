# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-process, two-GPU comparison of equal-byte KV handoffs."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import time
from multiprocessing.connection import Connection
from pathlib import Path

import msgspec
import torch

from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector
from vllm_omni.distributed.omni_connectors.connectors.paged_transfer import KVPagePool, PageOffer, ReservedKVPages

MODES = ("ordinary", "local_copy", "pages")


class Offer(msgspec.Struct):
    metadata: bytes
    start_ns: int
    prepare_ns: int
    byte_count: int


class Sample(msgspec.Struct):
    mode: str
    repeat: int
    sample: int
    bytes: int
    prepare_ns: int
    receive_install_ns: int
    e2e_ns: int
    receive_tensor_allocation_bytes: int


def pool(device: int, layers: int, blocks: int) -> KVPagePool:
    shape = (blocks * 2, 4, 16, 256)
    caches = {}
    for layer in range(layers):
        values = torch.arange(math.prod(shape), device=f"cuda:{device}")
        tensor = ((values % 127 + layer) / 32).to(torch.bfloat16).reshape(shape)
        caches[f"layer.{layer}"] = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    return KVPagePool(caches, "benchmark:bf16", "LBNHC")


def source_worker(wire: Connection, layers: int, blocks: int, samples: int, warmup: int, repeats: int) -> None:
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device("cuda:0")
    source = pool(0, layers, blocks)
    ids = tuple((i * 37) % (blocks * 2) for i in range(blocks))
    connector = NixlConnector({"role": "sender", "host": "127.0.0.1", "zmq_port": 0})
    connector.register_page_pool(source)
    ready = torch.cuda.Event()
    ready.record()
    ready.synchronize()
    byte_count = sum(region[1] for region in source.regions(ids))
    for mode in MODES:
        for repeat in range(repeats):
            for sample in range(-warmup, samples):
                key = f"{mode}-{repeat}-{sample}"
                start = time.perf_counter_ns()
                if mode == "ordinary":
                    packed = [tensor[list(ids)] for tensor in source.caches.values()]
                    torch.accelerator.synchronize(0)
                    ok, size, metadata = connector.put("0", "1", key, packed)
                    assert ok and size == byte_count and metadata is not None
                    encoded = msgspec.msgpack.encode(metadata)
                elif mode == "pages":
                    offer = connector.export_pages(key, source, ids, blocks * 16, expected_readers=1, ready=ready)
                    encoded = msgspec.msgpack.encode(offer)
                else:
                    encoded = b""
                wire.send_bytes(
                    msgspec.msgpack.encode(Offer(encoded, start, time.perf_counter_ns() - start, byte_count))
                )
                assert wire.poll(60) and wire.recv_bytes() == b"consumed"
                assert key not in connector._pending
    connector.unregister_page_pool(source)
    connector.close()
    assert connector._closed
    wire.send_bytes(msgspec.json.encode(connector._metrics))
    wire.close()


def target_worker(
    wire: Connection,
    layers: int,
    blocks: int,
    samples: int,
    warmup: int,
    repeats: int,
    out: str,
) -> None:
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device("cuda:1")
    destination = pool(1, layers, blocks)
    reference = pool(1, layers, blocks)
    source_ids = tuple((i * 37) % (blocks * 2) for i in range(blocks))
    target_ids = tuple((i * 17) % (blocks * 2) for i in range(blocks))
    connector = NixlConnector({"role": "receiver", "metadata_query_timeout_ms": 1000})
    connector.register_page_pool(destination)
    torch.accelerator.synchronize(1)
    records = []
    for mode in MODES:
        print(f"Measuring {mode}: {warmup} warmups and {samples} samples per repeat", flush=True)
        for repeat in range(repeats):
            for sample in range(-warmup, samples):
                assert wire.poll(60)
                message = msgspec.msgpack.decode(wire.recv_bytes(), type=Offer)
                key = f"{mode}-{repeat}-{sample}"
                memory_before = torch.accelerator.memory_allocated(1)
                start = time.perf_counter_ns()
                if mode == "ordinary":
                    metadata = msgspec.msgpack.decode(message.metadata)
                    received = connector.get("0", "1", key, metadata)
                    deadline = time.monotonic() + 30
                    while received is None:
                        assert time.monotonic() < deadline
                        received = connector.get("0", "1", key, metadata)
                    tensors, size = received
                    assert isinstance(tensors, list) and size == message.byte_count
                    extra_memory = torch.accelerator.memory_allocated(1) - memory_before
                    for tensor, target in zip(tensors, destination.caches.values(), strict=True):
                        assert isinstance(tensor, torch.Tensor)
                        target[list(target_ids)] = tensor
                    torch.accelerator.synchronize(1)
                    del tensors, received, tensor
                elif mode == "pages":
                    offer = msgspec.msgpack.decode(message.metadata, type=PageOffer)
                    claimed = None
                    deadline = time.monotonic() + 30
                    while claimed is None:
                        assert time.monotonic() < deadline
                        claimed = connector.claim_pages(
                            key,
                            offer.sender_host,
                            offer.sender_zmq_port,
                            generation=offer.generation,
                            claim_id=key,
                            page_claim_ids=(key,),
                            geometry=destination.geometry,
                        )
                    reservation = ReservedKVPages(key, repeat * samples + sample + warmup, destination, target_ids)
                    read_id = connector.read_into(key, claimed, reservation)
                    while not connector.poll_page_read(read_id):
                        assert time.monotonic() < deadline, "Native pages remain pinned for inspection"
                        time.sleep(0.0001)
                    extra_memory = torch.accelerator.memory_allocated(1) - memory_before
                    connector.retire_pages(reservation)
                else:
                    for tensor, target in zip(reference.caches.values(), destination.caches.values(), strict=True):
                        target[list(target_ids)] = tensor[list(source_ids)]
                    torch.accelerator.synchronize(1)
                    extra_memory = torch.accelerator.memory_allocated(1) - memory_before
                end = time.perf_counter_ns()
                if sample == 0:
                    for target, expected in zip(destination.caches.values(), reference.caches.values(), strict=True):
                        torch.testing.assert_close(target[list(target_ids)], expected[list(source_ids)], rtol=0, atol=0)
                if sample >= 0:
                    records.append(
                        Sample(
                            mode,
                            repeat,
                            sample,
                            message.byte_count,
                            message.prepare_ns,
                            end - start,
                            end - message.start_ns,
                            extra_memory,
                        )
                    )
                wire.send_bytes(b"consumed")
    assert wire.poll(60)
    source_metrics = msgspec.json.decode(wire.recv_bytes())
    connector.unregister_page_pool(destination)
    connector.close()
    assert connector._closed
    Path(out).write_bytes(msgspec.json.encode(records))
    Path(out).with_name("transport-metrics.json").write_bytes(
        msgspec.json.encode({"source": source_metrics, "target": connector._metrics})
    )
    wire.close()


def summarize(out: Path) -> None:
    samples = msgspec.json.decode((out / "samples.json").read_bytes(), type=list[Sample])
    rows = []
    for mode in MODES:
        group = [sample for sample in samples if sample.mode == mode]
        elapsed = sorted(sample.e2e_ns / 1e6 for sample in group)
        rows.append(
            {
                "mode": mode,
                "samples": len(group),
                "bytes": group[0].bytes,
                "p50_ms": statistics.median(elapsed),
                "p95_ms": elapsed[math.ceil(len(elapsed) * 0.95) - 1],
                "p99_ms": elapsed[math.ceil(len(elapsed) * 0.99) - 1],
                "prepare_p50_ms": statistics.median(sample.prepare_ns / 1e6 for sample in group),
                "receive_install_p50_ms": statistics.median(sample.receive_install_ns / 1e6 for sample in group),
                "receive_tensor_allocation_bytes": max(sample.receive_tensor_allocation_bytes for sample in group),
            }
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "| Path | Bytes / transfer | Samples | p50 ms | p95 ms | p99 ms | Speedup vs ordinary | Receive tensor bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['bytes']} | {row['samples']} | {row['p50_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {rows[0]['p50_ms'] / row['p50_ms']:.2f}× | "
            f"{row['receive_tensor_allocation_bytes']} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 100 or args.warmup < 20 or args.repeats < 3:
        parser.error("Require at least 20 warmups, 100 samples and three repeats")
    if args.blocks & (args.blocks - 1) or min(args.layers, args.blocks) < 1:
        parser.error("Require positive layers and a power-of-two block count")
    args.out.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    source_wire, target_wire = context.Pipe()
    common = (args.layers, args.blocks, args.samples, args.warmup, args.repeats)
    source = context.Process(target=source_worker, args=(source_wire, *common))
    target = context.Process(target=target_worker, args=(target_wire, *common, str(args.out / "samples.json")))
    source.start()
    target.start()
    source.join(600)
    target.join(600)
    assert not source.is_alive(), f"Source PID {source.pid} remains running for inspection"
    assert not target.is_alive(), f"Target PID {target.pid} remains running for inspection"
    assert source.exitcode == target.exitcode == 0
    summarize(args.out)


if __name__ == "__main__":
    main()
