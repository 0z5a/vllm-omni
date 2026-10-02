"""Check NCCL data correctness after the existing finite model queue exits."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def worker(rank: int, port: int, output: Path) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    group = dist.new_group([0, 1], backend="nccl")
    checks = []

    def record(operation: str, actual, expected, dtype: str, count: int, round_: int):
        checks.append(
            {
                "operation": operation,
                "dtype": dtype,
                "count": count,
                "round": round_,
                "mismatches": (actual != expected).count_nonzero().item(),
                "finite": torch.isfinite(actual).all().item(),
            }
        )

    for dtype in (torch.bfloat16, torch.float32, torch.int32):
        for count in (4096, 65536, 1048576, 4194304):
            for round_ in range(3):
                pattern = torch.arange(count, device=rank) % 97 - 48 + round_
                if dtype != torch.int32:
                    pattern = pattern.float() / 8
                inputs = [(pattern + peer * 16).to(dtype) for peer in range(2)]
                local = inputs[rank].clone()
                gathered = torch.empty(count * 2, device=rank, dtype=dtype)
                dist.all_gather_into_tensor(gathered, local, group=group)
                record(
                    "all_gather_into_tensor",
                    gathered,
                    torch.cat(inputs),
                    str(dtype),
                    count,
                    round_,
                )
                reduced = local.clone()
                dist.all_reduce(reduced, group=group)
                record(
                    "all_reduce_sum",
                    reduced,
                    inputs[0] + inputs[1],
                    str(dtype),
                    count,
                    round_,
                )
                for source in range(2):
                    broadcast = local.clone()
                    dist.broadcast(broadcast, source, group=group)
                    record(
                        f"broadcast_from_{source}",
                        broadcast,
                        inputs[source],
                        str(dtype),
                        count,
                        round_,
                    )
    torch.cuda.synchronize()
    report = {
        "rank": rank,
        "checks": checks,
        "passed": all(row["mismatches"] == 0 and row["finite"] for row in checks),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "nccl_environment": {
            k: v for k, v in os.environ.items() if k.startswith("NCCL_")
        },
        "scope": "Eager ProcessGroupNCCL data operations used by CFG/FSDP; no transport override, no performance claim",
    }
    (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    raise SystemExit(0 if report["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.rank is not None:
        worker(args.rank, args.port, args.output)
        return
    args.output.mkdir(exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = os.environ.copy()
    environment.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH")
    children = []
    logs = []
    for rank in range(2):
        log = (args.output / f"rank-{rank}.log").open("x")
        logs.append(log)
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--output",
                str(args.output),
                "--rank",
                str(rank),
                "--port",
                str(port),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append(child)
        (args.output / f"rank-{rank}.pid").write_text(str(child.pid))
    exits = [child.wait() for child in children]
    for rank, log in enumerate(logs):
        log.close()
        (args.output / f"rank-{rank}.exit").write_text(str(exits[rank]))
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "exits": exits,
                "passed": exits == [0, 0],
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "forced_signals": False,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if exits == [0, 0] else 1)


if __name__ == "__main__":
    main()
