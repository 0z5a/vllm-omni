# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Gate each real-feature probe and the trained-adapter switch sequence."""

import argparse
import json
from pathlib import Path

from verify_switch import read_rows, verify

EXPECTED = {
    "native": (1, None),
    "layer": (1, "LayerWiseOffloadBackend"),
    "module": (1, "ModelLevelOffloadBackend"),
    "fp8": (1, None),
    "u2-native": (2, None),
    "u2-vae2": (2, None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--feature", choices=tuple(EXPECTED), required=True)
    args = parser.parse_args()
    ranks, backend = EXPECTED[args.feature]
    verify(args.directory, ranks)
    rank_requests = []
    for rank in range(ranks):
        rows = read_rows(args.directory / f"feature-rank-{rank}.jsonl")
        assert len(rows) == 24
        rank_requests.append([row["request_id"] for row in rows])
        for row in rows:
            assert row["rank"] == rank and row["feature"] == args.feature
            assert row["offload_backend"] == backend
            if args.feature == "layer":
                operations = {event["operation"] for event in row["layer_events"]}
                assert operations == {"prefetch", "offload"} and not row["module_events"]
            elif args.feature == "module":
                targets = {event["target"] for event in row["module_events"]}
                assert targets == {"cpu", "cuda"} and not row["layer_events"]
            else:
                assert not row["layer_events"] and not row["module_events"]
            if args.feature == "fp8":
                assert row["fp8_kernels"] and set(row["fp8_kernels"]) == set(row["fp8_calls"])
                assert all(calls > 0 for calls in row["fp8_calls"].values())
            else:
                assert not row["fp8_kernels"] and not row["fp8_calls"]
            if args.feature == "u2-vae2":
                assert row["vae_events"]
                assert all(
                    event["rank"] == rank and event["world_size"] == 2 and event["parallel_size"] == 2
                    for event in row["vae_events"]
                )
            else:
                assert not row["vae_events"]
    assert all(requests == rank_requests[0] for requests in rank_requests)
    print(json.dumps({"feature": args.feature, "requests": 24, "ranks": ranks, "passed": True}))


if __name__ == "__main__":
    main()
