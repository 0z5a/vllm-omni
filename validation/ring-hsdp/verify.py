# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify Ring attention state and HSDP parity probes."""

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("hsdp", type=Path)
    args = parser.parse_args()
    native_results = rows(args.native / "results.jsonl")
    hsdp_results = rows(args.hsdp / "results.jsonl")
    assert [row["image_sha256"] for row in native_results] == [row["image_sha256"] for row in hsdp_results]
    for rank in range(2):
        native = rows(args.native / f"rank-{rank}.jsonl")
        hsdp = rows(args.hsdp / f"rank-{rank}.jsonl")
        assert len(native) == len(hsdp) == 3
        for left, right in zip(native, hsdp, strict=True):
            assert left["hsdp"] is False and right["hsdp"] is True
            assert left["ring_size"] == right["ring_size"] == 2
            assert left["sharded_parameters"] == 0
            assert right["sharded_parameters"] > 0
            assert left["attention"] == right["attention"]
            assert left["positions"] == right["positions"]
    print("Ring and HSDP probes match")


if __name__ == "__main__":
    main()
