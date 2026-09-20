"""Require actual multi-rank execution and correct trained LoRA tensor partitions."""

import argparse
import json
from pathlib import Path

from verify_switch import read_rows, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--parallel", required=True, type=Path)
    args = parser.parse_args()
    verify(args.native, 1)
    verify(args.parallel, 2)
    reference = read_rows(args.native / "rank-0.jsonl")
    native_execution = read_rows(args.native / "parallel-rank-0.jsonl")
    assert len(native_execution) == 24
    feature = read_rows(args.parallel / "results.jsonl")[0]["feature"]
    rank_requests = []
    for rank in range(2):
        lora = read_rows(args.parallel / f"rank-{rank}.jsonl")
        observations = read_rows(args.parallel / f"parallel-rank-{rank}.jsonl")
        assert len(observations) == 24
        requests = [row["request_id"] for row in observations]
        assert requests == [row["request_id"] for row in lora]
        rank_requests.append(requests)
        for ref, native, row, execution in zip(reference, native_execution, lora, observations):
            assert execution["rank"] == rank
            assert execution["tp_size"] == (2 if feature == "tp" else 1)
            assert execution["ulysses_size"] == (2 if feature == "ulysses" else 1)
            assert execution["ring_size"] == (2 if feature == "ring" else 1)
            assert len(execution["positions"]) == len(native["positions"]) > 0
            for positions, full_positions in zip(execution["positions"], native["positions"]):
                for name in ("attention_mask", "cos", "sin"):
                    current, full = positions[name], full_positions[name]
                    expected = full["hash"]
                    if feature in ("ulysses", "ring"):
                        reconstructed = current["shape"].copy()
                        reconstructed[1] *= 2
                        assert reconstructed == full["shape"]
                        expected = full["halves"][rank]
                    else:
                        assert current["shape"] == full["shape"]
                    assert current["hash"] == expected, "Position or mask partition differs"
            assert row["registered_adapters"] == ref["registered_adapters"] and row["scales"] == ref["scales"]
            ref_layers = {layer["name"]: layer for layer in ref["layers"]}
            assert {layer["name"] for layer in row["layers"]} == set(ref_layers)
            for layer in row["layers"]:
                base = ref_layers[layer["name"]]
                assert layer["active_slices"] == base["active_slices"]
                assert layer["calls"] == base["calls"] > 0
                for part, split_dimension in (("a", -1), ("b", -2)):
                    assert len(layer[f"{part}_shapes"]) == len(base[f"{part}_shapes"])
                    for index, shape in enumerate(layer[f"{part}_shapes"]):
                        full_shape = base[f"{part}_shapes"][index]
                        expected = base[f"{part}_hashes"][index]
                        if shape != full_shape:
                            assert feature == "tp"
                            reconstructed = shape.copy()
                            reconstructed[split_dimension] *= 2
                            assert reconstructed == full_shape
                            expected = base[f"{part}_half_hashes"][index][rank]
                        assert layer[f"{part}_hashes"][index] == expected, "LoRA partition values differ"
    assert rank_requests[0] == rank_requests[1]
    print(json.dumps({"feature": feature, "requests": 24, "ranks": 2, "partition_gate": "passed"}))


if __name__ == "__main__":
    main()
