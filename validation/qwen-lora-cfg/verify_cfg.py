"""Match observed CFG rank conditioning to native positive/negative forwards."""

import argparse
import json
from pathlib import Path

from verify_switch import read_rows, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--parallel", type=Path, required=True)
args = parser.parse_args()
native_hashes = verify(args.native, 1)
parallel_hashes = verify(args.parallel, 2)
assert native_hashes == parallel_hashes, "CFG2 PNG output differs from CFG1"
native = read_rows(args.native / "cfg-rank-0.jsonl")
assert len(native) == 24
for rank in range(2):
    branch = read_rows(args.parallel / f"cfg-rank-{rank}.jsonl")
    activations = read_rows(args.parallel / f"rank-{rank}.jsonl")
    assert len(branch) == len(activations) == 24
    for reference, observed, activation in zip(native, branch, activations):
        assert reference["cfg_world_size"] == 1 and observed["cfg_world_size"] == 2
        assert observed["cfg_rank"] == rank
        assert observed["request_id"] == activation["request_id"]
        assert observed["conditioning"] == reference["conditioning"][rank::2]
left = read_rows(args.parallel / "rank-0.jsonl")
right = read_rows(args.parallel / "rank-1.jsonl")
for positive, negative in zip(left, right):
    for field in ("request_id", "active_adapter", "registered_adapters", "scales"):
        assert positive[field] == negative[field], field
print(
    json.dumps(
        {"requests": 24, "cfg_ranks": 2, "branch_conditioning_matches_native": True, "adapter_rank_agreement": True}
    )
)
