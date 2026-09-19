"""Gate real switch probes; hashes are checked against saved PNG bytes."""

import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(directory: Path, ranks: int) -> list[str]:
    rows = read_rows(directory / "results.jsonl")
    assert len(rows) == 24 and all(row["probe"] for row in rows)
    hashes = []
    for index, row in enumerate(rows):
        case, position = divmod(index, 8)
        assert (row["case"], row["cycle"], row["position"]) == (case, 0, position)
        path = directory / f"case-{case}-cycle-0-position-{position}.png"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["image_sha256"]
        hashes.append(digest)
    for case in range(3):
        group = hashes[case * 8 : (case + 1) * 8]
        assert group[0] == group[1] == group[4] == group[7], "Adapter A restoration differs"
        assert group[3] == group[5] == group[6], "Zero scale differs from None"
        assert len({group[0], group[2], group[3]}) == 3, "Adapter change has no observable output effect"
    assert hashes[:8] == hashes[16:], "Shape reentry changes output"
    assert len(list(directory.glob("rank-*.jsonl"))) == ranks
    for rank in range(ranks):
        observations = read_rows(directory / f"rank-{rank}.jsonl")
        assert len(observations) == 24
        for index, observation in enumerate(observations):
            expected = (1, 1, 2, None, 1, None, None, 1)[index % 8]
            assert observation["rank"] == rank and observation["active_adapter"] == expected
        if rows[0]["feature"] in ("tea_cache", "cache_dit"):
            cache = read_rows(directory / "cache" / f"rank-{rank}.jsonl")
            assert len(cache) == 24
            assert [row["request_id"] for row in cache] == [row["request_id"] for row in observations]
            hits = [row["cache_hits"] if "cache_hits" in row else len(row["cached_steps"]) for row in cache]
            if rows[0]["forced_no_hit"]:
                assert not any(hits), "Forced no-hit request skipped transformer blocks"
            else:
                assert any(hits), "Cache enabled but no actual hit observed"
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    hashes = verify(args.directory, args.ranks)
    if args.native is not None:
        assert all(row["forced_no_hit"] for row in read_rows(args.directory / "results.jsonl"))
        assert hashes == verify(args.native, args.ranks), "No-hit output differs from native"
    print(json.dumps({"directory": str(args.directory), "requests": 24, "ranks": args.ranks, "passed": True}))


if __name__ == "__main__":
    main()
