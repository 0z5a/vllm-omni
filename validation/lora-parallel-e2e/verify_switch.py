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
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    verify(args.directory, args.ranks)
    print(json.dumps({"directory": str(args.directory), "requests": 24, "ranks": args.ranks, "passed": True}))


if __name__ == "__main__":
    main()
