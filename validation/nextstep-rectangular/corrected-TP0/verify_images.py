import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
archive = parser.parse_args().archive
root = Path(__file__).resolve().parent
with tarfile.open(archive) as source:
    members = source.getmembers()
    assert all(Path(member.name).name == member.name and member.isfile() for member in members)
    raw = source.extractfile("results.jsonl").read()
    assert raw == (root / "results.jsonl").read_bytes()
    for member in members:
        if member.name.endswith(".png") or member.name == "png.sha256":
            (root / member.name).write_bytes(source.extractfile(member).read())
rows = [json.loads(line) for line in raw.splitlines()]
paths = sorted(root.glob("*.png"), key=lambda path: int(path.stem.split("-")[1]))
assert len(rows) == len(paths) == 21
manifest = dict(line.split("  ", 1)[::-1] for line in (root / "png.sha256").read_text().splitlines())
verified = []
for index, (path, row) in enumerate(zip(paths, rows)):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row["image_sha256"] == manifest[path.name]
    with Image.open(path) as image:
        image.load()
        assert image.size == (row["width"], row["height"])
        assert image.mode == "RGB"
    assert int(path.stem.split("-")[2]) == row["iteration"]
    verified.append(
        {"request_index": index, "file": path.name, "sha256": digest, "width": row["width"], "height": row["height"]}
    )
assert Counter(row["sha256"] for row in verified).most_common()[0][1] == 14
report = {
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "requests": 21,
    "remote_and_local_hashes_match": True,
    "records": verified,
}
(root / "image-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "records"}))
