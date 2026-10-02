"""Verify weights and small config files against the pinned official HF Git tree."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / "models/FLUX.2-klein-4B"
    complete = json.loads((model / ".verified-complete").read_text())
    assert complete["revision"] == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    tree = {
        item["path"]: item
        for item in json.loads(
            (ROOT / "evidence/FLUX.2-klein-4B-official-git-objects.json").read_text()
        )
        if item["type"] == "file"
    }
    verified = []
    for item in complete["files"]:
        path = model / item["path"]
        official = tree[item["path"]]
        assert path.stat().st_size == official["size"] == item["size"]
        if "lfs" in official:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == official["lfs"]["oid"] == item["sha256"], item["path"]
            algorithm = "LFS SHA256"
        else:
            data = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            assert digest == official["oid"], item["path"]
            algorithm = "Git blob SHA1"
        verified.append(
            {"path": item["path"], "algorithm": algorithm, "digest": digest}
        )
    assert len(verified) == 18
    (ROOT / "evidence/klein-final-model-git-verification.json").write_text(
        json.dumps({"revision": complete["revision"], "files": verified}, indent=2)
    )
    print("Verified all18 pinned weights/configs against official Git objects")


if __name__ == "__main__":
    main()
