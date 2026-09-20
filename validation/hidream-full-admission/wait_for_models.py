# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import time
from pathlib import Path

ROOT = Path("/home/kxqandccx/omni-1217-20260919/models")
EXPECTED = {
    "hidream-i1-full": ("HiDream-ai/HiDream-I1-Full", "8ccbbfb270ccdae26d6bb0081df67dc81e4033bf"),
    "llama-3.1-8b-instruct": (
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
    ),
}


def main() -> None:
    manifests = [ROOT / slug / ".transfer-manifest.json" for slug in EXPECTED]
    while not all(path.is_file() for path in manifests):
        time.sleep(30)
    summary = []
    for slug, manifest_path in zip(EXPECTED, manifests, strict=True):
        manifest = json.loads(manifest_path.read_text())
        repo, revision = EXPECTED[slug]
        assert (manifest["repo"], manifest["revision"]) == (repo, revision)
        files = manifest["files"]
        assert files
        for row in files:
            path = ROOT / slug / row["path"]
            assert path.is_file() and path.stat().st_size == row["bytes"]
        summary.append({"slug": slug, "revision": revision, "files": len(files)})
    (ROOT / ".hidream-downloads.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / ".hidream-downloads.status").write_text("complete\n")


if __name__ == "__main__":
    main()
