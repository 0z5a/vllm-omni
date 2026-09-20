# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import time
from pathlib import Path

ROOT = Path("/home/kxqandccx/omni-1217-20260919/models")
EXPECTED = {
    "flux1-schnell": ("black-forest-labs/FLUX.1-schnell", "741f7c3ce8b383c54771c7003378a50191e9efe9"),
    "flux1-dev": ("black-forest-labs/FLUX.1-dev", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"),
    "flux1-kontext": ("black-forest-labs/FLUX.1-Kontext-dev", "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"),
    "sd35-medium": ("stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80"),
}


def main() -> None:
    manifests = [ROOT / slug / ".transfer-manifest.json" for slug in EXPECTED]
    while not all(path.is_file() for path in manifests):
        time.sleep(30)
    summary = []
    for slug, manifest_path in zip(EXPECTED, manifests):
        manifest = json.loads(manifest_path.read_text())
        repo, revision = EXPECTED[slug]
        assert (manifest["repo"], manifest["revision"]) == (repo, revision)
        files = manifest["files"]
        assert files
        for row in files:
            path = ROOT / slug / row["path"]
            assert path.is_file() and path.stat().st_size == row["bytes"]
        summary.append({"slug": slug, "revision": revision, "files": len(files)})
    (ROOT / ".gated-downloads.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / ".gated-downloads.status").write_text("complete\n")


if __name__ == "__main__":
    main()
