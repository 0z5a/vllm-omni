# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import shutil
from pathlib import Path

import hf_remote_signed_prefetch as core

REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"
REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
REMOTE_DIR = f"{core.REMOTE_ROOT}/llama-3.1-8b-instruct"
STAGE = Path("/Users/0z5a/Documents/infra/.hf-stage-1217-llama-mirror")
FILES = {
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def public_config(_token: str, url: str) -> str:
    return "\n".join(
        (
            f'url = "{url}"',
            f'proxy = "{core.PROXY}"',
            "fail",
            "retry = 20",
            "retry-all-errors",
            "retry-delay = 2",
            "connect-timeout = 30",
            "silent",
            "show-error",
        )
    )


def main() -> None:
    core.STAGE = STAGE
    core.auth_config = public_config
    STAGE.mkdir(parents=True, exist_ok=True)
    data = core.metadata("", REPO, REVISION)
    if data["sha"] != REVISION:
        raise ValueError("revision mismatch")
    siblings = data["siblings"]
    if not isinstance(siblings, list):
        raise TypeError("invalid repository inventory")
    selected = [
        item
        for item in siblings
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str) and item["rfilename"] in FILES
    ]
    if {item["rfilename"] for item in selected} != FILES:
        raise ValueError("required file set mismatch")
    core.ssh_run(f"mkdir -p {REMOTE_DIR}")
    manifest = []
    for position, item in enumerate(selected, 1):
        name = item["rfilename"]
        size = int(item["size"])
        target = f"{REMOTE_DIR}/{name}"
        lfs = item.get("lfs")
        if isinstance(lfs, dict):
            digest = lfs["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise TypeError(f"invalid digest for {name}")
            core.remote_download(core.signed_url("", REPO, REVISION, name), target, size, digest)
        else:
            digest = core.copy_small("", REPO, REVISION, name, target, size)
        manifest.append({"path": name, "bytes": size, "sha256": digest})
        print(
            json.dumps({"file": position, "of": len(selected), "path": name}),
            flush=True,
        )
    payload = json.dumps({"repo": REPO, "revision": REVISION, "files": manifest}, indent=2) + "\n"
    core.ssh_run(f"tee {REMOTE_DIR}/.transfer-manifest.json >/dev/null", payload)
    shutil.rmtree(STAGE / ".small", ignore_errors=True)
    print("Prefetch finished", flush=True)


if __name__ == "__main__":
    main()
