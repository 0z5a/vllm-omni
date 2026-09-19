"""Prefetch the immutable trained Qwen Edit adapter for K5 CFG admission."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
models = (("edit-r1", "chestnutlzj/Edit-R1-Qwen-Image-Edit-2509", "6f58a70502ea6560da338c5af91d7a3d156077ee"),)

for name, repo, revision in models:
    directory = root / name
    snapshot_download(
        repo,
        revision=revision,
        local_dir=directory,
        endpoint="https://hf-mirror.com",
        allow_patterns=["adapter_model.safetensors", "adapter_config.json", "README.md"],
        max_workers=2,
    )
    weight = directory / "adapter_model.safetensors"
    with safe_open(weight, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        shapes = {key: list(tensors.get_slice(key).get_shape()) for key in keys}
    config = json.loads((directory / "adapter_config.json").read_text())
    digest = hashlib.file_digest(weight.open("rb"), "sha256").hexdigest()
    report = {
        "repo": repo,
        "revision": revision,
        "bytes": weight.stat().st_size,
        "sha256": digest,
        "config": config,
        "tensor_shapes": shapes,
    }
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": name,
                "revision": revision,
                "bytes": report["bytes"],
                "sha256": digest,
                "tensors": len(keys),
                "rank": config["r"],
            }
        ),
        flush=True,
    )
