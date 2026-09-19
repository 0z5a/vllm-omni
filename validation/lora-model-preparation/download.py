"""Prefetch two immutable public trained adapters for the K5 matrix."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
models = (
    ("helio", "HelioAI/Helio-Z-Image-Turbo-LoRA", "38826969e8cb78f91d963a1cf7910e7d14749363"),
    ("fuli", "DownFlow/Z-Image-Turbo-Fuli-LoRA", "b1216f23d98df471e7b7c12dc662dae958b2c187"),
)
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
