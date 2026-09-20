"""Prefetch the second trained Qwen Edit adapter without modifying its tensors."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
models = (("cocoedit", "wyh6666/CoCoEdit", "be945097c8f9c3f643620f0d22ce9b30ce3aa0c8"),)

for name, repo, revision in models:
    directory = root / name
    snapshot_download(
        repo,
        revision=revision,
        local_dir=directory,
        endpoint="https://hf-mirror.com",
        allow_patterns=[
            "README.md",
            "qwen-cocoedit-lora/lora/adapter_model_converted.safetensors",
            "qwen-cocoedit-lora/lora/adapter_config.json",
            "qwen-cocoedit-lora/lora/README.md",
        ],
        max_workers=2,
    )
    (directory / "adapter_config.json").symlink_to("qwen-cocoedit-lora/lora/adapter_config.json")
    (directory / "adapter_model.safetensors").symlink_to("qwen-cocoedit-lora/lora/adapter_model_converted.safetensors")
    weight = directory / "adapter_model.safetensors"
    with safe_open(weight, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        shapes = {key: list(tensors.get_slice(key).get_shape()) for key in keys}
    config = json.loads((directory / "adapter_config.json").read_text())
    digest = hashlib.file_digest(weight.open("rb"), "sha256").hexdigest()
    assert digest == "d364540d2c555519ba4373e56bcf567e28cc7736b363aaffe082bbe4db371998"
    assert weight.stat().st_size == 377615760
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
