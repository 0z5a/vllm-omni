# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare native inputs for MOSS-VL-Realtime without running the reference model.

The capture adapter builds model inputs through the checkpoint's own offline
helper, which needs the full model instantiated. This module mirrors that helper
for a single user turn so prompts and frames can be prepared, replayed and
compared without loading 22.7 GB of weights first. `--verify` checks the mirror
against a captured reference run: identical token ids and bit-identical pixels
mean the two paths agree, and any drift is visible before it reaches a model.

Example:
    python tests/assets/moss_vl_realtime/prepare_inputs.py \
        --checkpoint /path/to/MOSS-VL-Realtime \
        --prompt "Describe the image." --image red.ppm \
        --out artifacts/moss/native-inputs/image

    python tests/assets/moss_vl_realtime/prepare_inputs.py --checkpoint ... \
        --case image --verify-against artifacts/moss/2cb8df5c-0e016ef7e/image/<run-id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ASSETS = Path(__file__).resolve().parent
IMAGE_PLACEHOLDER = "<|image|>"
VIDEO_PLACEHOLDER = "<|video|>"


def sanitize_prompt_text(processor: Any, text: Any) -> str:
    if text is None:
        return ""
    sanitized = str(text)
    for needle in (
        getattr(processor, "image_placeholder", None),
        getattr(processor, "video_placeholder", None),
        getattr(processor, "image_token", None),
        getattr(processor, "video_token", None),
    ):
        if needle:
            sanitized = sanitized.replace(needle, "")
    return sanitized.lstrip("\n")


def flatten_content(processor: Any, content: Any) -> str:
    if isinstance(content, str):
        return sanitize_prompt_text(processor, content)
    if not isinstance(content, list):
        return sanitize_prompt_text(processor, content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            if isinstance(item, str):
                parts.append(sanitize_prompt_text(processor, item))
            continue
        if item.get("type") == "image" or "image" in item:
            parts.append(IMAGE_PLACEHOLDER)
        elif item.get("type") == "video" or "video" in item:
            parts.append(VIDEO_PLACEHOLDER)
        if "text" in item:
            parts.append(sanitize_prompt_text(processor, item["text"]))
    return "".join(parts)


def build_messages(processor: Any, prompt: str, images: list[str], videos: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": path} for path in images)
    content.extend({"type": "video", "video": path} for path in videos)
    flattened = [{"role": "user", "content": flatten_content(processor, content)}]
    return flattened


def prepare(processor: Any, prompt: str, images: list[str], videos: list[str]) -> tuple[Any, str]:
    messages = build_messages(processor, prompt, images, videos)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=False,
    )
    return inputs, text


def case_to_prompt(case: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, list[str]]:
    prompt = " ".join(event["text"] for event in case["events"] if event["kind"] == "prompt")
    images = [
        str((ASSETS / manifest["assets"][event["asset"]]["path"]).resolve())
        for event in case["events"]
        if event["kind"] == "frame"
    ]
    return prompt, images


def verify(inputs: Any, reference_dir: Path) -> dict[str, Any]:
    with safe_open(reference_dir / "inputs.safetensors", framework="pt") as handle:
        reference = {key: handle.get_tensor(key) for key in handle.keys()}
    report: dict[str, Any] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor) or key not in reference:
            continue
        expected = reference[key]
        same_shape = tuple(value.shape) == tuple(expected.shape)
        report[key] = {
            "shape": list(value.shape),
            "reference_shape": list(expected.shape),
            "dtype": str(value.dtype),
            "exact": same_shape and torch.equal(value, expected),
            "max_abs": float((value.float() - expected.float()).abs().max()) if same_shape else None,
        }
    report["all_exact"] = all(entry["exact"] for key, entry in report.items() if isinstance(entry, dict))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image", action="append", default=None)
    parser.add_argument("--video", action="append", default=None)
    parser.add_argument("--case", default=None, help="fixture case id to prepare instead of a free-form prompt")
    parser.add_argument("--cases", default=str(ASSETS / "cases.json"))
    parser.add_argument("--manifest", default=str(ASSETS / "manifest.json"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify-against", default=None)
    args = parser.parse_args()

    prompt = args.prompt
    images = list(args.image or [])
    videos = list(args.video or [])
    if args.case:
        fixtures = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        case = next(item for item in fixtures["cases"] if item["id"] == args.case)
        prompt, images = case_to_prompt(case, manifest)
    if prompt is None:
        raise SystemExit("provide --prompt or --case")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    inputs, text = prepare(processor, prompt, images, videos)

    report: dict[str, Any] = {"prompt": prompt, "images": images, "rendered_text": text}
    if args.verify_against:
        report["verification"] = verify(inputs, Path(args.verify_against).resolve())
    if args.out:
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        tensors = {key: value.contiguous() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        save_file(tensors, out_dir / "inputs.safetensors")
        (out_dir / "inputs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
