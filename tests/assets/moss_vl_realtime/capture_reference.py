# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T01 reference capture adapter for MOSS-VL-Realtime.

Runs the pinned HF reference implementation on the fixtures in
``tests/assets/moss_vl_realtime`` and records processor tensors, positions,
per-query visibility metadata, cross-attention K/V shapes, per-step logits and
control decisions under ``artifacts/moss/<revision-pair>/<case>/<run-id>/``.

This is capture tooling for T01 (reference revisions and deterministic
fixtures). It is not a runtime dependency of the model implementation and it
never imports the native vLLM-Omni executor.

Example:
    python tests/assets/moss_vl_realtime/capture_reference.py \\
        --checkpoint /path/to/MOSS-VL-Realtime \\
        --case image --case short_video \\
        --out artifacts/moss/2cb8df5c-0e016ef7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch

ASSETS = Path(__file__).resolve().parent
DEFAULT_GENERATION = {"do_sample": False, "max_new_tokens": 64}

# Media kwargs recorded by the reference offline runner. Left at processor
# defaults unless the fixture asks otherwise; resizing is captured, not hidden.
DEFAULT_MEDIA_KWARGS: dict[str, Any] = {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sha256": sha256_bytes(tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()),
    }
    if tensor.dtype.is_floating_point:
        summary["min"] = float(flat.min())
        summary["max"] = float(flat.max())
        summary["mean"] = float(flat.float().mean())
    elif tensor.numel() and tensor.dtype.is_complex is False:
        summary["min"] = int(flat.min())
        summary["max"] = int(flat.max())
    return summary


def load_cases(cases_path: Path) -> dict[str, Any]:
    return json.loads(cases_path.read_text(encoding="utf-8"))


def case_to_query(case: dict[str, Any], manifest: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    """Translate one fixture case into a reference offline query.

    Frames are passed as ordered images so no codec and no resampling enter the
    comparison; the fixture records media timestamps separately.
    """
    content: list[dict[str, Any]] = []
    for event in case["events"]:
        if event["kind"] == "prompt":
            content.append({"type": "text", "text": event["text"]})
        elif event["kind"] == "frame":
            asset = manifest["assets"][event["asset"]]
            content.append({"type": "image", "image": str((ASSETS / asset["path"]).resolve())})
        else:
            raise ValueError(f"unsupported event kind {event['kind']!r} in case {case['id']!r}")
    return {
        "messages": [{"role": "user", "content": content}],
        "media_kwargs": dict(DEFAULT_MEDIA_KWARGS),
        "generate_kwargs": dict(generation),
    }


class Capture:
    """Records the tensors and decisions a native implementation must reproduce."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events: list[dict[str, Any]] = []
        self.tensors: dict[str, torch.Tensor] = {}
        self.details: dict[str, Any] = {}
        self.forward_logits: list[torch.Tensor] = []
        self.cross_attn: dict[str, dict[str, Any]] = {}
        self.step = 0

    def reset(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events = []
        self.tensors = {}
        self.details = {}
        self.forward_logits = []
        self.cross_attn = {}
        self.step = 0

    def record(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, "t": time.time(), **fields})

    def save_tensor(self, name: str, tensor: torch.Tensor, keep_first: bool = True) -> None:
        if keep_first and name in self.tensors:
            return
        self.tensors[name] = tensor.detach().to("cpu")
        self.record("tensor", name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))


def install_hooks(model: torch.nn.Module, capture: Capture) -> None:
    inner = model.model  # MossVLModel
    cross_layers = list(model.config.text_config.cross_attention_layers)

    def wrap(obj: Any, attr: str, extract: Any) -> None:
        original = getattr(obj, attr)

        def wrapper(*args: Any, **kwargs: Any):
            out = original(*args, **kwargs)
            for name, value in extract(out).items():
                capture.save_tensor(name, value)
            return out

        setattr(obj, attr, wrapper)

    wrap(inner, "compute_position_ids", lambda out: {"text_position_ids": out})
    wrap(
        inner,
        "compute_vision_position_ids",
        lambda out: {"vision_position_ids": out[0], "shifted_text_position_ids": out[1]},
    )
    wrap(inner, "_expand_cross_attention_mask", lambda out: {"cross_attention_mask_expanded": out})

    def vision_transform(out: Any):
        embeds, info = out
        capture.details["vision_token_info"] = [
            {k: (v if not isinstance(v, torch.Tensor) else v.tolist()) for k, v in item.items()} for item in info
        ]
        return embeds

    inner.get_vision_features = _wrap_vision(inner.get_vision_features, capture, vision_transform)

    # Vision-tower staging: localises a tower mismatch to embedding, RoPE, a
    # block, or the merger instead of leaving it in the merged output.
    vision = inner.visual
    wrap(vision, "fast_pos_embed_interpolate", lambda out: {"vision.pos_embeds": out})
    wrap(vision, "rot_pos_emb", lambda out: {"vision.rotary": out})
    wrap(vision.patch_embed, "forward", lambda out: {"vision.patch_embed": out})
    for block_idx in (0, 8, 16, 24, 26):
        wrap(vision.blocks[block_idx], "forward", lambda out, idx=block_idx: {f"vision.block_{idx}": out})
    wrap(vision.merger, "forward", lambda out: {"vision.merger_out": out})
    for block_idx in (0, 8):
        wrap(vision.blocks[block_idx].norm1, "forward", lambda out, idx=block_idx: {f"vision.block_{idx}.norm1": out})
        wrap(vision.blocks[block_idx].attn, "forward", lambda out, idx=block_idx: {f"vision.block_{idx}.attn": out})
        wrap(vision.blocks[block_idx].norm2, "forward", lambda out, idx=block_idx: {f"vision.block_{idx}.norm2": out})
        wrap(vision.blocks[block_idx].mlp, "forward", lambda out, idx=block_idx: {f"vision.block_{idx}.mlp": out})

    for layer_idx in cross_layers:
        attention = inner.language_model.layers[layer_idx].cross_attn
        attention.forward = _wrap_cross_attn(attention.forward, capture, layer_idx)

    # Model-code internals that are function locals: patch the remote-code
    # module so the vision RoPE inputs and the attention output before the
    # output projection become comparable.
    remote = sys.modules[type(model).__module__]

    vision_rope = remote.apply_rotary_pos_emb_vision

    def rope_wrapper(q: Any, k: Any, cos: Any, sin: Any):
        out = vision_rope(q, k, cos, sin)
        capture.save_tensor("vision.rope_q", out[0])
        capture.save_tensor("vision.rope_k", out[1])
        capture.save_tensor("vision.rope_cos", cos)
        capture.save_tensor("vision.rope_sin", sin)
        return out

    remote.apply_rotary_pos_emb_vision = rope_wrapper

    eager_attention = remote.eager_attention_forward

    def eager_wrapper(module: Any, query: Any, key: Any, value: Any, attention_mask: Any, scaling: float, dropout: float = 0.0, **kwargs: Any):
        out = eager_attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        capture.save_tensor("vision.eager_out", out[0])
        return out

    remote.eager_attention_forward = eager_wrapper

    original_forward = model.forward

    def forward_wrapper(*args: Any, **kwargs: Any):
        out = original_forward(*args, **kwargs)
        logits = out.logits
        capture.forward_logits.append(logits.detach().to("cpu"))
        capture.record(
            "forward",
            step=capture.step,
            logits_shape=list(logits.shape),
            argmax=int(logits[0, -1].argmax()),
            max=float(logits[0, -1].max()),
        )
        capture.step += 1
        return out

    model.forward = forward_wrapper


def _wrap_vision(original, capture: Capture, transform):
    def wrapper(*args: Any, **kwargs: Any):
        out = original(*args, **kwargs)
        value = transform(out)
        if isinstance(value, torch.Tensor):
            capture.save_tensor("vision_embeds", value)
        return out

    return wrapper


def _wrap_cross_attn(original, capture: Capture, layer_idx: int):
    def wrapper(*args: Any, **kwargs: Any):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        states = kwargs.get("cross_attention_states", args[1] if len(args) > 1 else None)
        mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        out = original(*args, **kwargs)
        entry = {
            "layer": layer_idx,
            "hidden_shape": list(hidden.shape) if isinstance(hidden, torch.Tensor) else None,
            "cross_attention_states_shape": list(states.shape) if isinstance(states, torch.Tensor) else None,
            "attention_mask_shape": list(mask.shape) if isinstance(mask, torch.Tensor) else None,
            "attention_mask_dtype": str(mask.dtype) if isinstance(mask, torch.Tensor) else None,
            "masked_fraction": (
                float((mask < 0).float().mean()) if isinstance(mask, torch.Tensor) and mask.dtype.is_floating_point else None
            ),
            "output_shape": list(out[0].shape) if isinstance(out, tuple) else list(out.shape),
        }
        cache = kwargs.get("past_key_values")
        if cache is not None:
            slot = cache.layers[layer_idx]
            if getattr(slot, "keys", None) is not None:
                entry["cache_keys_shape"] = list(slot.keys.shape)
                entry["cache_values_shape"] = list(slot.values.shape)
        capture.cross_attn[f"layer_{layer_idx}"] = entry
        capture.record("cross_attn", **entry)
        return out

    return wrapper


def build_environment(checkpoint: Path, revision: str | None, attn_impl: str) -> dict[str, Any]:
    import transformers

    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": revision,
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "seed": 0,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append(
                {
                    "local_ordinal": idx,
                    "name": props.name,
                    "compute_capability": list(torch.cuda.get_device_capability(idx)),
                    "reported_total_bytes": props.total_memory,
                }
            )
    return info


def load_report(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(index.keys())
    ignored_buffers = sorted(k for k in checkpoint_keys if k.endswith("rotary_emb.inv_freq"))
    gates = {
        key: float(value.float().abs().mean())
        for key, value in model.state_dict().items()
        if key.endswith("cross_attn_attn_gate") or key.endswith("cross_attn_mlp_gate")
    }
    return {
        "checkpoint_keys": len(checkpoint_keys),
        "model_keys": len(model_keys),
        "missing_from_checkpoint": sorted(model_keys - checkpoint_keys),
        "unexpected_in_checkpoint": sorted(checkpoint_keys - model_keys - set(ignored_buffers)),
        "intentional_non_persistent_buffers": ignored_buffers,
        "cross_attention_gate_abs_mean": gates,
        "cross_attention_gates_all_zero": all(v == 0.0 for v in gates.values()),
        "tied_weights_keys": list(getattr(model, "_tied_weights_keys", []) or []),
    }


def run_case(
    model: torch.nn.Module,
    processor: Any,
    query: dict[str, Any],
    case_id: str,
    run_dir: Path,
    environment: dict[str, Any],
    load: dict[str, Any],
    capture: Capture,
) -> dict[str, Any]:
    torch.manual_seed(0)
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    model.model.vision_token_info = None
    capture.reset(run_dir)
    prepared = model.offline_prepare_query_cpu(processor, query)
    inputs_cpu = prepared["inputs_cpu"]

    inputs: dict[str, Any] = {}
    for key, value in inputs_cpu.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.detach().to("cpu")
            capture.save_tensor(f"processor.{key}", value)
        else:
            inputs[key] = value
    capture.details["input_text"] = prepared["input_text"]
    capture.details["call_kwargs"] = {k: v for k, v in prepared["call_kwargs"].items() if not isinstance(v, torch.Tensor)}

    from safetensors.torch import save_file

    run_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in inputs.items() if isinstance(v, torch.Tensor)}, run_dir / "inputs.safetensors")

    moved = model._offline_move_inputs_to_devices(inputs_cpu)
    prompt_len = int(moved["input_ids"].shape[1])

    started = time.time()
    with torch.no_grad():
        outputs = model.generate(**moved, **prepared["call_kwargs"])
    elapsed = time.time() - started

    generated = outputs[:, prompt_len:].detach().to("cpu")
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    raw_text = processor.batch_decode(generated, skip_special_tokens=False)[0]

    logits = capture.forward_logits
    per_step = [
        {
            "step": i,
            "shape": list(step_logits.shape),
            "argmax_last": int(step_logits[0, -1].argmax()),
            "max_last": float(step_logits[0, -1].max()),
            "top5_last": [int(v) for v in step_logits[0, -1].topk(5).indices],
            "sha256": sha256_bytes(step_logits[0, -1].contiguous().view(torch.uint8).numpy().tobytes()),
        }
        for i, step_logits in enumerate(logits)
    ]
    if logits:
        save_file({"prefill_logits": logits[0].contiguous()}, run_dir / "logits.safetensors")
    save_file({k: v.contiguous() for k, v in capture.tensors.items()}, run_dir / "captured_tensors.safetensors")

    contract = {
        "case": case_id,
        "processor_keys": sorted(k for k in inputs if isinstance(inputs[k], torch.Tensor)),
        "non_tensor_keys": sorted(k for k in inputs if not isinstance(inputs[k], torch.Tensor)),
        "cross_attention_frames": (
            int(inputs["cross_attention_mask"].shape[-1]) if "cross_attention_mask" in inputs else 0
        ),
        "cross_attention_layers": list(model.config.text_config.cross_attention_layers),
        "captured": {k: list(v.shape) for k, v in capture.tensors.items()},
        "cross_attn": capture.cross_attn,
        "vision_token_info": capture.details.get("vision_token_info"),
        "not_reachable_without_checkpoint_edits": [
            "int32 cross_kv_boundary (reference FA3 kernel argument; not built by this revision)",
            "post-q_norm/post-k_norm/post-RoPE Q and K (function locals)",
            "full_text_row_masked_out_mask (function local)",
            "vision tower internals beyond vision_embeds and vision_token_info",
        ],
    }
    numerical = {
        "case": case_id,
        "prompt_len": prompt_len,
        "generated_len": int(generated.shape[1]),
        "generated_token_ids": [int(v) for v in generated[0].tolist()],
        "decoded_text": text,
        "decoded_text_with_special_tokens": raw_text,
        "per_step_logits": per_step,
        "tolerances": {
            "status": "pending_repeat_run",
            "note": "absolute/relative tolerances are filled from repeated reference runs before native comparison",
        },
    }
    performance = {
        "case": case_id,
        "wall_seconds": elapsed,
        "prompt_len": prompt_len,
        "generated_tokens": int(generated.shape[1]),
        "decode_tokens_per_second": (
            float((generated.shape[1] - 1) / elapsed) if elapsed > 0 and generated.shape[1] > 1 else None
        ),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
    memory = {
        "case": case_id,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    }

    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    (run_dir / "load-report.json").write_text(json.dumps(load, indent=2) + "\n", encoding="utf-8")
    (run_dir / "inputs.json").write_text(
        json.dumps(
            {
                "case": case_id,
                "input_text": prepared["input_text"],
                "call_kwargs": capture.details["call_kwargs"],
                "tensors": {k: tensor_summary(k, v) for k, v in inputs.items() if isinstance(v, torch.Tensor)},
                "assets": query["messages"][0]["content"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "contract-report.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (run_dir / "numerical-report.json").write_text(json.dumps(numerical, indent=2) + "\n", encoding="utf-8")
    (run_dir / "performance.json").write_text(json.dumps(performance, indent=2) + "\n", encoding="utf-8")
    (run_dir / "memory.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in capture.events:
            handle.write(json.dumps(event) + "\n")

    result = [
        f"# {case_id}",
        "",
        f"- status: captured",
        f"- prompt tokens: {prompt_len}",
        f"- generated tokens: {int(generated.shape[1])}",
        f"- wall seconds: {elapsed:.2f}",
        f"- decoded text: {text!r}",
        "",
        "Reproduce:",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 python tests/assets/moss_vl_realtime/capture_reference.py \\",
        f"  --checkpoint <checkpoint> --case {case_id} --out {run_dir.parents[1]}",
        "```",
        "",
    ]
    (run_dir / "result.md").write_text("\n".join(result), encoding="utf-8")
    return {
        "case": case_id,
        "generated_tokens": int(generated.shape[1]),
        "wall_seconds": elapsed,
        "text": text,
        "cross_attention_frames": contract["cross_attention_frames"],
        "vision_embeds_shape": contract["captured"].get("vision_embeds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cases", default=str(ASSETS / "cases.json"))
    parser.add_argument("--manifest", default=str(ASSETS / "manifest.json"))
    parser.add_argument("--case", action="append", default=None, help="case id; repeatable, default all offline cases")
    parser.add_argument("--out", required=True, help="artifact root, e.g. artifacts/moss/<revision-pair>")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    fixtures = load_cases(Path(args.cases))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    generation = dict(DEFAULT_GENERATION)
    generation.update(fixtures.get("generation", {}))
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_root = Path(args.out).resolve()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from transformers import AutoModelForCausalLM, AutoProcessor

    environment = build_environment(checkpoint, args.revision, args.attn_impl)
    print(f"[capture] environment: {json.dumps(environment, indent=2)}", flush=True)

    started = time.time()
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        device_map={"": args.device},
    )
    model.eval()
    environment["load_seconds"] = time.time() - started
    load = load_report(model, checkpoint)
    print(f"[capture] load report: {json.dumps(load, indent=2)}", flush=True)

    capture = Capture(out_root)
    install_hooks(model, capture)

    selected = args.case or ["prompt_only", "image", "short_video"]
    summary = []
    for case in fixtures["cases"]:
        if case["id"] not in selected:
            continue
        query = case_to_query(case, manifest, generation)
        run_dir = out_root / case["id"] / run_id
        print(f"[capture] case {case['id']} -> {run_dir}", flush=True)
        summary.append(run_case(model, processor, query, case["id"], run_dir, environment, load, capture))

    print("[capture] summary:", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
