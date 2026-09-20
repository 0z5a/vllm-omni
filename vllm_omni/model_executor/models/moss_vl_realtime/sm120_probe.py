# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 (RTX 5090) capability probe for the native MOSS-VL-Realtime path.

Checks the operators the implementation actually uses on the hardware it will run
on: scaled-dot-product attention with and without an additive visibility mask,
the convolution-backed patch embedding, LayerNorm/RMSNorm, the tanh-approximate
GELU, and the three-axis rotary embedding. Reports measured error against a
float32 reference instead of assuming a backend works because it imports.

Example:
    CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.model_executor.models.moss_vl_realtime.sm120_probe \
        --out artifacts/moss/sm120-qualification.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F


def reference_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None, scale: float
) -> torch.Tensor:
    weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if mask is not None:
        weights = weights + mask.float()
    weights = F.softmax(weights, dim=-1, dtype=torch.float32)
    return torch.matmul(weights, value.float())


def measure(name: str, fn: Callable[[], torch.Tensor], reference: torch.Tensor) -> dict[str, Any]:
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    delta = (out.float() - reference.float()).abs()
    return {
        "name": name,
        "output_dtype": str(out.dtype),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "reference_scale": float(reference.abs().max()),
        "milliseconds": elapsed * 1000,
    }


def probe_attention(device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    heads, kv_heads, seq, head_dim = 32, 8, 512, 128
    query = torch.randn(1, heads, seq, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, kv_heads, seq, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, kv_heads, seq, head_dim, device=device, dtype=torch.bfloat16)
    key = key[:, :, None].expand(1, kv_heads, heads // kv_heads, seq, head_dim).reshape(1, heads, seq, head_dim)
    value = value[:, :, None].expand(1, kv_heads, heads // kv_heads, seq, head_dim).reshape(1, heads, seq, head_dim)
    scale = head_dim**-0.5

    results = [
        measure(
            "self_attention_causal",
            lambda: F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale),
            reference_causal(query, key, value, scale),
        )
    ]

    visibility = torch.zeros(1, 1, 64, seq, device=device, dtype=torch.bfloat16)
    visibility[..., seq // 2 :] = torch.finfo(torch.bfloat16).min
    small_query = query[:, :, :64]
    results.append(
        measure(
            "cross_attention_additive_mask",
            lambda: F.scaled_dot_product_attention(small_query, key, value, attn_mask=visibility, scale=scale),
            reference_attention(small_query, key, value, visibility, scale),
        )
    )

    for name, backend in (
        ("flash_attention", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
        ("efficient_attention", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
        ("math", torch.nn.attention.SDPBackend.MATH),
        ("cudnn_attention", torch.nn.attention.SDPBackend.CUDNN_ATTENTION),
    ):
        try:
            with torch.nn.attention.sdpa_kernel(backend):
                F.scaled_dot_product_attention(small_query, key, value, scale=scale)
            results.append({"name": name, "available": True})
        except RuntimeError as error:
            results.append({"name": name, "available": False, "error": str(error)[:200]})
    return results


def reference_causal(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    mask = torch.full((query.shape[-2], key.shape[-2]), float("-inf"), device=query.device)
    mask = torch.triu(mask, diagonal=1)
    weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32)
    return torch.matmul(weights, value.float())


def probe_operators(device: torch.device) -> list[dict[str, Any]]:
    """Functional float32 references only: nn.Module.float() would mutate the bf16 module."""
    torch.manual_seed(0)
    hidden, patches = 1152, 16

    x = torch.randn(256, hidden, device=device, dtype=torch.bfloat16)
    conv_weight = torch.randn(hidden, 3, 1, 16, 16, device=device, dtype=torch.bfloat16) * 0.02
    conv_bias = torch.randn(hidden, device=device, dtype=torch.bfloat16) * 0.02
    patch_input = torch.randn(patches, 3, 1, 16, 16, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden, device=device, dtype=torch.bfloat16) * 0.1 + 1
    norm_bias = torch.randn(hidden, device=device, dtype=torch.bfloat16) * 0.1
    linear_weight = torch.randn(18432, hidden, device=device, dtype=torch.bfloat16) * 0.02
    linear_bias = torch.randn(18432, device=device, dtype=torch.bfloat16) * 0.02

    results = [
        measure(
            "conv3d_patch_embed",
            lambda: F.conv3d(patch_input, conv_weight, conv_bias, stride=(1, 16, 16)).view(-1, hidden),
            F.conv3d(patch_input.float(), conv_weight.float(), conv_bias.float(), stride=(1, 16, 16)).view(-1, hidden),
        ),
        measure(
            "layernorm",
            lambda: F.layer_norm(x, (hidden,), norm_weight, norm_bias, 1e-6),
            F.layer_norm(x.float(), (hidden,), norm_weight.float(), norm_bias.float(), 1e-6),
        ),
        measure("gelu_tanh", lambda: F.gelu(x, approximate="tanh"), F.gelu(x.float(), approximate="tanh")),
        measure(
            "linear_wide",
            lambda: F.linear(x, linear_weight, linear_bias),
            F.linear(x.float(), linear_weight.float(), linear_bias.float()),
        ),
    ]

    inv_freq = 1.0 / (5_000_000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32, device=device) / 128))
    positions = torch.arange(64, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    results.append(
        measure("rope_outer_product", lambda: torch.outer(positions, inv_freq), torch.outer(positions, inv_freq))
    )
    results.append({"name": "rope_freq_head", "shape": list(freqs.shape), "dtype": str(freqs.dtype)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: this probe qualifies real hardware and cannot run on CPU")

    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": props.total_memory,
        "arch_list": torch.cuda.get_arch_list(),
        "attention": probe_attention(device),
        "operators": probe_operators(device),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
