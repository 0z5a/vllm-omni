# FP8 online quantization in vLLM-Omni: call-path map

Pinned revision: `vllm-project/vllm-omni@9f3f475685ba0a2c86cee64d12f7611f937d549b`
Recorded: 2026-09-25 · GPU: 8× RTX 5090 (SM120) · driver 580.82.07

This file answers "which code paths perform FP8 *online activation*
quantization in vLLM-Omni, and what actually executes underneath". It is the
`adapter-map.md` required by the persistent-weight PR pipeline skill, with each
capability marked `SUPPORTED` / `UNSUPPORTED` / `UNVERIFIED`.

Scope: online **activation** quantization only. Weight quantization, FP8 GEMM
kernels, KV-cache quant, and MXFP4/NVFP4 are out of scope.

---

## 1. The two entry surfaces

vLLM-Omni reaches FP8 activation quantization through exactly two surfaces.

### Surface A — `QuantFP8` CustomOp (vLLM, shared)

`vllm/model_executor/layers/quantization/input_quant_fp8.py`

`QuantFP8` is a `CustomOp`, so `__call__` dispatches on platform:
`forward_cuda` / `forward_hip` / `forward_xpu` / `forward_native`.

`forward_cuda` (line 84) branches in this order:

| # | Condition | Callee | Final kernel |
|---|---|---|---|
| A1 | `is_group_quant` and `use_ue8m0` and deep-gemm supported and `not static` and `group_size == 128` and scale-fmt is UE8M0 | `fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm` | `torch.ops._C.per_token_group_quant_fp8_packed` |
| A2 | `is_group_quant` and `not static` | `fp8_utils.per_token_group_quant_fp8` | `torch.ops._C.per_token_group_fp8_quant`, else Triton `_per_token_group_quant_fp8(_colmajor)` |
| A3 | everything else (dynamic per-token or per-tensor; or static) | `ops.scaled_fp8_quant` | see Surface B |

`forward_native` (line 181) is the pure-torch decomposition
(`abs().max()` → `scale = (x_max / 448).clamp(min=1/(448*512))` → `x/s` →
`clamp` → `.to(fp8)`), used on CPU and as the compile-mode reference.

### Surface B — direct `ops.scaled_fp8_quant` calls (vLLM-Omni)

`vllm_omni/diffusion/models/bagel/mot/mot_row_parallel_linear.py:290`
`vllm_omni/diffusion/models/bagel/mot/mot_qkv_parallel_linear.py:279`

Both are inside `_mot_gemm_fp8_w8a8`, called from `_mot_gemm_dispatch` when the
MoT (Mixture-of-Transformers) weight dtype is `torch.float8_e4m3fn`:

```python
x_2d = x.view(-1, x.shape[-1])
input_scale = getattr(self, "input_scale", None)
x_fp8, x_scale = ops.scaled_fp8_quant(
    x_2d,
    input_scale,
    use_per_token_if_dynamic=True,
)
```

Note the `getattr(self, "input_scale", None)`: when no static input scale was
loaded, this is `None`, which selects **dynamic** quantization. That is the
common case, and it is the path this work targets.

`ops.scaled_fp8_quant` (`vllm/_custom_ops.py:1900`) then dispatches:

| # | Condition | Final kernel |
|---|---|---|
| B1 | `scale is None` and `use_per_token_if_dynamic` | `torch.ops._C.dynamic_per_token_scaled_fp8_quant` |
| B2 | `scale is None` and not per-token | `torch.ops._C.dynamic_scaled_fp8_quant` |
| B3 | `scale is not None` (static) | `torch.ops._C.static_scaled_fp8_quant` |

**B1 is the hot path for FP8-quantized diffusion in vLLM-Omni.**

---

## 2. Call graph

```text
Bagel MoT block
  └─ MoTRowParallelLinear / MoTQKVParallelLinear .forward
       └─ _mot_gemm_dispatch
            ├─ weight dtype bf16/fp16 ──→ _mot_gemm_unquantized        (no quant)
            ├─ weight dtype float8_e4m3fn
            │    └─ _mot_gemm_fp8_w8a8
            │         ├─ ops.scaled_fp8_quant(x2d, None, use_per_token_if_dynamic=True)
            │         │    └─ torch.ops._C.dynamic_per_token_scaled_fp8_quant   ← target
            │         └─ invoke_mot_gemm(..., use_fp8_w8a8=True)
            ├─ weight dtype int8 ───────→ _mot_gemm_weight_only        (no act quant)
            └─ otherwise ───────────────→ _mot_fallback
```

---

## 3. Capability matrix

| Capability | Status | Evidence |
|---|---|---|
| Locate vLLM-Omni FP8 activation-quant call sites at pinned SHA | SUPPORTED | §1 line numbers from `9f3f475` |
| `torch.ops._C.dynamic_per_token_scaled_fp8_quant` present on SM120 | SUPPORTED | probed on RTX 5090, torch 2.13.0+cu130 |
| `torch.ops._C.dynamic_scaled_fp8_quant` (per-tensor) present | SUPPORTED | probed |
| `torch.ops._C.per_token_group_fp8_quant` present | SUPPORTED | probed; not exercised by MoT |
| Triton fallback for group quant reachable | UNVERIFIED | only taken when CUDA kernel absent or non-contiguous |
| Helion per-token kernel active | UNSUPPORTED | `helion` not installed in this env (`vllm/kernels/helion/...` raises ImportError) |
| quack FP8 GEMM on SM120 | UNSUPPORTED | `quack_fp8._is_quack_capable()` requires capability 10.x (tcgen05); SM120 falls back to FlashInfer |
| `input_scale` static path exercised by MoT | UNVERIFIED | requires a checkpoint that ships a static input scale |
| Real BAGEL FP8 model weights available on this host | UNVERIFIED | not downloaded at time of writing |

### Why `dir()` is not a valid availability check here

`torch.ops._C` is a lazy namespace: `dir(torch.ops._C)` omits ops that have not
been touched yet. `dynamic_per_token_scaled_fp8_quant` is absent from `dir()`
but `getattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant")` succeeds. Any
dispatch code that feature-detects with `hasattr`/`dir` on this namespace is
unreliable.

---

## 4. Frozen numeric contract of the target kernel (B1)

Derived empirically by comparing the compiled kernel against candidate formulas
over 937 rows; see `results/contract_probe.json` and
`results/scale_formula_probe.json`.

```text
input   x        : [M, K] contiguous, bf16 / fp16 / fp32
output  out      : [M, K] float8_e4m3fn
scale   scale    : [M, 1] float32

amax      = max_j |x[r, j]|                       # fp32 comparison
scale[r]  = (double)amax / 448.0                  # ← DOUBLE precision
            clamped from below to 1/(448*512)
            rounded to float32
out[r, j] = e4m3( x[r, j] / scale[r] )            # fp32 division
```

with `e4m3(v) = clamp(v, -448, 448)` followed by round-to-nearest,
**ties away from zero** and saturation.

Two details are load-bearing and are easy to get wrong:

1. **The scale divide is a double-precision divide.** Computing
   `amax * (1/448)` or `amax / 448` in fp32 differs from the reference on ~50%
   of rows by 1 ULP. That ULP then changes payload bytes, i.e. it silently
   changes decoded activations. Measured: 477 float32 mismatches out of 937
   rows, 0 mismatches with the double divide.

2. **Tie-breaking is ties-away-from-zero, not ties-to-even.** PTX
   `cvt.rn.satfinite.e4m3x2.f32` rounds exact midpoints away from zero, whereas
   `torch.Tensor.to(torch.float8_e4m3fn)` rounds them to even. For example
   `y = -116` is exactly between the E4M3 codes `0xee` (-116.0) and `0xef`
   (-120.0): the reference kernel emits `0xef`, a torch cast emits `0xee`.
   PyTorch casts are therefore **not** a valid oracle for this kernel.

Contract for the per-tensor kernel (B2): same, minus the `min_scale` floor.

---

## 5. Notes on the FP8 W8A8 MoT GEMM

`_mot_gemm_fp8_w8a8` passes `A_per_channel_quant=True` and hands `A_scale`
(= the per-token `x_scale`) to `invoke_mot_gemm`, so the quantizer's scale layout
is part of the GEMM contract. Any change to the scale tensor's shape, stride, or
dtype would break the consumer even if the payload matched. The fast path
preserves the `[M, 1]` fp32 contiguous layout exactly.
