# FP8 online quantization: frozen numeric contract

This document records the exact semantics of the activation-quantization kernel
that vLLM-Omni's diffusion MoT layers call today, and which the CUDA fast path
in `vllm_omni/quantization/fp8_online.py` must reproduce **bit for bit**.

Reference: `torch.ops._C.dynamic_per_token_scaled_fp8_quant`, reached through
`vllm._custom_ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)`.

Pinned environment: vLLM 0.29.0, torch 2.13.0+cu130, CUDA 13.0, RTX 5090
(sm_120), driver 580.82.07. Evidence: `results/contract_probe.json`,
`results/scale_formula_probe.json`.

## Per-token (the hot path)

```text
input    x        [M, K]  contiguous, bf16 | fp16 | fp32
output   out      [M, K]  float8_e4m3fn
scale    scale    [M, 1]  float32

amax      = max_j |x[r, j]|                     # compared in fp32
scale[r]  = (double)amax / 448.0                # DOUBLE-precision divide
            clamped from below to 1/(448*512)
            rounded to float32
out[r, j] = e4m3_rn( x[r, j] / scale[r] )       # fp32 divide
```

Then `e4m3_rn(v)` clamps `v` to `[-448, 448]` and converts to E4M3 with
round-to-nearest, **ties away from zero**, saturating.

## Per-tensor

Identical, except there is no `min_scale` floor.

## Two details that are easy to get wrong

### 1. The scale divide is a double-precision divide

Measured over 937 randomly shaped rows, comparing the compiled kernel's stored
`scale` against candidate expressions:

| candidate scale expression | rows differing from the kernel |
|---|---:|
| `amax * (1.0f/448.0f)` then clamp (fp32) | 477 / 937 |
| `amax / 448.0f` then clamp (fp32) | 477 / 937 |
| `amax.clamp(min) * (1/448)` (fp32) | 477 / 937 |
| `1.0f / (448.0f / amax)` (fp32) | 316 / 937 |
| `(amax.double() / 448.0)` then clamp, cast to fp32 | **0 / 937** |

So roughly half of all rows differ by 1 ULP if the divide is done in fp32. That
ULP is not cosmetic: it changes which E4M3 code some values round to, i.e. it
changes the quantized payload the GEMM consumes. Any reimplementation must keep
the divide in double (as the reference kernel does).

### 2. Tie-breaking is ties-away-from-zero, not ties-to-even

`cvt.rn.satfinite.e4m3x2.f32` rounds an exact midpoint **away from zero**.
`torch.Tensor.to(torch.float8_e4m3fn)` rounds midpoints **to even**. These are
different functions.

Worked example: `y = -116.0` is exactly halfway between the E4M3 codes
`0xee` (`-116.0`) and `0xef` (`-120.0`).

* reference kernel → `0xef` (away from zero, magnitude increases)
* torch fp8 cast → `0xee` (to even: mantissa 4 is even)

Consequence: **a torch fp8 cast is not a valid oracle** for this kernel. Testing
against one produces a couple of dozen "mismatches" per million bytes that look
like rounding noise but are actually a different rounding rule. The test suite
therefore compares against the compiled kernel itself, and the standalone C++
harness carries its own deliberately independent E4M3 encoder that implements
the away-from-zero rule.

## Behaviour on special values

Recorded from the compiled kernel, finite-fixture rows only:

| input | resulting scale | resulting payload |
|---|---|---|
| all `+0` | `1/(448*512)` (the floor) | `0x00` |
| all `-0` | `1/(448*512)` (the floor) | `0x80` (negative zero preserved) |
| all `2^-30` | `1/(448*512)` (the floor) | `0x00` |
| single `300.0` outlier in zeros | `300/448` | `0x00` elsewhere |
| contains `+inf` | `+inf` | `0x7e` (saturated) |
| contains one NaN among finite | finite, NaN ignored by the max | NaN position → `0x7e` |

The `amax` reduction ignores NaN, matching a `max`-style reduction. Values that
overflow the scale (`amax` extreme) saturate rather than produce NaN. The fast
path reproduces all of these (see `test_fp8_online_quant.py`).

## Scale layout contract

`_mot_gemm_fp8_w8a8` passes the returned scale to `invoke_mot_gemm` with
`A_per_channel_quant=True`. The scale must therefore remain a contiguous
`[M, 1]` fp32 tensor; changing its shape, stride, or dtype would break the GEMM
consumer even if the payload matched. The fast path preserves this layout
exactly and never returns a strided or reshaped scale.
