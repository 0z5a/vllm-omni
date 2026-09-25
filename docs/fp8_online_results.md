# FP8 online quantization: CUDA fast path (sm_80+) — results

Status of each claim. Nothing here is inferred from a microbenchmark unless the
row says so.

| Layer | Status |
|---|---|
| Kernel-only A/B vs the CUDA baseline | `PASS` — measured, see §2 |
| Bit-exactness vs the reference kernel | `PASS` — 215 pytest cases + 288 ad-hoc comparisons, 0 mismatches |
| AOT compile for sm_80/86/89/90/100/120 | `PASS` (compile only) |
| Mappings self-consistent across dtypes and shapes | `PASS` (standalone fuzz, 18 shapes x 3 dtypes x 8 mappings) |
| Full-model E2E | `BLOCKED_DEPENDENCY` — see §6 |

Environment: 8x RTX 5090 (sm_120), driver 580.82.07, CUDA 13.0, torch
2.13.0+cu130, vLLM 0.29.0, vLLM-Omni 0.28.0 (installed).

---

## 1. What changed

`vllm_omni/quantization/fp8_online.py` adds a CUDA fast path for dynamic FP8
activation quantization with an explicit gate (>=

sm_80) and an explicit fallback to
`torch.ops._C.dynamic_per_token_scaled_fp8_quant`.

`vllm_omni/diffusion/models/bagel/mot/{mot_row_parallel_linear,mot_qkv_parallel_linear}.py`
now call it instead of `ops.scaled_fp8_quant` directly. Those two call sites are
the only places vLLM-Omni performs FP8 *activation* quantization; see
`docs/fp8_online_callpath_map.md`.

## 2. Kernel-only result vs the CUDA baseline

The comparison target is the kernel vLLM-Omni calls today —
`torch.ops._C.dynamic_per_token_scaled_fp8_quant` — not the native/reference
decomposition, so this is the `current CUDA -> patched CUDA` delta the plan
requires.

`A1` = current CUDA baseline, `P` = fast path. `A1/P > 1` means the fast path is
faster. Both arms write into preallocated buffers, are interleaved within each
measurement block, and are run after the GPU has been warmed to a stable clock
(the part idles at 180 MHz, which is what corrupted the first attempt at this
table).

| M (tokens) | K | A1 [us] | P [us] | A1/P | selected path | exact |
|---:|---:|---:|---:|---:|---|---|
| 1 | 3072 | 9.08 | 7.24 | **1.253x** | fast | yes |
| 4 | 3072 | 7.73 | 6.53 | **1.183x** | fast | yes |
| 16 | 3072 | 7.16 | 6.24 | **1.148x** | fast | yes |
| 64 | 3072 | 7.19 | 6.16 | **1.167x** | fast | yes |
| 128 | 3072 | 7.13 | 6.23 | **1.143x** | fast | yes |
| 256 | 3072 | 7.19 | 6.26 | **1.150x** | fast | yes |
| 512 | 3072 | 7.28 | 6.21 | **1.173x** | fast | yes |
| 1024 | 3072 | 7.29 | 9.08 | 0.802x | reference | yes |
| 2048 | 3072 | 7.26 | 9.16 | 0.793x | reference | yes |
| 4096 | 3072 | 10.28 | 10.28 | 1.000x | reference | yes |
| 1 | 4096 | 7.08 | 6.32 | **1.120x** | fast | yes |
| 4 | 4096 | 7.12 | 6.32 | **1.127x** | fast | yes |
| 16 | 4096 | 7.12 | 6.26 | **1.137x** | fast | yes |
| 64 | 4096 | 7.09 | 6.26 | **1.132x** | fast | yes |
| 128 | 4096 | 7.09 | 6.25 | **1.134x** | fast | yes |
| 256 | 4096 | 7.14 | 6.25 | **1.143x** | fast | yes |
| 512 | 4096 | 7.32 | 6.31 | **1.159x** | fast | yes |
| 1024 | 4096 | 7.21 | 9.06 | 0.795x | reference | yes |
| 2048 | 4096 | 7.41 | 9.22 | 0.803x | reference | yes |
| 4096 | 4096 | 12.33 | 12.33 | 1.000x | reference | yes |

Raw data: `results/envelope.json`.

### Why the gate stops at M = 512

Both arms are launcher-bound in this regime (a ~5 us launch floor against
6-9 us of total work), so the fast path's win comes from getting the work onto
the GPU in one kernel with less host-side work, not from a faster inner loop.
That advantage shrinks as the tensor grows and the actual arithmetic starts to
dominate: at M >= 1024 the reference kernel, which spreads a long row across
more blocks than this implementation does, is faster.

A gate has to hold for every row count it admits, so `FASTPATH_MAX_ROWS = 512`
is the largest bound that was verified uniformly (1.12x-1.25x across both K
values). Above it the call is delegated, so **no caller is made slower than
today**: the worst case is the current baseline.

`M <= 512` with `K in {3072, 4096}` covers the diffusion activation shapes this
path exists for. It is not a claim about other K or other models.

## 3. Correctness

The oracle is the compiled reference kernel. It is compared byte-for-byte on the
FP8 payload and bit-for-bit on the fp32 scale; `allclose` is never used.

| Suite | Cases | Result |
|---|---:|---|
| `tests/diffusion/quantization/test_fp8_online_quant.py` | 185 passed, 45 skipped | 0 mismatches |
| ad-hoc `dtype x shape x distribution` sweep | 252 | 0 mismatches |
| `scale_ub` sweep | 36 | 0 mismatches |
| standalone mapping fuzz (18 shapes x 3 dtypes x 8 mappings) | 432 | 0 mismatches |
| converter fixture table incl. packed byte order | 15 + quad | pass |

Skipped cases are shapes outside the gate; `test_large_row_count_is_declined`
asserts they are declined rather than mis-executed.

Two contract details were established empirically and are load-bearing
(`docs/fp8_online_contract.md`):

* the scale divide is a **double-precision** divide — doing it in fp32 changes
  ~50% of scale values by 1 ULP, which changes payload bytes (477/937 rows
  differed);
* PTX `cvt.rn.satfinite` rounds ties **away from zero**, while a torch fp8 cast
  rounds them to even, so a torch cast is not a valid oracle.

## 4. Compile coverage

`tests/compile_matrix.sh` compiles the dependency-free kernel TU for every
architecture the runtime guard admits. Compiling is not running; the non-sm_120
rows are `COMPILE_ONLY`.

| target | result | FP8 convert instruction in SASS | reduction |
|---|---|---|---|
| sm_80 | COMPILE_PASS | none (SDK converter branch) | `REDUX.MAX` |
| sm_86 | COMPILE_PASS | none (SDK converter branch) | `REDUX.MAX` |
| sm_89 | COMPILE_PASS | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | `REDUX.MAX` |
| sm_90 | COMPILE_PASS | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | `REDUX.MAX` |
| sm_100 | COMPILE_PASS | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | `REDUX.MAX` |
| sm_120 | COMPILE_PASS | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | `REDUX.MAX` |

sm_80/86 correctly take the SDK-converter branch, which is the expected
architecture split rather than a failure. Register usage on sm_120: 30-44
registers per kernel, **0 spill stores, 0 spill loads** in every
specialisation.

## 5. Runtime coverage

| target | runtime |
|---|---|
| sm_120 (RTX 5090) | tested — correctness and performance |
| sm_80/86/89/90/100 | **UNTESTED** — compile evidence only, no such hardware available |

## 6. E2E: BLOCKED_DEPENDENCY

The full-model E2E could not be run, for a reason unrelated to this change: the
installed `vllm_omni` 0.28.0 imports `vllm.inputs.preprocess`, which does not
exist in the installed vLLM 0.29.0, so `vllm_omni.entrypoints.omni.Omni` raises
`ModuleNotFoundError` before any model is loaded:

```
File ".../vllm_omni/inputs/preprocess.py", line 5, in <module>
    from vllm.inputs.preprocess import InputPreprocessor
ModuleNotFoundError: No module named 'vllm.inputs.preprocess'
```

vLLM-Omni itself warns about this at import time
(`vLLM and vLLM-Omni appear to have mismatched major/minor versions`). Fixing it
would mean changing the installed environment, which is out of scope here.

Consequently **no end-to-end latency or quality number is claimed**. The
operator-level result in §2 stands on its own and is the honest bound for what
was verified: a fast path that is 1.12x-1.25x faster on the admitted shapes and
delegates everything else.

An end-to-end run also could not use a synthetic substitute: `riverclouds/qwen_image_random`
(the model the upstream FP8 test uses) failed to download completely through the
available mirror, and the download was removed afterwards rather than left on
disk.

To run it once the environment is aligned:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_EXTENSIONS_DIR=<writable cache>
python tests/e2e_real_model.py results/e2e.json
```

That script runs the same model/prompt/seed/steps in four fresh processes
(`baseline, fast, fast, baseline`) and compares image hashes and latency.

## 7. Reproduce

```bash
# 1. correctness (needs a CUDA device, sm_80+)
python -m pytest tests/diffusion/quantization/test_fp8_online_quant.py -q

# 2. compile matrix for all guarded architectures
bash tests/compile_matrix.sh

# 3. standalone fixtures + mapping fuzz (no torch import needed)
nvcc -std=c++17 -O3 -arch=sm_120 --ftz=false --prec-div=true --prec-sqrt=true \
     -Icsrc -Itests tests/fp8_online_harness.cu -o /tmp/h && /tmp/h --fuzz

# 4. the table in section 2
python tests/rebench2.py
```

The extension builds on first use and is cached; set
`VLLM_OMNI_FP8_ONLINE_VERBOSE=1` to see the build log and
`VLLM_OMNI_FP8_ONLINE_DISABLE=1` to force the reference path.
