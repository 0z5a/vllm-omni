# Qwen2.5-Omni CUDA FP8 follow-up


## Opt-in CUDA hot-path port (2026-09-29)

The CUDA implementation is now included in this branch. See
[dispatch, Torch fallback and measured scope](https://github.com/0z5a/vllm-omni/blob/codex/qwen25-fp8-compiled-ar/docs/fp8_online_e2e_followup.md).
The following results were measured on the fork stack before this upstream
port; they are not a new benchmark. Enable with
`VLLM_OMNI_FP8_ONLINE_ENABLE=1`; use
`VLLM_OMNI_FP8_FORCE_TORCH=1` for the native Torch implementation.

## Summary

Replay upstream Qwen2.5-Omni compiled CUDA AR support (#8126) onto the online FP8 CUDA stack. Parent draft #23 carries the component routing from upstream #3466; this draft is stacked on #23, which is stacked on #22.

## Full E2E CUDA result

`Qwen/Qwen2.5-Omni-3B` revision `f0fe4326288865534ad09d70aa435ff7d1611541`, three RTX 5090 GPUs, full text→speech request, 1,978 prompt tokens, seed 42, stage-0 gate/up online FP8, eager execution. Each fresh-process arm used 3 warmups and 6 measured requests. A uses the existing CUDA activation quantizer; P uses the sm_80+ CUDA hot path.

| Arm | CUDA activation quant | Full E2E mean ± SD (s) | Thinker completion mean ± SD (s) | Full speedup vs pooled A |
|---|---|---:|---:|---:|
| A1 | current CUDA | 13.059 ± 0.231 | 1.079 ± 0.162 | 0.995× |
| P1 | sm_80+ hot path | 12.488 ± 0.351 | 0.924 ± 0.022 | 1.040× |
| P2 | sm_80+ hot path | 12.531 ± 0.127 | 0.982 ± 0.022 | 1.037× |
| A2 | current CUDA | 12.927 ± 0.483 | 1.044 ± 0.149 | 1.005× |

Pooled speedup: **1.039× full E2E**, **1.113× Thinker completion**. A2/A1 drift: 0.990×; P2/P1 drift: 1.004×. All 24 measured A/P/P/A requests returned text and finite 10.24-second audio.

Only 6/12 paired text and audio outputs were exact because runs branched between 27- and 28-token text. Both paths produced both variants. In an additional full-request hot-path diagnostic, FP8 activation outputs and scales matched the reference CUDA quantizer bit for bit for the two observed BF16 shapes, `(16, 2048)` and `(1, 2048)`. Full details are in `docs/fp8_online_e2e_qwen25.md`.

The installed vLLM 0.29.0 compiled `QuantFP8.forward_native` path bypasses #22's `QuantFP8.forward_cuda` hook, so these hot-path measurements use eager execution. Compiled mode needs a separate integration change before its requests can measure this CUDA extension.

## Checks

- 17 targeted config and scheduler tests passed in the existing 0z5a Python environment.
- Python syntax compilation and `git diff --check` passed.
