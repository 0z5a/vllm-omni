# OmniGen2 language encoder FP8

OmniGen2 supports opt-in dynamic online FP8 for the language decoder under
`mllm`. Use an explicit component map:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="OmniGen2/OmniGen2",
    quantization_config={"mllm": {"method": "fp8"}},
)
```

Add `"transformer": {"method": "fp8"}` to the map to select DiT FP8 as well.
An unspecified component stays unquantized. For compatibility, OmniGen2's
existing global FP8 flag continues to affect only the DiT; it does not opt the
HF language encoder into FP8.

The adapter replaces decoder-block linear layers and respects the backend's
ignored-layer rules. Vision, embeddings, the output head, and normalization
retain their original modules and precision. The HF encoder's activation dtype
is preserved around the FP8 GEMM, including when the checkpoint loads in FP32.
Serialized FP8 encoder checkpoints and static activation scaling are rejected.

The HF checkpoint is loaded before conversion. Smaller persistent weight
allocations do not imply a smaller loading peak, allocator reservation, or
process GPU footprint. Numerical outputs change with quantization; validate
quality for your inputs. The initial full-pipeline smoke coverage is single-GPU,
eager text-to-image generation; image editing, offload combinations and
parallel encoder execution need separate qualification.


## Opt-in CUDA hot-path port (2026-09-29)

The CUDA implementation is now included in this branch. See
[dispatch, Torch fallback and measured scope](https://github.com/0z5a/vllm-omni/blob/feat/omnigen2-mllm-fp8/docs/fp8_online_e2e_followup.md).
The following results were measured on the fork stack before this upstream
port; they are not a new benchmark. Enable with
`VLLM_OMNI_FP8_ONLINE_ENABLE=1`; use
`VLLM_OMNI_FP8_FORCE_TORCH=1` for the native Torch implementation.

## Summary

Rebase the OmniGen2 `mllm` online FP8 integration from upstream #7819 onto the sm_80+ CUDA activation quantization stack (#22), then run full image generation A/P comparisons. The [quantization guide](docs/user_guide/quantization/omnigen2_fp8.md) contains the Markdown speed table and test conditions.

## Stack and provenance

- Base: `0z5a/fp8-online-quant-routing` (#22), commit `7b0e47b1`.
- Upstream model integration: https://github.com/vllm-project/vllm-omni/pull/7819.
- Test host: RTX 5090, vLLM 0.29.0, PyTorch 2.13.0+cu130, existing `0z5a` environment.

## Full E2E comparison

Both paths quantize 252 `mllm` linears online. Each arm generates six measured 512 × 512 images at 28 steps, alternating two prompts with seed 42. A and P use fresh processes; P enables the sm_80+ CUDA activation quantization path.

| Run | Warmups | Current CUDA A1 / A2 (s, mean ± SD) | sm_80+ path P1 / P2 (s, mean ± SD) | Pooled speedup | A2/A1 drift | P2/P1 drift | Same pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU 0, layer offload | 1 | 7.796 ± 1.103 / 8.853 ± 0.203 | 8.377 ± 0.365 / 7.745 ± 0.796 | 1.033× | 1.136× | 0.925× | 12/12 |
| GPU 1, resident | 1 | 8.143 ± 0.173 / 6.225 ± 2.014 | 7.161 ± 1.467 / 8.193 ± 0.222 | 0.936× | 0.765× | 1.144× | 12/12 |
| GPU 0, layer offload | 5 | 8.484 ± 0.354 / 8.375 ± 0.088 | 8.287 ± 0.136 / 8.073 ± 0.206 | **1.031×** | 0.987× | 0.974× | 12/12 |

The five-warmup repeat shows a 3.0% lower full-request latency with 1.3% A/A and 2.6% P/P drift. The one-warmup comparisons were less stable. A two-step full-image probe confirmed 504 supported CUDA per-token activation quantization calls.

## Checks

- 13 targeted OmniGen2 tests passed against the exact stacked source.
- `git diff --check` passed.
