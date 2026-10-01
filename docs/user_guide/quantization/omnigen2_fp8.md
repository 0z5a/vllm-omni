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
quality for your inputs. Image editing and parallel encoder execution need
separate qualification.

### RTX 5090 full-pipeline CUDA comparison

The following measurements use the FP8 CUDA routing stack from #22 with 252
OmniGen2 `mllm` decoder linears quantized online. Each arm starts a fresh
process and generates 512 × 512 images at 28 diffusion steps and guidance 4.
Two text prompts alternate with seed 42. `A` disables the sm_80+ CUDA activation
quantization path; `P` enables it. Both paths use the same FP8 weights.

| RTX 5090 run | Warmups | Current CUDA A1 / A2 (s, mean ± SD) | sm_80+ path P1 / P2 (s, mean ± SD) | Pooled A/P speedup | A2/A1 drift | P2/P1 drift | Paired images |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU 0, layer offload | 1 | 7.796 ± 1.103 / 8.853 ± 0.203 | 8.377 ± 0.365 / 7.745 ± 0.796 | 1.033× | 1.136× | 0.925× | 12/12 exact |
| GPU 1, resident | 1 | 8.143 ± 0.173 / 6.225 ± 2.014 | 7.161 ± 1.467 / 8.193 ± 0.222 | 0.936× | 0.765× | 1.144× | 12/12 exact |
| GPU 0, layer offload | 5 | 8.484 ± 0.354 / 8.375 ± 0.088 | 8.287 ± 0.136 / 8.073 ± 0.206 | **1.031×** | 0.987× | 0.974× | 12/12 exact |

Values are means of six full requests per arm. The one-warmup runs had
within-path drift larger than their A/P differences. With five warmups, the
full-pipeline speedup was 1.031× (3.0% lower latency), A/A drift was 1.3%,
P/P drift was 2.6%, and all 12 paired images were pixel-identical. A separate
two-step full-image path probe saw 504 CUDA per-token
quantization calls across four supported BF16 input shapes. It also confirmed
that the extension loaded successfully. The measurement host used vLLM 0.29.0,
PyTorch 2.13.0+cu130, and a benchmark-only compatibility shim for the removed
`gelu_and_mul_sparse` IR priority field.
