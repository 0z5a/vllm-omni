# Qwen-Image LoRA output projection validation

| Check | Baseline | Patched | Speedup |
| --- | ---: | ---: | ---: |
| `to_out.0` adapter loads | Rejected | 9/9 targeted CPU tests passed | N/A — correctness fix |
| Qwen-Image CPU suite | — | 43 passed, 13 deselected | N/A — correctness fix |
| Tiny-model GPU generation and adapter switching | — | Base → adapter → adapter → base passed at true CFG 1 and 2 | N/A — no timing comparison |

The CPU runs used the rebased PR head on vLLM 0.29.0. The GPU run used the
feature-equivalent original PR head `c2cb83c7` because current main's vLLM
0.30 kernel configuration does not start under the installed 0.29.0 runtime.
The output-projection loader and its test are identical across those heads.

GPU validation used an existing 5.3 MB, two-layer Qwen-Image test checkpoint
on one RTX 5090, BF16, 256 × 256, one denoising step and a locally created
synthetic rank-2 adapter targeting both `attn.to_out.0` projections. A
negative prompt was supplied for the true CFG 2 case.

| True CFG | Base SHA256 before and after | Adapter SHA256 on both runs |
| ---: | --- | --- |
| 1 | `2e799274a78bbb5a406f9633304202ba47ba6b07470ff2f71bb6257b6003a6e6` | `9b4a20bb461977359eae8bb788fd7cf06a07e737ae078a2b548f84225b858749` |
| 2 | `526974efe445b4c77a19c4fd2de492b69c7860d12219ce7c5b8acf2beb507d93` | `1322f93a17e81294f982841eb1c1168b5c6614313a3a62c61b7c8c277a763f95` |

This is full generation through the tiny pipeline, including adapter
activation and restoration. It does not validate full-size Qwen-Image GPU
injection, the trained Edit-R1 adapter's output quality, or latency speedup.
