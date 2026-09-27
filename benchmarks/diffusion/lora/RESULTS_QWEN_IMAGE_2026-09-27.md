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

The same output-projection path was then tested end to end with the full
54 GB Qwen-Image checkpoint on four RTX 5090s, TP=4, BF16 and model-level
CPU offload. Eight 256 × 256 images were generated with four denoising steps:
base → adapter → adapter → base at true CFG 1 and again at true CFG 2.
The adapter was a local synthetic rank-2 adapter targeting two
`attn.to_out.0` projections; it was removed after the run. The checkpoint
was already on this host and was left untouched.

| Full-checkpoint E2E | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Valid generated images | No output-projection adapter path | 8/8, with deterministic activation and restoration | N/A — correctness fix |
| End-to-end request latency | No equivalent baseline | 5.04–6.76 s after first warmup | N/A — no paired speed comparison |

| True CFG | Base SHA256 before and after | Adapter SHA256 on both runs |
| ---: | --- | --- |
| 1 | `5e271b9ee60fee0f4e210852b162405f87d3193fb4d289e1c4517be2fff7ae1d` | `1c56f469cb9bcf8769d084dcc02b5ef25792cefc2caa21b9bbdeb209aa6c81c2` |
| 2 | `982cf1e70a0c628b3860f04333cf83eef65da78d7cc245976a1709d413f6c1f1` | `4ec48de1fb0ea36f7e18e1eba88061c4b2b51a89110c848681ef2a6c38f5096c` |

Generation completed without OOM and the process exited with status 0; the
engine logged a shutdown timeout after producing all outputs. The run used
the feature-equivalent original PR head on vLLM 0.29.0 for the same runtime
version reason noted above. It does not validate the trained Edit-R1
adapter's output quality or establish a latency improvement.
