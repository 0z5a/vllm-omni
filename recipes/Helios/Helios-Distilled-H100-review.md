# Helios dense attention: independent H100 reproduction

This review reproduces [#8206](https://github.com/vllm-project/vllm-omni/pull/8206)
on two H100 80GB HBM3 cards, each running an isolated single-GPU pipeline.
FA3 lowers median request latency by **8.76–9.53%** and cuDNN by **6.89–7.61%**
versus SDPA. Both cards agree on the ordering; peak reserved memory is unchanged.
These are additional SM90 portability measurements, not replacements for the H20 results.

The same-workload and backend-identity gates pass. Final videos differ across
backends: the lowest mean frame SSIM is **0.89061** for FA3 and **0.91796** for
cuDNN. Neither result supports a pixel-equivalence claim. Both are lower than
the H20 minima (0.93431 and 0.94409); those observed envelopes are not quality
acceptance thresholds. The cause of the larger final-video divergence is not
isolated by this reproduction. Synthetic operator checks pass, which does not
establish equivalence for every attention tensor in the full pipeline.

## Provenance and execution

- Reviewed head: `fba8d2e2a123e8f808376ed784e4f19356789f36`.
- Recorded H20 runtime: `b63e35ab4ffae2f78b556150943ec0e08a444030`.
  The intervening diff changes benchmark, tests, and documentation; no production
  runtime implementation changes.
- Model: `BestWishYsh/Helios-Distilled`, revision
  `b991c0379a018f4de3227d95468237f56066f5bb`, excluding `transformer_ode/*`.
- Unmodified benchmark SHA256:
  `47daf1043c80b39d1aa44b4425416e6802a70ec103f90aeaad082ebd4fbee776`.
- Unmodified comparator SHA256:
  `bbda025672e85d8b8212361986d8aacdfc9c57cfdea719514449473c979aa658`.
- Both cards are SM90, with driver 595.71.05, 81,559 MiB visible memory and
  a 700 W power limit. GPU UUIDs are recorded in the JSON evidence.
- Python 3.12.3, PyTorch 2.13.0+cu130, CUDA runtime 13.0, cuDNN 9.20.0,
  vLLM 0.30.0, Transformers 5.14.1, Diffusers 0.40.0, Triton 3.7.1,
  kernels 0.16.1, fa3-fwd 0.0.3.
- 640×384; 33/66 frames; BF16 DiT/text encoder; FP32 VAE; eager execution;
  no offload; guidance 1.0; pyramid `[2,2,2]`; first-chunk amplification.
- Seeds 42, 7, 123; three repetitions; one excluded warmup per frame count.
  Each backend/card pair contains 18 measured requests and two warmups.
- All six performance jobs ran sequentially. Card 0: SDPA → FA3 → cuDNN.
  Card 1: cuDNN → FA3 → SDPA. No GPU training overlapped these measurements.
- An external launcher set `DIFFUSION_ATTENTION_BACKEND` before importing the
  unchanged benchmark and recorded provider identity in the same process.
  Both FlashAttention functions resolved to `fa3_fwd_interface` in all six jobs.
  The inline single-process diffusion executor binds the measured implementation
  in that process; wrapper inventories independently match each requested backend.
- Each request resets the existing Helios cross-attention cache before the wall
  timer. Results cover isolated requests, not mixed-prompt serving without reset.

## Request latency and speedup

Speedup = SDPA median / backend median. Reduction = 1 − backend / SDPA.
Every row contains nine measured requests. Transformer timing uses CUDA events;
request timing covers `Omni.generate()` and excludes startup and cache reset.

| GPU | Frames | Backend | N | Wall ms | Transformer ms | Speedup | Latency reduction | Peak reserved MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h100-0 | 33 | TORCH_SDPA | 9 | 7774.047 | 6924.038 | 1.000× | +0.00% | 47094 |
| h100-0 | 33 | FLASH_ATTN | 9 | 7092.939 | 6204.726 | 1.096× | +8.76% | 47094 |
| h100-0 | 33 | CUDNN_ATTN | 9 | 7182.478 | 6345.851 | 1.082× | +7.61% | 47094 |
| h100-0 | 66 | TORCH_SDPA | 9 | 12002.159 | 10379.608 | 1.000× | +0.00% | 47590 |
| h100-0 | 66 | FLASH_ATTN | 9 | 10924.244 | 9303.990 | 1.099× | +8.98% | 47590 |
| h100-0 | 66 | CUDNN_ATTN | 9 | 11101.414 | 9520.418 | 1.081× | +7.50% | 47590 |
| h100-1 | 33 | TORCH_SDPA | 9 | 7709.418 | 6869.603 | 1.000× | +0.00% | 47094 |
| h100-1 | 33 | FLASH_ATTN | 9 | 6975.048 | 6143.477 | 1.105× | +9.53% | 47094 |
| h100-1 | 33 | CUDNN_ATTN | 9 | 7153.218 | 6292.494 | 1.078× | +7.21% | 47094 |
| h100-1 | 66 | TORCH_SDPA | 9 | 11919.211 | 10298.503 | 1.000× | +0.00% | 47590 |
| h100-1 | 66 | FLASH_ATTN | 9 | 10832.904 | 9233.597 | 1.100× | +9.11% | 47590 |
| h100-1 | 66 | CUDNN_ATTN | 9 | 11097.779 | 9459.857 | 1.074× | +6.89% | 47590 |

Transformer median reductions explain **95.39–105.61%** of request median
reductions across the eight comparisons. Values slightly above 100% reflect
separately aggregated medians and non-transformer variation. This supports
transformer-time attribution rather than a startup or VAE speedup claim.
FA3 is faster than cuDNN in all four card/frame groups in this run; this does
not establish a universal ordering across workloads or hardware.

## Numerical checks and architecture consistency

| Check | Result |
| --- | --- |
| Requested implementation | SDPAImpl / FlashAttentionImpl / CuDNNAttentionImpl, 80 modules each |
| FA3 provider | `fa3_fwd_interface`, `fa3-fwd==0.0.3` |
| Forward count | 12 for every 33-frame request; 18 for every 66-frame request |
| Latent shapes and prompt | Identical sequence and prompt across backends for every case |
| Raw array integrity | All 108 SHA256 values recomputed and matched; FP32, finite, expected FHWC shape, pixels in [0,1] |
| Within-backend repeatability | Exact in all 36 card/backend/frame/seed groups |
| Cross-card agreement | All 54 corresponding measured arrays have identical SHA256 |
| Synthetic BF16 attention vs FP32 math SDPA | All 18 checks pass predeclared `atol=0.005, rtol=0.01`; largest absolute error 0.00268615 |
| Synthetic attention repeatability | Exact for all 18 checks |
| Relevant upstream tests | 62 passed |

Synthetic inputs use batch 1, 40 heads, head dimension 128, and query/key lengths
128/128, 512/256, and 2048/2048. They test the actual three implementation classes
on each card. They are not a full distribution of Helios activations or an
early-latent diagnostic.

The original comparator produced the following ranges over six frame/seed cases.
Both cards produce the same numerical metrics because their corresponding arrays
are byte-identical.

| Versus SDPA | MAE | Mean frame SSIM | Maximum local error | Temporal-delta MAE |
| --- | --- | --- | --- | --- |
| FA3 | 0.00479850–0.02904551 | 0.89060926–0.99291587 | 0.59017140–1.00000000 | 0.00390536–0.02876623 |
| cuDNN | 0.00404852–0.02100331 | 0.91796142–0.99427104 | 0.57952410–1.00000000 | 0.00319683–0.02420443 |

There is no observed silent fallback, same-work mismatch, or repeat-to-repeat
cache contamination in this matrix. The data supports the performance claim
and measured execution consistency. It does not justify describing final videos
as bit-exact, numerically equivalent, or perceptually lossless across backends.
A useful follow-up is to record the resolved provider directly in upstream
benchmark metadata, so that installed package versions cannot be mistaken for
proof of the active provider.

## Validation and reproduction

At the pinned checkout, run each command with only one GPU visible. Use forward
backend order on card 0 and reverse order on card 1, waiting for each job to finish:

```bash
CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  python -m benchmarks.diffusion.benchmark_helios_attention \
  --model /path/to/Helios-Distilled --backend FLASH_ATTN \
  --frames 33 66 --seeds 42 7 123 --warmup 1 --repeats 3 \
  --output-dir review/h100-0/FLASH_ATTN
python -m benchmarks.diffusion.compare_helios_attention review/h100-0
python -m benchmarks.diffusion.compare_helios_attention review/h100-1
python -m pytest -o addopts="" -q \
  tests/diffusion/attention/test_sdpa.py \
  tests/diffusion/attention/test_cudnn_attn.py \
  tests/diffusion/attention/test_flash_attn.py \
  tests/diffusion/test_helios_attention_benchmark.py
```

The first test invocation encountered the repository's xdist option with xdist
absent. The reported run clears `addopts` and runs the selected tests serially:
62 passed, with existing Torch deprecation warnings and an unknown `asyncio_mode`
configuration warning. A strict full documentation build was not run because the
full documentation toolchain is not installed.

[Machine-readable evidence](Helios-Distilled-H100-review-results.json) includes
all original request records, summaries, comparator outputs, provider identity,
operator checks, and checkpoint-file hashes. The 15 GB raw arrays/videos remain
in the experiment archive; they are not committed. Model weights were deleted
after all six model runs and saved-array verification, releasing 80,502,702,085
bytes. The original H20 recipe and measurements are unchanged.
