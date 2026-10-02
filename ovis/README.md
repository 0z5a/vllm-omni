# Ovis combined dual-A100 validation

Tested source: `446f42d8eb812f0d40d1b402ed7cfc8dbaad7a48`. All 24 fresh processes naturally completed; 144 measured full requests, 192 PNG hashes verified, both physical ranks audited. Native traces are omitted from this compact upload; their hashes and actual kernel counts are in `summary.json`. The complete archive index is `canonical-index.json`.

# A100 Ovis dual-GPU complete-request validation

Two fresh processes per row; six timed requests per process, warmups excluded.
SM80 encoder FP8 uses the audited weight-only fallback with BF16 activations; native FP8 GEMM is not available on A100.
Memory reports each worker's lifetime peak reserved memory, including untimed profiling/lifecycle audits; it is not a timing-phase-only peak.

| Resolution HxW | Precision | HSDP | Mean ms | Median ms | Sample SD ms | Process means ms | Peak reserved R0/R1 MiB |
|---|---|---|---:|---:|---:|---|---|
| 512x512 | bf16 | none | 3095.83 | 3095.22 | 17.15 | 3083.77, 3107.89 | 18818/18818 |
| 512x512 | encoder | none | 3145.64 | 3145.28 | 9.68 | 3139.50, 3151.79 | 17466/17466 |
| 512x512 | bf16 | block | 39769.24 | 39724.57 | 124.78 | 39723.66, 39814.82 | 13804/13804 |
| 512x512 | encoder | block | 39999.27 | 39983.93 | 542.42 | 39483.75, 40514.80 | 12848/12848 |
| 512x512 | bf16 | resident | 4346.32 | 4334.41 | 48.41 | 4362.43, 4330.20 | 27052/27052 |
| 512x512 | encoder | resident | 4389.40 | 4374.62 | 38.26 | 4389.73, 4389.07 | 25834/25834 |
| 768x1024 | bf16 | none | 7805.99 | 7805.90 | 48.50 | 7779.82, 7832.15 | 21232/21232 |
| 768x1024 | encoder | none | 7838.83 | 7837.72 | 32.18 | 7836.63, 7841.03 | 19880/19880 |
| 768x1024 | bf16 | block | 40279.26 | 40267.69 | 72.17 | 40224.59, 40333.93 | 16218/16218 |
| 768x1024 | encoder | block | 40732.11 | 40666.25 | 483.98 | 40374.04, 41090.18 | 14002/14002 |
| 768x1024 | bf16 | resident | 8871.00 | 8862.07 | 61.75 | 8850.04, 8891.96 | 29834/29834 |
| 768x1024 | encoder | resident | 8925.19 | 8929.35 | 28.06 | 8920.71, 8929.67 | 26998/26998 |

Untimed encoder CUDA events: sum of positive and negative forward GPU durations, profiled separately on each rank; tokenization and input preparation excluded.

| Resolution HxW | Precision | HSDP | Encoder CUDA mean R0/R1 ms |
|---|---|---|---|
| 512x512 | bf16 | none | 191.989/197.072 |
| 512x512 | encoder | none | 244.173/245.921 |
| 512x512 | bf16 | block | 213.194/211.937 |
| 512x512 | encoder | block | 265.377/260.679 |
| 512x512 | bf16 | resident | 215.700/211.853 |
| 512x512 | encoder | resident | 252.614/263.117 |
| 768x1024 | bf16 | none | 200.451/194.375 |
| 768x1024 | encoder | none | 247.869/245.050 |
| 768x1024 | bf16 | block | 207.625/212.939 |
| 768x1024 | encoder | block | 258.094/257.520 |
| 768x1024 | bf16 | resident | 207.533/213.100 |
| 768x1024 | encoder | resident | 263.891/263.482 |

Matched comparisons; negative reductions mean slower execution.

| Resolution HxW | Comparison | Request latency reduction |
|---|---|---:|
| 512x512 | Encoder FP8 weights / BF16 activations vs BF16, none | -1.61% |
| 512x512 | Encoder FP8 weights / BF16 activations vs BF16, block | -0.58% |
| 512x512 | Encoder FP8 weights / BF16 activations vs BF16, resident | -0.99% |
| 512x512 | Resident vs block HSDP, bf16 | 89.07% |
| 512x512 | Resident vs block HSDP, encoder | 89.03% |
| 768x1024 | Encoder FP8 weights / BF16 activations vs BF16, none | -0.42% |
| 768x1024 | Encoder FP8 weights / BF16 activations vs BF16, block | -1.12% |
| 768x1024 | Encoder FP8 weights / BF16 activations vs BF16, resident | -0.61% |
| 768x1024 | Resident vs block HSDP, bf16 | 77.98% |
| 768x1024 | Resident vs block HSDP, encoder | 78.09% |


## Image review

The two contact sheets contain 72 representatives across both resolutions, six precision/residency modes, two process-order repeats and three prompts. All show one red teapot with two blue cups and handles, a watercolor storefront with a bicycle and readable OPEN text, or exactly three yellow rubber ducks in turquoise water. The matched repeat, request-resident/block and HSDP/unsharded pairs are pixel-exact in the archived comparisons. No nonexact repeat or HSDP pairs require additional images.

Encoder quantization is not lossless. In the 512-square teapot pair, the cup interiors change from blue to red, and cup proportions, rim contours, pot reflections and highlights change. The larger storefront pair retains the requested OPEN signs and bicycle, while letter spacing, window details, street/tree layout and facade details change; the FP8 image introduces garbled extra lettering resembling FILOMG LOR. Duck count remains three, with changes to water ripples, caustics, reflections and contours. Subject/count checks pass for these prompts; fidelity differences remain material and are reported rather than treated as pixel equivalence. Baseline images also have prompt/style limitations. This review covers the recorded prompts and seeds only, not general image quality.

The 36 encoder-quantization comparisons are nonexact (SSIM 0.78847–0.98913, maximum MAE 14.25376/255). SSIM/MAE support the difference report; they do not establish semantic correctness. Same-precision process repeats (36), request repeats (72), resident/block pairs (24) and HSDP/unsharded pairs (48) are all exact. Review completion is not a claim of lossless quantization.

