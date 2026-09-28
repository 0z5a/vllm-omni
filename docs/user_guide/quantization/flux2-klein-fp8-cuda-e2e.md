# FLUX.2-klein online FP8 CUDA hot path: full model E2E

Official `black-forest-labs/FLUX.2-klein-4B` checkpoint at revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`; model files were verified
against the checkpoint manifest before testing. The run used an RTX 5090
(GPU 4), BF16, online FP8 for both the Qwen3 text encoder and transformer,
1024×1024 output, four inference steps, and the text encoder CUDA graph.
Each A/P/P/A arm generated three warmup images followed by six timed full
requests with the same prompts and seed. A disables the new CUDA activation
quantization path; P enables it. The model and all other settings are equal.

| Arm | CUDA activation path | Full request mean ± SD (s) | Speedup vs pooled A | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| A1 | reference | 0.637 ± 0.006 | 1.003× | 10.52 |
| P1 | sm_80+ | 0.649 ± 0.004 | 0.985× | 10.53 |
| P2 | sm_80+ | 0.644 ± 0.002 | 0.993× | 10.52 |
| A2 | reference | 0.641 ± 0.005 | 0.997× | 10.53 |

Pooled full request latency: **0.639 s reference → 0.646 s CUDA**, a
**0.989× speedup** (1.1% slower). A2/A1 drift was 1.006×; P2/P1 drift was
0.992×. The P arms made 80 CUDA FP8 activation quantization calls and one
text encoder graph replay per measured request; the A arms made zero calls
and one graph replay.

Outputs were deterministic within each path: A1/A2 and P1/P2 were each
6/6 pixel exact. The CUDA and reference outputs differed (0/12 pixel exact;
mean RGB SSIM **0.961489**). A one prompt whole model eager comparison
produced pixel exact A/P images, while leaving the diffusion model compiled
reproduced the difference even with eager text encoding. Representative
activation quantization shapes (`512×2560` and `512×9728`) matched the vLLM
reference payload and scale bit for bit, both directly and under CUDA graph
capture. The compiled full model difference remains unresolved; this E2E
result does not support a speed or output equivalence claim for this path.
