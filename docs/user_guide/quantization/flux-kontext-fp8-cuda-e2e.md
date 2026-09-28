# FLUX.1-Kontext online FP8 CUDA hot path: full image E2E

The official `black-forest-labs/FLUX.1-Kontext-dev` checkpoint at revision
`24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d` was verified against its
23-file manifest. One RTX 5090 (GPU 3) ran the complete image editing
pipeline at 512×512, 28 steps, BF16, with online FP8 for T5 and the DiT,
layerwise transformer offload, and VAE tiling. All A/P/P/A arms used the
same source, prompts, input images, and seed. Each had a separate model
load, three warmup requests, and six timed full requests. A uses the
reference CUDA activation quantizer; P uses the sm_80+ hot path.

| Arm | CUDA activation path | Full request mean ± SD (s) | Speedup vs pooled A | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| A1 | reference | 9.904 ± 0.021 | 1.007× | 7.50 |
| P1 | sm_80+ | 10.121 ± 0.212 | 0.986× | 7.50 |
| P2 | sm_80+ | 10.156 ± 0.178 | 0.982× | 7.50 |
| A2 | reference | 10.050 ± 0.205 | 0.993× | 7.50 |

Pooled full request latency: **9.977 s reference → 10.139 s CUDA**,
**0.984× speedup** (1.6% slower). A2/A1 drift was 1.015× and P2/P1
drift was 1.003×. All 12 paired A/P PNGs were pixel exact (mean RGB
SSIM 1.000000). The P arms made 144 CUDA FP8 activation quantization calls
per measured request; the A arms made zero. Runtime inspection found 144
T5 and 114 DiT FP8 linear modules in every arm.

With layerwise offload enabled, moving the already loaded text encoders and
VAE to CPU before loading the transformer prevents startup GPU OOM. The
same placement is used in all four measured arms. Earlier GPU 6 A1/P1/P2
results were kept separately; its A2 attempts were disturbed by external
GPU work and are excluded from this comparison.
