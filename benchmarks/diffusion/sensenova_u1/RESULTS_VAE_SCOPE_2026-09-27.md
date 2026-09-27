# SenseNova VAE scope validation

`SenseNovaU1Pipeline` declares `_vae_modules = []`; its FM head generates
pixel-space patches that are unpatchified directly. A temporary synthetic
106 MB checkpoint exercised that complete pipeline on RTX 5090 GPUs with
TP1 and TP2. Each process generated two 64 × 64 images with four denoising
steps, seed 42 and CFG 1. The checkpoint was removed after testing.

| Check | TP1 | TP2 | TP2 relative speed |
| --- | ---: | ---: | ---: |
| Complete image requests | 2/2 | 2/2 | N/A — docs validation |
| Second request latency | 83.29 ms | 163.46 ms | 0.510× |
| Declared VAE modules | 0 | 0 | N/A |
| Repeated output SHA256 | `f3cc103136423a57975750907ebc1d367e2985ac6338976d4d5a439f50323f4a` | Same | N/A |

Both processes exited with status 0. The random-weight checkpoint validates
the pipeline route and absence of a VAE module, not trained-model quality or
full-checkpoint performance. These single warm-request timings on a shared
host do not establish a TP speedup.
