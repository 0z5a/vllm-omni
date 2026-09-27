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
the pipeline route and absence of a VAE module. These single warm-request
timings on a shared host do not establish a TP speedup.

## Full-checkpoint E2E on GPUs 5 and 6

All eight SHA256-verified shards of `sensenova/SenseNova-U1-8B-MoT`
(revision `bfa9b436503cb8aed4f2bc60e3236710cc77468d`) were loaded
with TP2 on RTX 5090 GPUs 5 and 6. The real pipeline generated two
256 × 256 images at four denoising steps, seed 42, CFG 1 and think mode off.

| Full checkpoint, TP2 | First request | Repeated request | Relative speed |
| --- | ---: | ---: | ---: |
| Complete image requests | 1/1 | 1/1 | N/A — docs validation |
| Request latency | 2,008.49 ms | 309.15 ms | 6.50× after warmup |
| Declared VAE modules | 0 | 0 | N/A |
| Image SHA256 | `386bd2c90e9e356d172ce5adbb970b36907bc9723b482b06741010e82db6d2aa` | Same | N/A |

Initialization took 44.58 s, including loading 16.52 GiB per GPU.
Both outputs were 256 × 256 and the process exited with status 0. The
first-to-second-request difference reflects warmup, not this documentation
change. The installed vLLM 0.29.0 runtime required removing one unsupported
`gelu_and_mul_sparse` priority argument from the remote test copy only; no
environment or PR source was changed. The checkpoint was removed afterward.
