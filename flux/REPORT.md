## Summary

Consolidates #7828 into this PR: opt-in online FP8 for FLUX.1-Kontext T5 and FLUX.2-klein Qwen3 text encoders, with fixed-shape Klein CUDA graph replay.

- Component settings explicitly select `text_encoder_2` (Kontext T5) or `text_encoder` (Klein). Global DiT FP8 does not enable encoder quantization.
- T5 keeps embeddings, relative position bias, norms and FFN output projections in their original precision. Klein returns the decoder hidden states used by the pipeline without executing the unused language-model head.
- Klein replay orders work across streams using events and stream waits, serializes shared graph buffers, and uses eager execution for unsupported shapes, enforce-eager and offload configurations.
- Follow-up fixes keep offloaded DiT initialization on CPU, include late-loaded biases in streaming quantization, and avoid restoring all offloaded weights onto the GPU during shutdown.

The optional activation CUDA kernel from #8059 is disabled. Existing DiT FP8 performance is reported separately from the new encoder paths.

## Validation

Runtime implementation: `1d39542fb477d4b1954be489115f95850106e5de`; baseline: `27a6321a6311870fe68c2c9c13b694f2612ff779`. Final head `fd38e834332186050550162bad4faea98bd931c5` adds only validated documentation; runtime code is unchanged after validation.

Official checkpoint revisions, sizes, Git objects and every LFS SHA256 were verified:

| Model | Revision |
|---|---|
| FLUX.2-klein-4B | `e7b7dc27f91deacad38e78976d1f2b499d76a294` |
| FLUX.1-Kontext-dev | `24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d` |

Each complete-request cohort uses two fresh processes per configuration in opposite order, two warmups and six measured requests per process: 12 timed requests per row. Image saving and native profiling are outside the measured requests. Tables use wall time and sample standard deviation; positive reduction means faster than the stated reference. Full-request memory is peak **reserved** MiB; standalone encoder memory is peak **allocated** MiB. Hardware and execution configurations are kept separate.

### Klein: RTX 5080, DiT layer offload

1024×1024, four steps, guidance0, seed42, BF16 pipeline, TP1, same DiT-only layerwise offload/pageable memory. Encoder graph and regional DiT compilation are disabled in all four arms.

| FP8 components | Mean ± SD ms | Median ms | Request reduction vs BF16 | Peak reserved MiB | Encoder / DiT FP8 projections |
|---|---:|---:|---:|---:|---:|
| None | 3039.98 ± 86.04 | 3022.17 | — | 13704 | 0 / 0 |
| Encoder | 3074.07 ± 108.87 | 3062.87 | −1.12% | 10222 | 252 / 0 |
| DiT | 1906.32 ± 47.31 | 1921.08 | +37.29% | 13234 | 0 / 80 |
| Both | 1889.46 ± 56.70 | 1865.39 | +37.85% | 9758 | 252 / 80 |

Most of the combined reduction comes from existing DiT FP8. Encoder-only FP8 saves memory but is slightly slower in this execution mode.

### Klein: RTX 5080, resident encoder graph comparison

Same image/steps/prompts/seed, no offload. All arms actually use 25 regional compiled DiT forwards. The only control within each precision pair is native encoder graph versus eager encoder execution.

| FP8 components | Encoder graph | Mean ± SD ms | Process means ms | Peak reserved MiB |
|---|---|---:|---|---:|
| Both | Eager | 1068.36 ± 2.89 | 1068.37, 1068.34 | 12966 |
| Both | Native | 1043.68 ± 3.80 | 1044.16, 1043.19 | 13128 |
| Encoder | Eager | 4390.74 ± 29.65 | 4408.81, 4372.67 | 16832 |
| Encoder | Native | 4295.15 ± 51.56 | 4332.48, 4257.83 | 16690 |

With both components FP8, native encoder graph reduces full-request latency by **2.31%**. Encoder-only rows numerically differ by 2.18%, but reserved memory exceeds the GPU's reported16303MiB and VAE decode takes about3s, with substantial process variation. They do not establish a reliable encoder-only graph speed benefit on this device; driver paging was not directly measured.

Actual graph audits verify capture/replay on native arms and graph disabled on eager arms. Two native profiles each contain one `cudaGraphLaunch`; actual SM120 CUTLASS FP8 GEMMs are572 (both) and252 (encoder). All64 output PNGs are byte-identical across eager/graph and repeated processes within each precision/prompt group.

Standalone Klein encoder wall means: BF16 **53.68ms**, FP8 eager **40.18ms**, FP8 graph **33.87ms**. The graph reduction relative to FP8 eager is15.71% for the encoder only. Ten graph/eager/fallback/IDs-mask/four-stream checks match exactly, max absolute difference0. BF16-to-FP8 hidden-state NRMSE is5.42–7.09%.

### Kontext: Jetson AGX Thor, two separate editing workloads

512×512 outputs,28 steps, guidance2.5, seed42, BF16 pipeline, TP1, same DiT-only layerwise offload/pageable memory, single-thread lazy weight loading. No encoder graph or regional compiled forwards; FP8 linears have separate internal Torch compile activity.

Fixed teapot input is1024×1024:

| FP8 components | Mean ± SD ms | Median ms | Request reduction vs BF16 | Peak reserved MiB |
|---|---:|---:|---:|---:|
| None | 42706.73 ± 989.37 | 42700.63 | — | 13416 |
| Encoder | 42060.62 ± 427.59 | 41955.46 | +1.51% | 9968 |
| DiT | 45723.84 ± 567.35 | 45659.14 | −7.06% | 13418 |
| Both | 44081.70 ± 358.15 | 44158.19 | −3.22% | 9962 |

Real astronaut photograph input is512×512:

| FP8 components | Mean ± SD ms | Median ms | Request reduction vs BF16 | Peak reserved MiB |
|---|---:|---:|---:|---:|
| None | 22329.80 ± 539.73 | 22314.11 | — | 11874 |
| Encoder | 22554.99 ± 1013.86 | 22117.09 | −1.01% | 8426 |
| DiT | 20105.40 ± 143.90 | 20106.51 | +9.96% | 11876 |
| Both | 20393.27 ± 275.89 | 20383.39 | +8.67% | 8420 |

All16 processes naturally completed:128 full requests/96 timed. A healthy CPU model copy ran throughout this campaign, observed at approximately0.544MiB/s; these are measurements with concurrent CPU/I/O work. The538 GPU observations contained no foreign GPU process.

Standalone T5,30 samples per mode:

| Precision | Wall mean ms | CUDA mean ms | Peak allocated MiB |
|---|---:|---:|---:|
| BF16 | 144.77 | 144.73 | 9308.52 |
| FP8 | 168.96 | 168.92 | 5852.59 |

T5 FP8 saves **3455.93MiB (about3.38GiB)**, while standalone latency increases **16.71%** on Thor. Full-request speed depends on the workload. Hidden-state NRMSE is18.17–22.45%. Native profiles confirm144 T5 FP8 projections;24 FFN output projections remain BF16. The fixed-both full-request profile contains3336 actual CUTLASS calls (144T5 +114DiT×28steps), with no CUDA graph runtime calls. The real-photo cohort did not have a separate native trace.

## Output quality and limits

Within every precision/prompt group, independent-process and request repeats produce identical PNG hashes. FP8 outputs differ from BF16.

| Workload / FP8 components | SSIM range versus BF16 | MAE per255 range |
|---|---:|---:|
| Fixed / Encoder | 0.8730–0.9931 | 0.709–6.661 |
| Fixed / DiT | 0.8676–0.9347 | 3.684–5.832 |
| Fixed / Both | 0.8356–0.9407 | 3.364–8.342 |
| Real / Encoder | 0.9644–0.9893 | 1.555–3.442 |
| Real / DiT | 0.9809–0.9859 | 1.671–1.851 |
| Real / Both | 0.9571–0.9826 | 1.913–3.215 |

Manual review: fixed-scene green-pot and flower edits are present, but the warm-light edit loses the blue cup in all modes, including BF16. The real-photo edits preserve the astronaut, suit and helmet; a small red flower/mark on the helmet is absent or less discernible with encoder/both FP8 than with BF16/DiT-only. Klein FP8 changes cup handles and storefront details; BF16 also fails the requested two-cup count. This is lossy quantization and the experimental T5 option needs workload-specific quality validation. Encoder TP and other offload combinations remain unvalidated.

## Regression checks and evidence

CPU encoder quantization/graph tests, offload-shutdown/loading regressions, and applicable repository hooks pass. A real tiny BF16 encoder CUDA graph regression also passes across streams. No workflow file changed; actionlint was skipped. Final documentation hooks pass separately.

All original logs, configs, source identities, actual helpers, images, hidden tensors, native traces and natural-exit records are SHA256-backed locally:

- Klein four-component/standalone,234 files: `e57a743403345a2f7d57d68848f8fc1dc4da4d493f3f10be52294738859b213e`
- Klein final resident,182 files: `4b3f0e599f98466decee572b807ed7d8c0cbe7497e32f9e7f188c335f7b88a5b`
- Kontext formal/T5/background,366 files: `7664c31aad62eb48edb5a1541e184e843acbd65665e405d41999df20811f3bf0`

Completed model files were removed after backup and consumer checks.
