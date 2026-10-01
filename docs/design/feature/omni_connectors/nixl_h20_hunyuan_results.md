# Hunyuan full-model validation on H20

Checkpoint: `feizhai123/hunyuan-image3-modelopt-fp8`, revision
`d9a3991e01fcaefe0fb14169e8c9d04fc126c4e1`.
The downloaded checkpoint contains all 18 weight shards, totalling
86,177,283,726 bytes. Both stages use the same FP8 weights, BF16 native KV,
TP1, eager execution, Triton MoE and paged FlashAttention.
Requests use 512 × 512 output, 50 diffusion steps, guidance 2 and seeds
1234/1235 for two fixed prompts. Each prompt has one warmup and three
measured requests.

| Full-model path | Completed requests | Measured requests | Median seconds | Speedup vs ordinary |
| --- | ---: | ---: | ---: | --- |
| Local prefix recomputation | 8 | 6 | 18.587 | Pending ordinary reference |
| Ordinary NIXL into native pages | 0 | 0 | Pending | Pending |
| NIXL READ into native pages | 0 | 0 | Pending | Pending |

| Local path prompt | Measured requests | Median seconds |
| --- | ---: | ---: |
| Brown and white dog running on grass | 3 | 23.686 |
| Coffee on a wooden table beside a window | 3 | 13.279 |

All eight local requests completed AR generation, DiT denoising and VAE
decoding into nonblank 512 × 512 RGB images. Both Workers exited through
the normal shutdown protocol. The FP8 loader fix separately passed 21 CPU
regressions and a real checkpoint AR load/forward check.

The first ordinary-path attempt stopped before DMA because the AR stage
removed `model.` from canonical cache layer names while DiT retained it.
The configuration now removes the same prefix on both stages, and the
producer validates geometry before installing claims. The revised connector
passed 478 cases in the broader CPU run. After consolidating ordinary/page
claim decoding and reservation ownership, all 122 related CPU cases and all
nine native CUDA cases passed without skips. The added reference-path test
checks scatter, cancellation, delayed ACK and computation ownership through
the shared implementation. Three source modules passed focused typing with
dependency imports skipped. Earlier geometry/cancellation evidence remains
committed with its recorded source hashes.

That failed attempt left the original AR Worker holding its already-claimed
source lease after the receiver exited. It retains roughly 87 GB of VRAM;
its frontend control channel is closed. No process was killed or signalled.
The pending model runs require another available GPU pair. Consequently,
ordinary/page pixel identity, local/page PSNR/SSIM and full-model speedup
have not passed. The model files remain until that verification is complete.

The consolidated code was also measured with the same 8 MiB
payload, 20 warmups per repeat and 1,002 measured samples per path:

| Transfer path | p50 ms | p95 ms | p99 ms | Speedup vs ordinary | Receive tensor allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary NIXL | 5.435 | 25.464 | 25.511 | 1.00× | 8 MiB |
| Local GPU copy | 0.507 | 0.757 | 0.804 | 10.72× | 0 |
| NIXL page READ | 1.858 | 2.142 | 2.226 | 2.93× | 0 |

Each pool registered once across 1,062 transfers including warmups, with
zero transport errors and all ownership drained before natural exit.
This rerun shared the host with the retained AR Worker, which held VRAM
and polled its pending lease. It measures transfer behavior, not model speed.
Raw samples and counters use the `page-consolidation-*` filenames, with
`consolidation-source-hashes.json` recording the tested source. The previous
geometry/cancellation run (`page-integration-*`) measured 2.51×; shared-host
latency varies, so these runs do not isolate a speed change from refactoring.

The completed transfer benchmark and raw timings are recorded separately in
[page transfer results](nixl_h20_page_results.md). The local request timings,
checkpoint manifest, example images and JUnit evidence are committed under
[`benchmarks/nixl/h20x2-20261001`](../../../../benchmarks/nixl/h20x2-20261001/).
