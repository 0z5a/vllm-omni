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
passed all 128 related cases, including all eight original native CUDA cases
and the new two-GPU page scatter/paged-attention case. Nineteen subsequent
page/bridge tests passed, including idempotent Worker cancellation before
READ submission. The broader CPU run passed 478 cases before those last two
cancellation cases were added. The two new core modules passed the focused
type check with dependency imports skipped.

That failed attempt left the original AR Worker holding its already-claimed
source lease after the receiver exited. It retains roughly 87 GB of VRAM;
its frontend control channel is closed. No process was killed or signalled.
The pending model runs require another available GPU pair. Consequently,
ordinary/page pixel identity, local/page PSNR/SSIM and full-model speedup
have not passed. The model files remain until that verification is complete.

The final geometry/cancellation code was also measured with the same 8 MiB
payload, 20 warmups per repeat and 1,002 measured samples per path:

| Transfer path | p50 ms | p95 ms | p99 ms | Speedup vs ordinary | Receive tensor allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary NIXL | 4.370 | 5.558 | 6.339 | 1.00× | 8 MiB |
| Local GPU copy | 0.509 | 0.765 | 0.846 | 8.58× | 0 |
| NIXL page READ | 1.741 | 2.033 | 2.090 | 2.51× | 0 |

Each pool registered once across 1,062 transfers including warmups, with
zero transport errors and all ownership drained before natural exit.
This rerun shared the host with the retained AR Worker, which held VRAM
and polled its pending lease. It measures transfer behavior, not model speed.
Raw samples and counters use the `page-integration-*` filenames.

The completed transfer benchmark and raw timings are recorded separately in
[page transfer results](nixl_h20_page_results.md). The local request timings,
checkpoint manifest, example images and JUnit evidence are committed under
[`benchmarks/nixl/h20x2-20261001`](../../../../benchmarks/nixl/h20x2-20261001/).
