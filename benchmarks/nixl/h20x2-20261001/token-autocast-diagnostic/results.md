# Token-ID and autocast diagnostic

Four complete 512×512, 50-step ordinary-NIXL images were generated with
the token-ID boundary and FP32-router autocast fixes. Each prompt uses
one warmup and one repeated request. Source and receiver CPU snapshots
were enabled; this run does not establish a transport speedup.

| Prompt | Repeat pixels equal warmup | PSNR vs earlier local | SSIM vs earlier local |
| --- | --- | ---: | ---: |
| Dog | Yes | 42.55 dB | 0.99498 |
| Coffee | Yes | 36.18 dB | 0.98332 |

All eight CFG receives match the source bytes exactly. All sixteen captured
first-layer rows match their KV writes and logical-to-physical slot mappings;
maximum CPU-reference attention RMSE is 0.00001023. Transport errors and
remaining ownership are zero. Workers 299820 and 300542 exited naturally.

The trace identified Instruct AR input framing against the checkpoint's
pretrain DiT template. The validation input builder now uses the checkpoint
template; four real-tokenizer prefix checks pass. The final matched local,
ordinary and page-READ runs remain pending. Earlier local images predate
the autocast correction, so the comparisons above are diagnostic evidence.
