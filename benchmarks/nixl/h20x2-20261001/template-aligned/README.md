# Checkpoint-template E2E replay

Both stages use the pinned checkpoint pretrain template, 512 × 512 output,
50 diffusion steps, guidance 2, and seeds 1234/1235. Each path has two warmups
and six measured requests. The ordinary reference runs without diagnostic
source snapshots or attention tracing.

| Path | Completed / measured | Median seconds | Speedup vs ordinary | Status |
| --- | ---: | ---: | ---: | --- |
| Local prefix recomputation | 8 / 6 | 14.654 | Pending | Complete; both Workers exited naturally |
| Ordinary NIXL | Unconfirmed | Pending | Pending | Started; final logs unavailable after SSH interruption |
| Native-page READ | 0 / 0 | Pending | Pending | Awaiting ordinary completion and available GPU window |

The saved local PNG SHA-256 values and all eight 512 × 512 images were
rechecked offline. Each prompt has four pixel-identical images; see
[local evidence check](local-evidence-check.json). Input-template changes
make earlier Instruct-input timing ratios unsuitable for this comparison.

The local source manifest records the 36 files checked before its run.
The subsequent token-checked text-prefix change in request_layout.py is
inactive for local requests, which have no kv_transfer_params.
The related layout/routing/bridge/reference suites passed 76 CPU cases
with zero skips; the JUnit record is [text-prefix.xml](text-prefix.xml).

Acceptance requires eight matched images per path, identical ordinary/pages
pixels, local/pages PSNR ≥30 dB and SSIM ≥0.97, zero transport errors and
drained ownership. Timing alone does not establish acceptance.
