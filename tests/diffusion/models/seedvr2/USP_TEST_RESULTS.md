# USP support validation — 2026-09-22

The native pipeline uses the existing `ulysses_degree` configuration and SP group
for specialized window attention. Sequence shards exchange QKV for head shards
within reference regular/shifted windows, then restore sequence ownership before
MLP execution. No new framework parallel option is introduced.

| Check | Result |
|---|---|
| CPU model suite | 651 passed, 5 deselected |
| Shared framework regression | 294 passed, 1 skipped |
| Shared transport worker, 8 Gloo ranks | Passed; uneven 20-head splits and empty token shards |
| Full-model requests, USP=2 | All six passed; both ranks produced identical outputs |
| Quality versus FP16 SP1 | Minimum PSNR 58.9811 dB; maximum error 0.069824 |
| Actual distributed path | SeedVR2UlyssesRuntime; 64 exchanges and 32 head gathers per forward |

Native USP also passed all six cases on four GPUs: minimum PSNR 53.0735 dB and maximum absolute error 0.134766. All four ranks exited successfully.

## Validation scope

RTX 5090, PyTorch 2.13.0+cu130, existing 0z5a environment, released FP16 3B DiT
and VAE. Six request cases cover five and six frames, fractional frame rate,
2x/4x output sizes, a changed seed, and exact repeated-seed output. Quality gates:
MAE <= 0.02, maximum error <= 0.25, PSNR >= 30 dB. Quantization failures above
are retained rather than relaxing those gates.

Full-model request validation invokes the production DiffusionWorker/model runner
with externally launched ranks. It does not include HTTP, scheduler/IPC, or
encoded-video output. HTTP E2E remains pending. These are correctness results;
no WP-versus-USP performance comparison or speedup claim is made.

Model source hashes and actual runtime counters were recorded per rank. The
finite validation launcher waits for natural worker completion without sending
termination signals. Model weights remain needed for ongoing validation.

## Reproduction

CPU model suite: `python -m pytest -q -o addopts="" -m cpu tests/diffusion/models/seedvr2`.
The real-checkpoint worker accepts `--case seedvr2 --ulysses`; its report records
the runtime class and communication counters. `ulysses_transport_worker.py`
validates exact routing using externally provided rank/rendezvous variables.
