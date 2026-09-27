# MammothModa2 DLO Ulysses validation results

Measured on 2026-09-27 against PR #7188 head `66972ed6` and this draft's
`8fb5a7c8`. Both runs used the same four RTX 5090 GPUs, BF16, NCCL, and
PyTorch `2.13.0+cu130` in `/home/gongji/0z5a`. The benchmark script SHA-256
was `e1a393a4bf97b99f26ff51b7c58a66674e8fd473cc7dc38e94b8372e5abe05c1`.
The pipeline source hashes matched those two commits:
`de849b9931182fc000bc10efce264540b46a90a14989ba6adbfb88292f0b8912`
and `c5b8eeb6714ffdaa6dcd989883a415f5dbff3d1bd517826017033277879e87cd`.

## Unsupported `ulysses_degree=4` rejection speed

One first-run measurement per rank. Timing starts immediately before tiny
pipeline construction, after process import and NCCL group setup. The base
builds the pipeline, applies SP hooks, and reaches the existing model-side
validation. The fix raises during pipeline construction. The slowest rank
determines the distributed operation's elapsed time.

| Rank | PR #7188 base (ms) | This draft (ms) | Failure stage |
| --- | ---: | ---: | --- |
| 0 | 72.119 | 14.782 | Model SP check → constructor |
| 1 | 88.168 | 13.713 | Model SP check → constructor |
| 2 | 78.944 | 14.205 | Model SP check → constructor |
| 3 | 77.510 | 14.730 | Model SP check → constructor |
| Slowest rank | **88.168** | **14.782** | **73.386 ms less; 5.96× faster rejection** |

This is a tiny-model validation benchmark with one run per source revision.
It excludes checkpoint download/loading, DLO buffer setup, AR, denoising,
and VAE decoding. It does not measure valid-request inference throughput.
The real `bytedance-research/MammothModa2-Preview` checkpoint at revision
`ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6` is still unavailable on
the test machine, so full-checkpoint E2E and its speed comparison remain
pending.

## Correctness checks

| Check | Result |
| --- | --- |
| MammothModa2 DLO CPU tests | 24 passed |
| Two-GPU BF16 rank-local lifecycle | 1 passed |
| Two-GPU BF16 AllGather lifecycle | 1 passed |
| Ruff on changed files | passed |
| Full-checkpoint E2E | pending model download |

The measured rank records are retained in
`/home/gongji/0z5a-work/mm2v/bench_invalid_ulysses_base_v2.log` and
`/home/gongji/0z5a-work/mm2v/bench_invalid_ulysses_candidate.log`.
