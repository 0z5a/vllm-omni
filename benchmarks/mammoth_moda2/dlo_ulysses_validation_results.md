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

## Full-checkpoint E2E and valid-request speed

On the same RTX 5090 machine, starting from PR head `b08254c4`, the full 37.1 GB
`bytedance-research/MammothModa2-Preview` checkpoint at revision
`ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6` was downloaded directly
and verified shard by shard against the Hugging Face SHA-256 manifest. The
shared text-to-image example generated one real AR-conditioned 1024×1024
image with BF16, Ulysses SP2, SDPA, 50 denoising steps, and seed 42. The
same recorded conditioning was then replayed through the native DiT engine
on GPUs 1 and 3 in resident, rank-local DLO, and AllGather DLO modes. Each
mode used one warmup and three measured requests. Timings include the DiT
request and output transfer, but exclude AR, engine startup, and result
serialization.

| DiT mode | Mean latency (ms) | Speed vs resident | Peak allocated per rank (GiB) | Memory saved vs resident |
| --- | ---: | ---: | ---: | ---: |
| Resident | 11,513.07 | 1.00× | 7.95 | — |
| Rank-local DLO | 12,801.72 | 0.90× | 5.17 | 35.0% |
| AllGather DLO | 14,422.86 | 0.80× | 5.37 | 32.5% |

Both DLO modes reproduced the resident decoded tensor exactly: maximum
absolute difference 0, normalized RGB MAE 0, and SSIM 1.0. The offload
modes saved GPU memory but were slower for this one-request workload.

The 5090's preinstalled vLLM `0.29.0` required four API compatibility edits
in the test copy only: omit the removed `gelu_and_mul_sparse` IR priority,
call `_init_model_kwargs()` without `num_reqs`, use the current structured
output grammar interface, and skip an unrelated Cosmos3 import while
detecting Mammoth's direct-mmap adapter. The Mammoth model, DLO hooks,
checkpoint, request, and replay code were unchanged; all modes used the
same test copy. The model files were removed after E2E completion. Raw
timings and comparisons are in
`/home/gongji/0z5a-work/mm2v-results-20260927` on the 5090 machine.

## Correctness checks

| Check | Result |
| --- | --- |
| MammothModa2 DLO CPU tests | 24 passed |
| Two-GPU BF16 rank-local lifecycle | 1 passed |
| Two-GPU BF16 AllGather lifecycle | 1 passed |
| Ruff on changed files | passed |
| Full-checkpoint AR→DiT generation | passed; 1024×1024 image produced |
| Resident, rank-local, AllGather replay | passed; 3 measured requests each |
| DLO decoded output vs resident | exact match in both modes |

The measured rank records are retained in
`/home/gongji/0z5a-work/mm2v/bench_invalid_ulysses_base_v2.log` and
`/home/gongji/0z5a-work/mm2v/bench_invalid_ulysses_candidate.log`.
