# SeedVR2 framewise VAE normalization validation

Owner: 0z5a. Baseline native commit `4f1bb5e` (#7848). This candidate independently
fuses framewise GroupNorm and SiLU in the CUDA FP16 VAE. It keeps frame/group
statistics separate, merges centered FP32 partial moments once per group, rounds the affine
output to FP16 before SiLU, and emits contiguous NCTHW tensors for convolution.
Addresses use 64-bit arithmetic for long clips; the statistics merge is linear
in spatial area. CPU and non-FP16 calls retain the existing PyTorch path. No temporal/spatial
tiling, activation quantization or attention change is included.

## Full native HTTP latency

Five frames at 848×480, FP16 full 3B, CFG 1, one Euler step, grouped SDPA, SP1, eager.
Five independent APPA/PAAP quartets; 20 fresh services, 5 warmups + 24 measurements each.
Timer: multipart upload through MP4 download; model startup excluded.

| Quartet | Order | Baseline ms | Candidate ms | Speedup | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | APPA | 1374.026 | 1310.327 | 1.0486× | 4.64% |
| 2 | PAAP | 1383.863 | 1314.066 | 1.0531× | 5.04% |
| 3 | APPA | 1386.649 | 1312.626 | 1.0564× | 5.34% |
| 4 | PAAP | 1387.364 | 1315.502 | 1.0546× | 5.18% |
| 5 | APPA | 1376.997 | 1310.794 | 1.0505× | 4.81% |
| Geometric aggregate | 20 fresh processes | **1381.770** | **1312.662** | **1.0526×** | **5.00%** |

All five quartets improve. The predeclared 1% MES passes; descriptive 95%
whole-quartet bootstrap interval is 4.78–5.22%. With five independent units, this
is not a tail SLA. No samples were trimmed. The host is shared; no scaling claim.

All 480 measured outputs are repeatable within each variant; audio hashes match
across variants. Decoded MP4 difference is MAE 0.005048, max 0.082353 (codec-domain
error, distinct from the raw-frame values below). All 20 services exit 0 normally.
Actual Python source hashes and the 44 loaded candidate Triton binaries match
across all ten candidate workers. The source audit wrapper is removed before
timed requests; every worker records exactly six audited warmup forwards.

[Every timing sample](seedvr2_vae_evidence/requests.json),
[statistics](seedvr2_vae_evidence/statistics.json),
[source/kernel/shutdown audit](seedvr2_vae_evidence/verified-source-kernel-shutdown.json),
[decision rule](seedvr2_vae_evidence/decision-rule.md),
[per-rank SP4 stages and allocated/reserved memory](seedvr2_vae_evidence/sp4-stage-memory.json).

## Supplemental small-input result

| Five frames at 224×128 | Baseline ms | Candidate ms | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| One fresh APPA quartet | 244.083 | 247.568 | 0.9859× | **−1.43%** |

This smaller case is slower in the supplemental quartet. It passes the predeclared
5% regression guard, but one quartet does not establish a stable effect. The
positive formal claim applies to the 848×480 workload above.
[Supplemental samples, A/A pilot and excluded runs](seedvr2_vae_evidence/pilots-and-excluded.json)
are retained; the interrupted earlier formal run predates the int64 addressing
and linear statistics-merge fixes and is excluded from the final estimate.

## Full-checkpoint correctness

| Raw full-model comparison | Worst MAE | Worst max error | Minimum PSNR |
| --- | ---: | ---: | ---: |
| SP1: five/six/2×/4×/different seed/repeat | 0.000231787 | 0.016113281 | 68.25 dB |
| SP2: five/six/2×/4×/different seed/repeat | 0.000446866 | 0.030761719 | 59.55 dB |
| SP4: five/six/2×/4×/different seed/repeat | 0.000446866 | 0.030761719 | 59.55 dB |

The floating-point reduction order changes, so this candidate does not claim
bitwise reference parity. It passes the predeclared existing raw-frame gates
(MAE≤0.02,max≤0.25) and additional PSNR≥30dB. Same-variant A/B/A output is
exact; frame count and fractional source FPS are preserved. GPU/CPU tests: 14 passed,
covering constant/noncontiguous frames, large offsets including GPU addresses
beyond 2^31 elements, frame isolation and
VAE causal prefixes. All applicable precommit hooks, including mypy, pass.

The final HTTP media sweep passed all 15 requests: five/six frames, 2×/4× and
return-to-five, repeated three times. Frame count, dimensions, fractional FPS,
relative PTS and 16 kHz audio passed; repeated five-frame decoded output matched.
SP4 diagnostic peak allocated memory was at most 13.237 GiB per rank and peak
reserved memory 17.955 GiB per rank. These are resource diagnostics, not a scaling claim.

The matched 107-frame 896×512 C0 request still fails with HTTP 500: VAE encode
has 43.14 GiB allocated and requests another 11.92 GiB on a 44.39 GiB L20.
Health remains healthy and the subsequent five-frame request succeeds (HTTP 200,
five decoded frames). The diagnostic service exits normally with code 0.
The task-owned model directory was removed after testing, reclaiming 7,284,938,761 bytes.

[HTTP media cases](seedvr2_vae_evidence/http-cases.json),
[long-clip capacity result](seedvr2_vae_evidence/B-107f-2x.json),
[recovery](seedvr2_vae_evidence/recovery.json),
[model cleanup](seedvr2_vae_evidence/model-cleanup.json).

## Candidate screening and rejected data

Channels-last3d and lowered2D convolutions were slower and were rejected.
cuDNN autotuning produced no meaningful gain. A persistent-weight VAE-only
fusion screen improved stage time by approximately 6–7%; this is not an HTTP
speedup claim. The final implementation uses centered partial moments to avoid
cancellation on nearly constant frames, with full-model quality gates above.

A first service attempt reused a busy port; worker-source verification rejected
it before measurement. A subsequent GPU 2 AP screen was contaminated by another
job starting on GPU 2/3 (~24GiB per GPU). The apparent 3.912 s → 1.341 s difference
is excluded from performance estimation. All failed/rejected records are retained.
On free GPU 4, the fresh-process A/A pilot measured 1.41259 s and 1.39695 s (1.11%
drift), with both processes exiting normally. Formal measurements use GPU 4,
server CPU 40–47/client 48 and a dedicated port 19825 on the shared host.

## Reproduce

Use the model and five-frame fixture from [native serving](seedvr2_native_results.md).
Create independent BASELINE at `4f1bb5e` and CANDIDATE checkouts. Both retain the
same parent RoPE and grouped-attention implementations. Use the driver in this
change for both arms; the worker audit refreshes source and compiled-kernel hashes through the
engine warmup and five request warmups. The wrapper is removed before measuring;
the driver asserts this boundary. The compiled-kernel audit uses Triton 3.7.1
cache metadata and records actual cubin/JSON hashes and JIT source keys.

```bash
taskset -c 48 python benchmarks/diffusion/benchmark_seedvr2_native_http.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --models "$MODEL_DIR" \
  --input five-frames-audio.mkv --output vae-aa --gpu "$GPU" --port 19825 \
  --server-cpus 40-47 --width 848 --height 480 --pilot
taskset -c 48 python benchmarks/diffusion/benchmark_seedvr2_native_http.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --models "$MODEL_DIR" \
  --input five-frames-audio.mkv --output vae-formal --gpu "$GPU" --port 19825 \
  --server-cpus 40-47 --width 848 --height 480 \
  --video-mae-limit 0.02 --video-max-error-limit 0.25
```

Audit the A/A drift and device occupancy before formal measurements. The driver
requires exact audio and exact repeatability within each variant; it separately
records decoded MP4 video differences across variants. Quality checks run outside
the upload-through-download timer. Model startup/first-use kernel compilation is
excluded from the steady-state metric; warmup samples remain in the raw records.
The inherited whole-clip 107-frame VAE capacity limitation remains; this change
has no large-video capacity or multi-GPU scaling claim.
