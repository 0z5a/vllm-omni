# SeedVR2 causal VAE tiling and patch parallelism

Owner: 0z5a. Baseline: `af59b66fc0c2c32696661618265ce0bb93d6f9e2` (#7852).
The opt-in candidate combines causal temporal chunks, spatial convolution tiles,
and height-sharded VAE execution. The default whole-clip path is retained.

## HTTP speed comparison

| Quartet | Order | Baseline ms | Candidate ms | Speedup | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | APPA | 3591.899 | 3012.397 | 1.1924× | 16.13% |
| 2 | PAAP | 5238.402 | 3025.173 | 1.7316× | 42.25% |
| 3 | APPA | 3596.066 | 3253.415 | 1.1053× | 9.53% |
| 4 | PAAP | 3581.507 | 3124.123 | 1.1464× | 12.77% |
| 5 | APPA | 3512.263 | 3157.216 | 1.1125× | 10.11% |
| Geometric aggregate | 20 fresh services | **3854.790** | **3113.199** | **1.2382×** | **19.24%** |

The workload is five frames at 848×480, full released 3B FP16 checkpoints,
CFG 1, one Euler step, eager execution and grouped attention. Both variants
use two L20 GPUs and window-SP degree 2. The baseline replicates the VAE;
the candidate enables `--vae-use-tiling` and VAE patch degree 2.
Five APPA/PAAP quartets use 20 fresh services, five request warmups and
24 measured requests per service. The timer covers upload through MP4 download;
startup and source audits are excluded. CPU affinity and model files are fixed.
The host is shared, and absolute latency varies with concurrent work.

All five quartets improve and pass the predeclared 1% minimum-effect gate.
The descriptive 95% whole-quartet bootstrap interval is 10.42–31.93% latency
reduction (all 5^5 resamples). The second quartet has a substantially slower
baseline; it is retained, along with every other sample. This shared-host result
is not an isolated-GPU scaling claim or a tail-latency SLA.

All 480 measured responses pass frame count, dimensions, PTS, audio and
same-variant repeatability checks. Cross-variant decoded MP4 MAE is 0.005023,
maximum error 0.066667. All 20 services exit normally with code 0. Each of the
40 workers completes exactly six untimed audited forwards; Python source and
loaded Triton binary fingerprints match within each variant.

[Every sample](seedvr2_vae_parallel_evidence/requests.json),
[statistics](seedvr2_vae_parallel_evidence/statistics.json),
[decision rule](seedvr2_vae_parallel_evidence/decision-rule.md),
[source/kernel/shutdown audit](seedvr2_vae_parallel_evidence/verified-source-shutdown.json),
[local source verification](seedvr2_vae_parallel_evidence/local-source-verification.json).


## Capacity

| 107 frames, 896×512 | Result | Allocated memory |
| --- | --- | ---: |
| Original whole-clip VAE, one L20 | OOM | 43.14 GiB allocated; another 11.92 GiB requested |
| Causal temporal chunks, one L20 | HTTP 200 | 21.22 GiB peak |
| Temporal chunks + VAE patch parallel, two L20s | HTTP 200 | 15.98 GiB peak per rank |

Both successful services return 107 frames at 24 FPS with 32 kHz audio and
accept the following five-frame request. The two successful MP4 outputs have
MAE 0.00008516, maximum error 0.043138 and PSNR 64.30 dB; timestamps and decoded
audio match. Capacity observations use different GPU counts and are not scaling
or latency comparisons. No speed ratio is computed against an OOM.

Two earlier SP2 capacity attempts on GPUs 4/5 failed when other tasks consumed
27–32 GiB on GPU 5. The successful attempt used GPUs 2/3. All attempts remain
in [capacity evidence](seedvr2_vae_parallel_evidence/capacity-attempts.json).

## Correctness and execution evidence

| Full-model raw RGB comparison | Worst MAE | Worst max error | Minimum PSNR |
| --- | ---: | ---: | ---: |
| SP1: five/six frames, 2×/4×, seed change, repeat | 0.00004434 | 0.000977 | 81.30 dB |
| SP2: same six cases | 0.00024655 | 0.013733 | 67.87 dB |
| SP4: same six cases | 0.00050433 | 0.049595 | 57.23 dB |
| Temporal SP1: 17/33/107 frames | 0.00002565 | 0.002198 | 82.03 dB |
| Combined SP2: 17/33/107 frames | 0.00006848 | 0.003418 | 77.07 dB |

The long raw-frame comparisons use 432×240 for 17/33 frames and 224×128 for
107 frames. Same-variant repeats are exact. Five-frame SP1 cases are bitwise
identical to the baseline on the tested shapes. All cases pass the predeclared
MAE≤0.02, max≤0.25 and PSNR≥30 dB gates. FP16 reduction/convolution order can
change rounding across variants; bitwise equivalence is not a general guarantee.

65 relevant CPU tests pass: temporal partitions and causality, 2/4-process
unequal spatial bands and small-image fallback, configuration validation,
existing VAE/pipeline behavior and checkpoint contracts. Real-weight FP16 VAE
stage comparisons also pass on two and four GPUs. Applicable precommit checks,
including mypy, pass. Runtime versions: PyTorch 2.13.0, vLLM 0.29.0,
Diffusers 0.40.0, PyAV 18.1.0 and Triton 3.7.1.

A separate untimed [trace](seedvr2_vae_parallel_evidence/spatial-trace-summary.json)
records Conv3d, NCCL SendRecv, FP32 all-reduce and all-gather on both ranks.
The timed services use warmup-only source/group audits, with no profiler.

## Limits and rejected measurements

Single-GPU spatial tiling bounds convolution workspace, while full-frame
activations remain resident. Distributed VAE uses global normalization statistics
and gathers bottleneck attention; it does not normalize independent image tiles.
Patch degree must equal window-SP degree. Width sharding and batch slicing are
unsupported. Whole-clip DiT activations can still exceed available memory.
Restart a patch-parallel service after OOM because a peer may be waiting in a
collective; no automatic distributed recovery is claimed.

The temporal HTTP screen is unsuitable for a speed claim: identical baseline
runs drifted from 1.258 s to 15.110 s, and an earlier candidate timed out during
startup under disk I/O pressure. Increasing the startup allowance to 900 seconds
changed readiness handling only. The spatial AP screen and A/A pilot are
retained separately from formal results in
[pilot and failure records](seedvr2_vae_parallel_evidence/pilots-and-failures.json).


## Reproduce the HTTP comparison

Use the model files documented in `docs/models/seedvr2.md` and the same five-frame,
25 FPS, 16 kHz fixture; its SHA-256 is recorded in the
[contract](seedvr2_vae_parallel_evidence/contract.json). Both checkouts must be
available locally. The default benchmark runs all five quartets with 24 samples
per service; use `--pilot --samples 4` for the A/A prerequisite.

```bash
python benchmarks/diffusion/benchmark_seedvr2_native_http.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --models "$MODEL_DIR" \
  --input "$FIXTURE" --output "$RESULTS" --gpu 2,3 --sp 2 \
  --candidate-tiling --candidate-spatial --width 848 --height 480 \
  --server-cpus 64-71 --startup-timeout 900 \
  --video-mae-limit 0.02 --video-max-error-limit 0.25
```

Task-owned model files were removed after all services stopped, reclaiming
7,284,938,756 bytes. Raw outputs, traces and reports were retained.
[Cleanup receipt](seedvr2_vae_parallel_evidence/model-cleanup.json).
