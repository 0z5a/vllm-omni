# SeedVR2 native HTTP validation

Owner: 0z5a. This integration adds native whole-clip VAE → NaDiT → VAE
restoration to the actual `/v1/videos/sync` endpoint. It builds on #7739 and
#7846. The performance comparison below isolates #7846's RoPE change within
this native service; it is **not** a claimed speedup from adding the service.

## Full-service latency

Five-frame 212×120 input → 224×128, FP16 3B, one Euler step, CFG=1,
seed7723, grouped SDPA, SP1, eager, batch/concurrency1. The timer covers multipart
upload through MP4 response download, including decoding, preprocessing, VAE,
NaDiT and mux; model load is excluded. Each arm starts a fresh service, warms
five requests, then measures24. Each quartet combines its two process medians
geometrically. No samples were trimmed.

| Quartet | Order | Baseline ms | Candidate ms | Speedup | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | APPA | 275.403 | 233.161 | 1.1812× | 15.34% |
| 2 | PAAP | 256.058 | 244.356 | 1.0479× | 4.57% |
| 3 | APPA | 243.468 | 229.226 | 1.0621× | 5.85% |
| 4 | PAAP | 245.072 | 224.622 | 1.0910× | 8.34% |
| 5 | APPA | 247.127 | 241.255 | 1.0243× | 2.38% |
| Geometric aggregate | 20 fresh processes | **253.159** | **234.409** | **1.0800×** | **7.41%** |

All480 measured outputs have identical decoded video and audio hashes across
arms. All20 services exited0 without forced shutdown. The predeclared1% minimum
improvement gate passes; all five quartets improve. Whole-quartet bootstrap95%
interval is3.96–11.50% latency reduction. With five independent units this is
descriptive evidence, not a tail SLA or a universal hardware claim.

Hardware: one NVIDIA L20,44.39GiB, GPU UUID
`GPU-7bf0f2c6-9e45-028e-e5bd-1859ad38f82e`, shared eight-GPU host. Other GPUs
had unrelated jobs. Server CPU affinity40–47, client48, OMP threads4.
PyTorch2.13+CUDA13, vLLM0.29, Diffusers0.40, PyAV18.1. The installed runtime
also applies existing NVFP4/Inductor/CuMem patches equally to both arms.
Unpinned A/A pilots showed33% and23% drift and were rejected; the CPU-pinned
pilot showed2.63%. No exclusive-host or multi-GPU scaling claim is made.

[All480 samples and process records](seedvr2_native_evidence/requests.json),
[statistics](seedvr2_native_evidence/statistics.json),
[measurement contract](seedvr2_native_evidence/contract.json), and
[shutdown/source audit](seedvr2_native_evidence/shutdown-source-audit.json).
The measured immutable A/P snapshots differ only in `seedvr2/rope.py`:
A uses parent9a62e04, P uses4dc1bfb. Both include the same native integration.
Subsequent startup-only unsupported-feature gates and annotation/test fixes
were separately revalidated; they were not retroactively added to the measured
snapshot. Worker source hashes are retained in the evidence directory. The original
measurement driver remains in the local evidence archive; its SHA256 is
`1d8fc3ef3cb4b1973881da9980643641f4eca75b5e46a2c9beee1dd672abb10f`.

## Correctness, resources and lifecycle

| Check | Result |
| --- | --- |
| Native VAE vs pinned reference,1/5/9 frames | Full FP16 checkpoint encode/decode exact |
| Actual HTTP5/6-frame raw stages | Conditioning, velocity, FP16 VAE input and unencoded frames exact |
| Six-frame boundary | Pad6→9 internally; output6,30000/1001fps and correct PTS |
| SP2/SP4 raw frames vs SP1 | MAE0.000171566,max0.00387573; below declared0.02/0.25 gates |
| SP1/2/4 HTTP five/six/2×/4×/five-repeat | All successful; same-seed repeated output exact; audio/FPS/PTS preserved |
| Invalid dimensions/corrupt video | HTTP400; subsequent valid request succeeds |
| Three in-progress async cancellations | DELETE200, GET404, subsequent request succeeds |
| Own SP2 worker killed during request | Request fails; other rank exits; health503; fresh restart succeeds |
| Shared request/media/config regression | 874 passed,14 skipped; latest focused follow-up54 passed |
| Precommit | Applicable hooks pass except14 mypy errors reproduced on untouched4dc1bfb base |

Reference Euler stores FP32 subtraction before casting to FP16 for VAE. Native
stores FP16 directly. The raw intermediates differ, while the consumed FP16
boundary and final frames match exactly; see [raw stage comparison](seedvr2_native_evidence/raw-stage-parity.json).
Audio is re-encoded as AAC. The valid0.2-second fixture interval has correlation
0.9967; AAC decoded padding is outside the container duration.

At848×480, SP1 diagnostic model time is1285.75ms: encode301.26ms,
DiT295.53ms, decode684.15ms; peak allocated13.267GiB. These are diagnostic
samples, not formal speedup measurements. VAE runs on every SP rank.
[Per-rank stages/resources](seedvr2_native_evidence/stage-resource-results.md).

The aligned B case107 frames448×256→896×512 OOMs during whole-clip VAE
encode:43.14GiB allocated plus11.92GiB requested exceeds L20 capacity.
A following short request succeeds; cached reserved memory remains until
service shutdown. The larger4× long-video case has no capacity/speedup claim.
Practical WF-07 P0 batch5/overlap1/LAB/swap32/tiled-VAE reference semantics
are distinct from C0. [B case manifest](seedvr2_native_evidence/b-case-manifest.json).

[Model contract and release checks](seedvr2_native_evidence/model-contract.md),
[pinned reference identity](seedvr2_native_evidence/reference-provenance.md),
[review ownership](seedvr2_native_evidence/review-audit-7739.md).

## Reproduce

Install the repository and PyAV in the same Python environment. Prepare the
[model directory](../../docs/models/seedvr2.md), two separate checkouts containing
this integration, and set `BASELINE`, `CANDIDATE`, `MODEL_DIR` and a free `GPU`.
For the RoPE comparison, replace only baseline `rope.py` with the version from
9a62e04ea8e4bca41d69ad75c07322db13a35f82; retain4dc1bfb's version in candidate.

```bash
curl -L --fail -o Eyes_212x120.mp4 \
  https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/example_workflows/example_inputs/Eyes_212x120.mp4
python benchmarks/diffusion/create_seedvr2_native_fixture.py \
  Eyes_212x120.mp4 five-frames-audio.mkv
# Select valid CPU IDs for your machine; use separate output directories.
taskset -c 48 python benchmarks/diffusion/benchmark_seedvr2_native_http.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --models "$MODEL_DIR" \
  --input five-frames-audio.mkv --output native-aa --gpu "$GPU" \
  --server-cpus 40-47 --pilot
taskset -c 48 python benchmarks/diffusion/benchmark_seedvr2_native_http.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --models "$MODEL_DIR" \
  --input five-frames-audio.mkv --output native-formal --gpu "$GPU" \
  --server-cpus 40-47
```

The source clip SHA256 is
`ddf5d6c0710fb37336b8675b89f9336c466cb9e7c067101b24091668d0dba833`.
The fixture generator takes the first five decoded RGB frames losslessly and
adds3200 mono PCM samples of a440Hz tone at16kHz. Matroska container IDs can
vary between encodes; compare decoded content, and record each fixture hash.
The portable driver is a refactor of the original measured driver, with an
independent fresh-service pilot (246.767 vs245.860ms,0.37% drift, both
normal exit0). It records executed worker module paths/hashes,
GPU state, every sample and process shutdown. Check pilot drift before running
formal measurements; a successful process exit alone is not a performance gate.
