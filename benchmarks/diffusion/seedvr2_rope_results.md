# SeedVR2 RoPE host synchronization: validation results

Follow-up to vllm-project/vllm-omni#7739 and issue #7723. The production change replaces GPU-derived RoPE cache keys with the known length of zero-based, unit-stride positions. The offline benchmark uses the reference VAE and sampler; it does not add native serving or claim ownership of VAE integration.

## Five-frame E2E comparison

Baseline: `9a62e04ea8e4bca41d69ad75c07322db13a35f82`. Both arms use native NaDiT, the same reference VAE, released 3B FP16 weights, one Euler step, CFG=1, seed 7723, and grouped SDPA. The only production difference is `rope.py`.

Input: the first five frames of `Eyes_212x120.mp4`, 25 fps. Output: 224×128. One NVIDIA L20 (46068 MiB reported memory), physical GPU 5, on a shared host. PyTorch 2.13.0+cu130, CUDA runtime 13.0, driver 595.91.07, vLLM 0.29.0, diffusers 0.40.0, PyAV 18.1.0.

Eight fresh processes in A–P–P–A–P–A–A–P order; one warmup and nine measured requests per process. Time includes video read, resize, VAE encode, the full 32-layer NaDiT, VAE decode, and MP4 encode/write. It excludes model loading and warmup. Both arms write to tmpfs to avoid the observed shared-disk write congestion. All samples are retained.

| Pair | Baseline median (ms) | Candidate median (ms) | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 198.868 | 185.296 | 1.0732× | 6.82% |
| 2 | 200.506 | 186.884 | 1.0729× | 6.79% |
| 3 | 200.640 | 187.403 | 1.0706× | 6.60% |
| 4 | 309.582 | 279.129 | 1.1091× | 9.84% |
| Geometric mean of process medians | 223.085 | 206.303 | 1.0814× | 7.52% |

All four pairs improve. Pair 4 is slower in both arms; it remains in the aggregate. GPU 5 background memory rose from 24828 MiB before process 5 to 33976 MiB before process 7 (100% utilization at the latter snapshot). These are paired latency results on a shared GPU, not a concurrency-throughput or isolated-hardware claim.

## Raw measured milliseconds

| Process | Nine measured requests (ms), in order |
| --- | --- |
| 0-baseline | 279.227, 194.822, 195.991, 198.868, 199.213, 201.819, 198.368, 199.176, 197.412 |
| 1-candidate | 189.677, 188.374, 190.121, 184.100, 185.296, 182.534, 183.288, 184.007, 185.581 |
| 2-candidate | 187.178, 185.730, 185.772, 184.578, 186.621, 187.404, 187.908, 189.622, 186.884 |
| 3-baseline | 201.259, 203.023, 200.506, 201.828, 198.709, 199.509, 199.647, 198.388, 209.026 |
| 4-candidate | 185.607, 186.361, 190.278, 191.514, 187.007, 189.185, 190.728, 187.403, 184.158 |
| 5-baseline | 195.949, 200.640, 196.239, 201.098, 200.193, 201.917, 199.687, 202.369, 202.293 |
| 6-baseline | 321.775, 297.510, 299.795, 314.494, 309.582, 282.825, 296.373, 330.374, 339.449 |
| 7-candidate | 281.474, 274.150, 272.011, 274.810, 279.129, 282.695, 299.996, 278.216, 296.595 |

## Correctness and local checks

- All four pairs are bitwise equal for VAE condition, NaDiT velocity, restored latent, and unencoded output frames.
- 80/80 MP4 files have five frames, 224×128 dimensions, and 25 fps.
- 566 CPU tests pass (14 existing dependency deprecation warnings). The three new RoPE tests fail against the baseline and pass with this change.
- Ruff check/format, SPDX, forbidden-import, CUDA-call, test-marker, and Python 3.10 mypy checks pass for the applicable changed files. Model and benchmark paths follow the existing mypy wrapper exclusions.

## Supplemental cases

Each row below is one fresh baseline/candidate process pair, one warmup plus five measured requests per process. These are supplemental checks, not additional independent confirmation pairs for the headline result. The GPU 5 4× baseline encountered OOM after other processes occupied about 33 GiB; its failed log is retained. All supplemental cases were then run on GPU 2 with unchanged model, dtype, step count, and dimensions.

| Case | GPU | Baseline median (ms) | Candidate median (ms) | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 frames, 432×240 (busy card) | 5 | 728.322 | 693.875 | 1.0496× | 4.73% |
| 5 frames, 432×240 (≈2×) | 2 | 435.147 | 423.945 | 1.0264× | 2.57% |
| 5 frames, 848×480 (4×) | 2 | 1467.095 | 1413.362 | 1.0380× | 3.66% |
| 6 frames, 240×144; pad to 9 | 2 | 289.393 | 275.680 | 1.0497× | 4.74% |

All supplemental SP1 stage tensors are bitwise equal. Across the full campaign, 146/146 MP4 files pass frame-count, dimension, and frame-rate checks.

| SP degree | Velocity relative L2 vs SP1 | Frame mean absolute error | Frame max absolute error |
| --- | ---: | ---: | ---: |
| SP2 | 0.00170679 | 0.00017157 | 0.00387573 |
| SP4 | 0.00170679 | 0.00017157 | 0.00387573 |

SP gates fixed before testing: exact condition; velocity/restored-latent relative L2 ≤0.01; frame mean absolute error ≤0.02 and max ≤0.25; finite tensors. SP measurements validate correctness, not scaling performance.

| Supplemental process | Five measured requests (ms), in order |
| --- | --- |
| 2x/baseline | 742.594, 723.926, 728.322, 727.734, 729.768 |
| 2x/candidate | 687.154, 719.348, 693.875, 684.294, 702.086 |
| supplement-gpu2/2x/baseline | 470.815, 448.321, 433.081, 434.858, 435.147 |
| supplement-gpu2/2x/candidate | 426.541, 421.796, 423.945, 424.103, 421.817 |
| supplement-gpu2/4x/baseline | 1521.842, 1465.562, 1462.154, 1467.095, 1484.498 |
| supplement-gpu2/4x/candidate | 1410.899, 1413.362, 1414.294, 1412.803, 1416.985 |
| supplement-gpu2/boundary/baseline | 292.247, 289.393, 288.430, 288.912, 292.680 |
| supplement-gpu2/boundary/candidate | 276.398, 276.302, 275.680, 274.421, 274.820 |

## Profile mechanism

A separate profiled five-frame request on physical GPU 2 shows the expected removal of 448 scalar-read synchronizations. Profile timings are excluded from the latency comparison.

| Trace event | Baseline count | Candidate count |
| --- | ---: | ---: |
| `aten::_local_scalar_dense` | 702 | 254 |
| Device-to-pinned-host copies | 693 | 245 |
| Device-to-pageable-host copies | 99 | 99 |

In the fresh profiled processes, peak allocated GPU memory is unchanged at 7,854,314,496 bytes (reserved: 8,076,132,352 bytes in both arms).

The new regression is collected by the existing L1 CPU diffusion test job in `.buildkite/cuda/test-ready.yml`; no CI routing change is needed.

## Reproduction

Requires a compatible vLLM-Omni environment with PyAV, OmegaConf, safetensors, torchvision, and the reference SeedVR dependencies (including rotary-embedding-torch 0.9.1). Keep one fixed SeedVR reference snapshot and checkpoint set for both arms. The historical reference snapshot has no verifiable Git revision; checkpoint hashes and a source-file manifest are retained with the raw evidence.

| Artifact | SHA256 |
| --- | --- |
| `seedvr2_ema_3b_fp16.safetensors` | `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304` |
| `ema_vae_fp16.safetensors` | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |
| `Eyes_212x120.mp4` | `ddf5d6c0710fb37336b8675b89f9336c466cb9e7c067101b24091668d0dba833` |
| `pos_emb.pt` | `fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799` |
| VAE YAML config | `88135a1dd1d7fe0558c1d52633bb251cb3a86b420b98c416a735ab768e0f929d` |

Set `CHECKOUT` to the baseline or candidate checkout, `CANDIDATE` to the candidate checkout, `SEEDVR_REF` to the SeedVR reference checkout, `MODELS` to the matching checkpoints, `VIDEO` to the input MP4, and `NEW_OUTPUT` to a new tmpfs directory. Run the same candidate benchmark script against each checkout:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$CHECKOUT:$SEEDVR_REF" python "$CANDIDATE/benchmarks/diffusion/benchmark_seedvr2_restoration.py" --reference "$SEEDVR_REF" --models "$MODELS" --input "$VIDEO" --output "$NEW_OUTPUT" --frames 5 --height 128 --width 224 --repeats 9
```

CPU regression (from the candidate repository, no weights needed):

```bash
python -m pytest tests/diffusion/models/seedvr2/test_rope.py -q
python -m pytest tests/diffusion/models/seedvr2/test_rope.py tests/diffusion/models/seedvr2/test_window_attention.py tests/diffusion/models/seedvr2/test_checkpoint_contract.py tests/diffusion/models/seedvr2/test_window_sp_plan.py -q
```

Optional RoPE-only diagnostic, excluding all other restoration work:

```bash
CUDA_VISIBLE_DEVICES=5 python -m benchmarks.diffusion.benchmark_seedvr2_rope --repeats 20
```
