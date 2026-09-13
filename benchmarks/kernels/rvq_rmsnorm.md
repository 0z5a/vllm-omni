# Native-order RVQ and first RMSNorm experiment

The opt-in A13 primitive uses two Triton kernels. The first reduces 16 embedding
rows in native accumulation order into a rounded raw residual, using 128-channel
tiles. The second reads that residual directly in the native RMS reduction lanes
and writes normalized values. The raw residual remains observable, and the cast
before live gamma multiplication is preserved.

The experiment remains under `benchmarks/kernels`. Model defaults and production
consumers are unchanged. The previous one-program region prototype is superseded;
its 1.638–58.314% region measurements do not describe A13.

## Complete Code2Wav validation

The archived A13 r2 run used one RTX PRO 4000 Blackwell 24 GB (SM120), Torch
2.13.0+cu130, CUDA 13.0, Triton 3.7.1 and Transformers 5.14.1. The checkpoint
revision is `12a3ac6661e685620bbc6616b7217b26469a81dd`; all three required files,
230 tensors and the HF implementation were hash-verified. All eight Transformer
layers and the complete waveform decoder execute. Inputs are seeded synthetic
codec IDs, so this is complete codec-to-waveform execution, not full text/audio
Omni request E2E.

The baseline is compiled RVQ followed by native first RMSNorm and the shared
complete consumer. A prior whole-region compiled baseline failed waveform
correctness in two cases and was rejected. Both arms use the same isolated mask
adapter for capture; the installed HF forward is unchanged and is the output
reference. Initial r1 timings overlapped other work and were rejected; only the
fresh r2 cohort is reported.

All 24 original/changed raw-and-normalized stage cases and all 72 original,
changed-input and restored complete-waveform comparisons are bit-exact. Four
semantic GPU groups pass without skips. The test retains its original
`atol=rtol=0.002` and reports the stronger observed bit equality separately.
All 768 timing arms and 48 path checks completed. The candidate traces contain
both `_rvq_native_mean_kernel` and `_native_first_rmsnorm_kernel`; neither occurs
in the baseline. Graph traces verify replay independently of timing.

Each case/mode has three warmup calls and eight alternating APPA/PAAP blocks,
with ten complete forward calls per timing arm. Synchronized wall time includes
the complete waveform consumer and excludes compilation, model loading, warmup,
input copies and profiling. Positive values below mean lower candidate latency.

| Dtype | Frames | Strided codes | Eager latency reduction | Graph latency reduction |
|---|---:|---|---:|---:|
| float16 | 1 | False | -0.091675% | +0.470098% |
| float16 | 1 | True | +0.351706% | +0.193401% |
| float16 | 32 | False | +0.565889% | -0.229305% |
| float16 | 32 | True | +0.053497% | -0.200676% |
| float16 | 128 | False | +0.117682% | +0.030157% |
| float16 | 128 | True | +0.050532% | +0.009659% |
| bfloat16 | 1 | False | +0.541099% | +0.341701% |
| bfloat16 | 1 | True | +0.378751% | +0.271446% |
| bfloat16 | 32 | False | +0.447490% | -0.225574% |
| bfloat16 | 32 | True | +0.727869% | -0.190366% |
| bfloat16 | 128 | False | +0.071706% | +0.048593% |
| bfloat16 | 128 | True | +0.061728% | +0.043783% |

There are **19 positive and 5 negative scenarios**. All four T32 Graph scenarios
regress by about 0.19–0.23%; FP16/T1/contiguous eager also regresses. No causal
attribution for these small differences has been established. These are
single-cohort descriptive observations, with no confidence interval or stable
all-scenario speedup claim. Peak VRAM was not recorded in this cohort.

## Reproduce and review

[The complete frozen runner, model/source hashes, stage checks and every timing arm](https://gist.github.com/0z5a/32cd168d1ed19b4bf86f9a75596d788c)
are available as one archive. Its README gives the exact frozen-runtime command,
source layout and checkpoint checks. The `performance_accepted` field only means
that timing passed correctness/provenance/path gates; `all_speedups_positive`
is false and records the failed performance goal.

The repository also retains a small generated-parameter region diagnostic:

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 python -m benchmarks.kernels.bench_rvq_rmsnorm --output rvq-results.json
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 pytest tests/model_executor/layers/test_rvq_rmsnorm_gpu.py
pytest tests/model_executor/layers/test_rvq_rmsnorm_cpu.py
```

That region runner compares against whole-region compiled PyTorch and does not
reproduce the checkpoint-derived Code2Wav table above. Unsupported input metadata
uses native arithmetic. Bit-exact behavior is established only for the frozen
runtime and tested cases; other shapes retain the numerical regression contract.

Publication preserves all three Triton function ASTs, arithmetic and launch
arguments from the validated source. It adapts imports/current-device lookup to
project APIs and adds a frozen-runtime exact-output regression. The public wrapper
and new regression have not received a fresh GPU run. CPU metadata checks and
changed-file hooks are reported separately in the PR.
