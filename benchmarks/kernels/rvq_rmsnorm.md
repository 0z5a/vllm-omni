# Dual-output RVQ and first RMSNorm fusion

This opt-in benchmark fuses a 16-quantizer embedding mean with the first input
RMSNorm. It returns both the low-precision raw residual and normalized values.
The residual is rounded before FP32 RMS statistics, and normalization is cast
before multiplication by the live gamma. Keeping both outputs observable is
necessary to represent the complete consumer region.

The primitive is isolated under `benchmarks/kernels`; no model default or
Transformer consumer is changed. This is a proposal for a reusable fusion
boundary, not a claim about complete text-to-audio requests.

## Reproduce

Install vLLM-Omni with an SM120-capable CUDA runtime. The measured environment
used Torch 2.13.0+cu130, Triton 3.7.1 and an NVIDIA RTX PRO 5000 Blackwell.
The benchmark checks SM120 and the recorded Triton version explicitly.

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 python -m benchmarks.kernels.bench_rvq_rmsnorm --output rvq-results.json
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 pytest tests/model_executor/layers/test_rvq_rmsnorm_gpu.py
pytest tests/model_executor/layers/test_rvq_rmsnorm_cpu.py
```

Both timing arms use `torch.compile(fullgraph=True, dynamic=False)` and
`emulate_precision_casts=True`. Four warps are fixed for all confirmation
cases. The matrix covers FP16/BF16, 1/32/128 frames, contiguous/strided codes,
hidden width 1024, 16 codebooks with 2048 entries each, and epsilon 1e-5.
Gamma and table values are generated, not extracted from a model checkpoint.
Each case measures eager and Graph execution in three paired quartets,
retaining all 288 timing arms across 24 case/mode combinations.

## Recorded result and numerical contract

The archived confirmation improved all 24 case/mode combinations by
**1.638–58.314%** relative to compiled PyTorch over this complete dual-output
region. These are descriptive paired reductions, not full-model speedups or
confidence intervals. The checked-in runner exposes the same matrix; its new
publication entry point has not been timed on a GPU.

Finite outputs are compared with `atol=rtol=0.002`; NaN, positive/negative
infinity and finite masks must match exactly. This fusion does not promise
bitwise equality. GPU regressions cover hidden tails, changed live buffers,
nondefault streams, Graph replay, nonfinite values, lazy views, mixed gamma
dtype, autocast, gradients and empty inputs. Unsupported metadata uses the
native arithmetic, preserving dtype promotion and gradient behavior.

The kernel arithmetic and launch arguments are preserved from the tested
source. Publication changes use project Triton imports and the platform
current-device API, format the source, and provide a standalone matrix runner.
New hardware, compiler versions and production consumer integration require
their own numerical and complete-model validation.
