# NIXL baseline on two H20 GPUs

Measured on 2026-10-01 against vLLM-Omni
`27a6321a6311870fe68c2c9c13b694f2612ff779`, before adding page transfer.
The validation harness records the exact checkout, imports, package freeze,
GPU topology and JUnit results, and rejects missing or skipped native cases.

| Component | Recorded configuration |
| --- | --- |
| GPUs | 2 × NVIDIA H20, 97,871 MiB each |
| Interconnect | Same host, NV18 between GPU 0 and GPU 1 |
| Driver | 580.105.08 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM / NIXL | 0.30.0 / 1.3.0 |
| Transformers | 5.14.1 |

| Baseline suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Omni NIXL connector | 102 | 0 | 0 |
| Diffusion KV layout | 16 | 0 | 0 |
| Two-process native CUDA transfer | 8 | 0 | 0 |

The eight native cases cover both metadata paths for `generic`, `h3_ref2va`,
`h3_fl2va` and `h3_t2va`. They verify exact structured tensor/semantic
round trips, source retention before the terminal READ, and release after ACK.
They use synthetic conditioning tensors; these results do not represent a
MiniMax model E2E run.

Run the suites serially because the CPU connector fixtures use a fixed port:

```bash
UCX_RCACHE_MAX_UNRELEASED=1024 python tools/validate_nixl.py \
  --repo "$PWD" --out /tmp/nixl-baseline
```

The native test harness joins children and leaves a timed-out child running
for inspection. It does not terminate or kill children.
Recorded evidence is in
[`benchmarks/nixl/h20x2-20261001`](../../../../benchmarks/nixl/h20x2-20261001/):
`baseline-summary.json`, `baseline-native.xml`, `baseline-manifest.json`,
`pip-freeze.txt` and `topology.txt`.
