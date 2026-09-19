# Ovis request weight residency validation

Author: 0z5a. Hardware: two NVIDIA L20 46,068 MiB GPUs (physical 2,3), shared host.

[Speed comparison](speed-comparison.md) reports the complete pre-rebase N/A/P/P/A/N run, including the no-HSDP reference and all per-arm ranges. The result is observational because unrelated processes were active during A0/A1. All 84 saved PNGs were verified against their recorded SHA256.

| Source | Revision |
|---|---|
| Original main | `232dbc82f8fcb6cb698860b0210b47d60a13ebdc` |
| Original HSDP | `1839c0888c5755537e2343666d3d18d045f055f2` |
| Pre-rebase request residency | `6db11a1` |
| Current main | `698f716` |
| Rebased request residency | `97d2a756f03cce3d1abb4a9691e1077a8cac2d90` |
| Checkpoint | `ATH-MaaS/Ovis-Image-7B`, revision `41be1c5821a92c970d63d7eb595a2fd3fe32b22e` |

The pipeline SHA256 is unchanged by rebase and matches both remote source snapshots: `0ae03d3a7ef432ff264f5c5dfd07819314eba99630932fda9fa5070977c22516`.

Runtime: PyTorch 2.13.0, vLLM 0.29.0, diffusers 0.40.0, transformers 5.14.1, Triton 3.7.1. Omni is loaded from the frozen source trees via PYTHONPATH; installed distribution metadata is not the source revision.

| Mode | Maximum device memory GPU 2 (MiB) | Maximum device memory GPU 3 (MiB) |
|---|---:|---:|
| No HSDP | 22,704 | 22,704 |
| Original HSDP | 20,189 | 18,341 |
| Request residency | 31,779 | 31,382 |

Memory is 1-second sampled device-wide NVML over startup and inference, including unrelated processes. It is not allocator-only memory or a separated steady-state estimate. The residency option trades active memory for fewer all-gathers.

Rebase validation: all changed-file pre-commit gates pass; 3 CPU lifecycle tests pass; two-rank CUDA sequence 16/24/16 outputs are exact and exception cleanup restores shards. The first full-model repeat completed N0 but A0 hit OOM when unrelated PID 2156826 occupied 33.76 GiB on GPU 2. This incomplete run must not be included in speed calculations. A fresh run waits for the user-reserved GPU 2,3.

Reproduce the existing summary with `python summarize.py prerebase`. Raw measurements, logs, process telemetry and PNGs are stored in `prerebase/<arm>/`.
