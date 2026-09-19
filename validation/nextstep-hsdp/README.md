# NextStep HSDP call coverage

Author: 0z5a. Baseline `698f716`; candidate `9545f40`.

The pipeline now enters the model through its module call for prefill and cached decode. Prefill embeds input IDs inside that call, so the root FSDP state initializes before the embedding child. Decoder layers, embedding, image projectors and the flow-head network have separate shard boundaries; checkpoint loading, cache updates and sampling equations are unchanged.

The first two-rank attempt exposed an FSDP initialization error when embedding ran before the root. The retained failure log documents that rejected intermediate implementation. The corrected candidate passed the following tiny-model BF16 checks on physical GPUs 2 and 3, with actual NextStep modules and `StaticCache`:

| Prefix length | CFG branches | Decode steps | Ranks | Hidden states and sampled tokens | RNG |
|---:|---:|---:|---:|---|---|
| 4 | 1 | 3 | 2 | Exactly equal | Equal |
| 7 | 2 | 3 | 2 | Exactly equal | Equal |
| 6 | 3 | 3 | 2 | Exactly equal | Equal |
| 4, repeated | 1 | 3 | 2 | Exactly equal | Equal |

The probe also checks seven FSDP modules (six children plus root) and that every parameter is sharded after each request. CPU CFG regressions: 10 passed. All changed-file pre-commit checks passed, including mypy.

## Remaining full-model validation

| Measurement | Baseline | HSDP candidate | Speedup |
|---|---|---|---|
| Full prompt → AR generation → VAE image latency | Pending | Pending | Pending |
| Prefill latency | Pending | Pending | Pending |
| Cached token latency, including all-gathers | Pending | Pending | Pending |
| Peak process GPU memory | Pending | Pending | Pending |

No tiny-model timing is presented as E2E performance. The approximately 60 GB checkpoint cannot be assumed to fit a single L20. A two-GPU TP baseline and two-GPU HSDP candidate compare different parallel strategies, so that comparison must be labeled accordingly. Both strategy arms use candidate source `9545f40`; the TP arm is the runtime reference, not an earlier HSDP implementation. Real-checkpoint image quality, loading peak, long-cache behavior and AR all-gather cost remain unverified.

NextStep checkpoint revision `05486ff90769ad32109d219fd4659d41671a770a` is downloaded. All 13 safetensors shards were opened and their 691 tensor keys checked against the index; total shard size is 59,815,331,784 bytes. The VAE checkpoint is present but has not yet been loaded in this validation. Weights remain required for pending E2E.
