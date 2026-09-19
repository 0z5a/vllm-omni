# HiDream existing implementation audit

Validation base: `698f716`. These are external implementations by `dpeng123`; their runtime changes are retained for validation and are not re-authored as new production changes.

| Scope | Existing PR | Immutable head | Application to validation base | CPU evidence |
|---|---|---|---|---|
| B1 TeaCache | [5408](https://github.com/vllm-project/vllm-omni/pull/5408) | `e857c8c7cfd21478c119f8a3dcbcec846f181e5b` | Context drift; transplant preserves all three added extractor functions exactly by AST | 11 passed; 29 unrelated cases deselected |
| B2 CFG parallel | [5409](https://github.com/vllm-project/vllm-omni/pull/5409) | `7f1d09e074e5ab96f0927430c4303e604c4befe2` | Runtime and CPU test apply cleanly | 9 passed |
| B3 HSDP | [5410](https://github.com/vllm-project/vllm-omni/pull/5410) | `6bd897750bf561ecbb43c5b42e59384514e3ec6d` | Runtime and CPU test apply cleanly | 3 passed |

The original API file patches and separately adapted TeaCache patch are retained. Documentation/example hunks are excluded from execution snapshots. Each snapshot is separate from the other candidates and from live GPU workloads.

CFG changes separate positive/negative embeddings, delegate prediction to the existing CFG mixin, and preserve the negative sign on model noise. The CPU cases cover dispatch and conditioning but do not prove distributed noise-prediction equivalence or complete model output parity. HSDP adds only top-level double/single stream block matching; its CPU cases check exact matching and reject nested children. They do not prove actual sharding, loading fit or memory reduction. TeaCache introduces an extractor and fixed coefficients; real no-hit equivalence, observed hits, request resets and held-out quality calibration remain necessary.

The previous task's `l20-admission-20260918.md` reports complete-model TP2 and TP4 construction OOMs with four replicated text encoders and mostly replicated MoE feed-forward weights. That historical report uses another source/environment and is retained as a loading-budget warning, not a fresh result for these immutable candidates. Current base still directly places T5 and Llama on the local device and does not declare the generic component/offload interface. These full-model gates remain open; no encoder is removed or replaced for a passing result.

Shared real checkpoint revision: HiDream `8ccbbfb270ccdae26d6bb0081df67dc81e4033bf`; auxiliary Llama mirror `d10aef7999a2b5ba950ab3974312feeedbfe0b77` according to the prior admission record. Existing weight directories are reused; no duplicate download or shared-weight cleanup is performed.

There is no speedup claim or completed E2E table for B1–B4 yet. Production VAE candidate `f6098af` remains independent of these three PRs.
