# Trained LoRA prerequisites for K5

Two public trained Z-Image Turbo adapters were downloaded before GPU admission. They supplement the existing synthetic rank-16 fixtures; synthetic fixtures alone do not establish real-adapter E2E coverage.

| Adapter | Immutable revision | Rank / alpha | Safetensors bytes | Tensors | Verification |
|---|---|---:|---:|---:|---|
| [Helio](https://huggingface.co/HelioAI/Helio-Z-Image-Turbo-LoRA/tree/38826969e8cb78f91d963a1cf7910e7d14749363) | `38826969e8cb78f91d963a1cf7910e7d14749363` | 64 / 128 | 284,151,840 | 408 | Size and SHA-256 match upstream LFS object |
| [Fuli](https://huggingface.co/DownFlow/Z-Image-Turbo-Fuli-LoRA/tree/b1216f23d98df471e7b7c12dc662dae958b2c187) | `b1216f23d98df471e7b7c12dc662dae958b2c187` | 32 / 32 | 284,151,432 | 408 | Size and SHA-256 match upstream LFS object |

The repository API identifies both as ungated. Their model cards identify Z-Image Turbo as the base; this is provenance, not proof of compatibility with the pinned runtime. The weight keys use PEFT `base_model.model` prefixes and include context-refiner attention projections. Loading, target matching, alpha/rank scaling, dtype, adapter switching and cache reset must be verified on the actual runtime before performance claims.

Task-owned directories are `/home/kxqandccx/omni-1217-20260919/models/lora/helio` and `/home/kxqandccx/omni-1217-20260919/models/lora/fuli`. Only README, adapter configuration and safetensors were fetched. No repository code was executed. Keep these weights for the pending K5 cases and remove them after their tests and evidence are complete. The shared base checkpoint is not task-owned.

K5's twelve subcases remain distinct. The same adapters cannot establish EP or CFG capability where the base model does not support it. The planned A→A→B→None→A sequence and zero/normal scale controls still require complete execution and quality/performance reporting; no subcase is marked passed by this download.
