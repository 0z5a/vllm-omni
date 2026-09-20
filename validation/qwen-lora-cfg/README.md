# Qwen Edit CFG × trained LoRA E2E preparation

Status: harness prepared; full model admission and GPU execution pending. This is K5.5 preparation, not a passing E2E result.

Pinned source `3ea51522ba780697f3670cadfdb117781b3ea7f3` contains the Qwen output-projection loader fix. The verified full Qwen-Image-Edit-2509 checkpoint and two immutable trained adapters (Edit-R1 and CoCoEdit) are already present; see `../lora-offload-preparation`. Both contain 960 tensors and differ in 957 tensor values.

Both arms use BF16 eager execution, model-level CPU offload, native VAE tiling/slicing, one reference image, 50 denoising steps, true CFG scale 4, seed 142 and identical positive/negative prompts. Native uses CFG1 on one GPU; the candidate uses CFG2 on two GPUs. TP, SP and VAE parallelism are not added. Full-model startup and peak memory must pass on the authorized L20s before performance conclusions. Native versus CFG2 costs must include the different allocated GPU counts.

Each fresh process runs 512×512, 768×512 and 512×512 reentry. For each shape, the adapter sequence is A → A → B → None → A → A(scale=0) → B(scale=0) → A. Probe processes run one cycle per shape; timing processes run two warmup and five measured cycles. Latency includes adapter switching. Final comparison requires independent A/P/P/A processes and PNG error metrics; no speedup numbers exist yet.

The untimed probe composes the existing worker activation/projection observer with Qwen transformer conditioning hashes. It verifies actual CFG group size, 100 transformer calls per 50-step native request or 50 per CFG rank, and separate positive/negative embeddings. `verify_cfg.py` checks each rank against native alternating branch embeddings, rank agreement of adapter IDs/scales, and the existing 24-request image restoration gates. The gate also requires identical PNGs across CFG1/CFG2 because this configuration changes branch placement without changing the arithmetic. Semantic image quality is not inferred from conditioning hashes.

The symlinks reuse the existing probes and image verifier. Stage a dereferenced, hash-verified copy before execution. The new controller waits for the existing LoRA cache controller to finish successfully, then runs both probes before the full quartet. It does not interrupt the current GPU queue. Each full arm allows 24 hours and retains all requests; startup allows 1800 seconds.
