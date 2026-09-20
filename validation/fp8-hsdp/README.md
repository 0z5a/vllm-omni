# K4: online FP8 × HSDP

Prepared against source `698f716`. Existing FP8/HSDP CPU contracts passed: **6 tests**, retained in `cpu.log`. GPU admission, kernel dispatch, image quality and speed remain pending. No new quantization implementation is introduced.

All four configurations keep Ulysses degree 2 on GPUs 2,3: BF16, online FP8, BF16+HSDP2, online FP8+HSDP2. The public `quantization_config="fp8"` converts the same BF16 checkpoint at load time. This does not validate offline FP8 checkpoint formats.

Each fresh process runs 512×512 → 768×512 → 512×512 reentry with seed 142, 9 steps, eager execution, two warmups and five timed requests per shape. Balanced timing order is base/FP8/HSDP/both/both/HSDP/FP8/base. PNG encoding and saving occur after the timed generate call.

Four separate untimed probes run first. They observe actual scaled FP8 GEMM calls, activation/weight dtype, column-major GEMM weight layout, finite scales, concrete kernel class, per-layer calls, post-request DTensor residency, and per-rank peak allocated/reserved memory. Missing configured FP8 execution or incorrect sharding stops the matrix. Probe synchronizations and memory resets are excluded from timing arms.

Report BF16 versus HSDP, FP8 versus BF16, and FP8+HSDP versus FP8 separately. Quantization drift must not be conflated with additional sharding drift. Retain original images and hashes, RGB error/PSNR, repeat and reentry stability, medians, speed ratios, and GPU-seconds per output. A successful kernel probe alone does not establish acceptable image quality or a speed gain.

The serial waiter requires the preceding Qwen LoRA/CFG quartet to finish successfully before starting. It does not preempt another job. Shared base weights remain in place; this harness creates no model copy.
