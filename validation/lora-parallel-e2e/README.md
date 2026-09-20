# K5.3, K5.4, K5.6, K5.7: trained LoRA × parallel execution

Full GPU validation is pending. Ruff, shell syntax, actual runtime imports and all five configuration world sizes passed; seven remote script hashes matched (`import.log`). This harness uses source `698f716`, the existing ZImage checkpoint, and the Helio/Fuli trained adapters verified in `../lora-model-preparation/`. It introduces no production implementation or replacement checkpoint.

Each feature independently compares native one-GPU execution against Ulysses2, Ring2, TP2, or HSDP2 on authorized GPUs 2,3. Fresh process order is A/P/P/A. Every process runs 512×512 → 768×512 → 512×512 reentry, two warmup cycles and five timed cycles. Each cycle requests A → A → B → None → A → A(scale 0) → B(scale 0) → A: 168 requests per timing arm. BF16 eager, 9 steps, seed 142. GPU-seconds/output must account for one GPU versus two.

Separate 24-request probes precede timing. The worker verifies actual LoRA activation and projection execution, buffer/input dtype and device alignment, actual attention strategy and Q/K/V/output shapes, subgroup sizes, post-request DTensor residency and preserved FSDP module count. Raw LoRA tensor hashes are compared against the native replica or exact native TP partition; this checks trained values, not just compatible shapes. First-block RoPE and input-mask tensors must match their native replicas or exact SP halves. Each rank retains request IDs, source paths/hashes and peak allocated/reserved memory.

The existing ZImage attention passes no explicit mask to its backend for this single-prompt path. Capturing the block input mask checks partition transport; it does not prove general masked-attention support. Multi-prompt masking is outside this workload and must not be inferred from a successful run.

Gates also verify all PNG hashes, repeated/restored A, zero-scale equivalence to None, visible adapter effects, shape reentry, and matching adapter states across ranks. Cross-topology image differences and quality are reported separately; exact adapter partitions do not guarantee pixel-identical generation or acceptable quality. Timing output is excluded from the intrusive probes.

The waiter starts only after the existing FP8/HSDP matrix succeeds. An unsuccessful probe retains its evidence and stops the subsequent timing arms; it is not silently relabeled unsupported or passed. All twelve K5 subcases remain in scope, including the other pending combinations.
