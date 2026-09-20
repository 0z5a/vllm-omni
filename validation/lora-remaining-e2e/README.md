# K5.9-K5.12 full E2E matrix

Status: prepared and queued after the K6 matrix. No GPU result is claimed before `evidence/remaining-cases.status` and generated `evidence/RESULTS.md` pass verification.

Pinned source: `698f7160125d3071b8c0eef69b1b03fa8dfba766`. The matrix uses the real Z-Image base and the pinned Helio/Fuli trained adapters already retained for K5.1-K5.7. GPU use is limited to devices 2 and 3.

| Subcase | Reference | Feature arm | Required runtime evidence |
| --- | --- | --- | --- |
| K5.9 | native, 1 rank | layer offload, 1 rank | Actual layer prefetch/offload events; injected LoRA projection executes on the input device |
| K5.10 | native, 1 rank | module offload, 1 rank | Actual CPU and CUDA module moves include the wrapped adapter parameters |
| K5.11 | Ulysses2 + native tiled VAE | Ulysses2 + VAE parallel size 2 | Both ranks execute the distributed VAE executor and agree on every request |
| K5.12 | BF16, 1 rank | online FP8, 1 rank | Actual FP8 scaled-MM kernels execute while LoRA accumulation remains dtype/device-correct |

Every probe and timing arm runs `A → A → B → None → A → A(scale=0) → B(scale=0) → A` at 512×512, 768×512, then 512×512 reentry. Probes validate adapter IDs, ranks, tensors, offload moves, VAE work, and FP8 kernels. Timing uses two warmup and five measured cycles in two fresh feature processes bracketed by two fresh reference processes. `summarize.py` verifies all PNG hashes before writing the required Markdown speed, GPU-cost, stability, and RGB quality tables.

Remote run directory: `/home/kxqandccx/omni-1217-20260919/lora-remaining-validation`. The queued controller waits for the already-running K6 controller and requires its terminal success marker before using GPUs.
