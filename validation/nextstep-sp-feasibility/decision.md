# NextStep sequence-parallel feasibility decision

Decision: **DEFER_NO_SIGNAL** for the measured full-checkpoint 512² workload. No runtime SP patch is submitted.

| Full-checkpoint phase, steady request | Rank 0 | Rank 1 | SP implication |
| --- | ---: | ---: | --- |
| Text/condition prefill | 0.053191 s | 0.053477 s | Only candidate sequence boundary; 24 queries with CFG batch 2 |
| Autoregressive backbone | 44.759123 s | 44.024192 s | Each of 1,024 decode calls has query length 1 |
| Flow-matching head | 50.765098 s | 51.498393 s | Conditioned MLP path; no image-sequence attention axis |
| VAE decode | 0.046581 s | 0.046604 s | Separate C3 scope |
| Prefill share | 0.0556% | 0.0559% | Too small for useful E2E acceleration |
| Zero-cost prefill upper bound | 1.000557× | 1.000560× | Excludes any partition, gather, or KV-layout cost |

The profile used source `43c4e79`, the real checkpoint, TP2, a fixed seed, a 512×512 output, and a complete token-generation→VAE request. Shape probes show that the steady prefill starts with query shape `2×24`, then the cache grows from 24 to 1,048 while every AR query has length one. Both ranks observed the same topology.

Ordinary Ulysses cannot split a one-token decode query into two nonempty rank-local sequences. The flow head has no matching self-attention sequence to split. Prefill is the only viable boundary, but even deleting it entirely would save about 0.056% of the profiled time. Communication and complete per-layer KV reconstruction can only reduce that ceiling.

The 4.1-second startup prefill is excluded from the decision because it contains first-request initialization and falls to 0.05 seconds after warmup. Longer text and image-conditioned prefixes were not measured, so this result does not claim that every possible NextStep workload has the same phase ratio. A future proposal must first show a representative workload where prefill is material, then pass the cache, mask, position, padding, and A→B→A correctness contract in `contract.md`.
