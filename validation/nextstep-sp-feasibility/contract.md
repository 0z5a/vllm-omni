# NextStep sequence-parallel feasibility contract

Runtime inspected at `698f716`; the full-checkpoint validation candidate is `9545f40`. This is a P0 design and profiling contract, not implemented SP support.

| Stage | Query and KV semantics | Possible sequence parallel boundary |
| --- | --- | --- |
| Text/condition prefill | Full tokenized prefix; global `cache_position` and RoPE positions; padding plus causal mask; fills every layer of `StaticCache` | Candidate boundary. Local query positions must retain global causal visibility. Gather complete per-layer KV before continuing decoding, or explicitly support a persistent distributed KV layout |
| Causal image-token loop | One query per active CFG branch on each iteration; KV grows by one; next image token depends on the previous generated token | Ordinary sequence splitting has no useful nonempty query partition for two ranks. Do not alter the autoregressive dependency or claim the replicated decode path is SP |
| Flow-matching head | Samples one image token from the current context; multiple denoising evaluations per token; CFG branches occupy batch, not sequence | No existing image-sequence axis to shard as ordinary DiT Ulysses |
| VAE | Runs after unpatchify and latent scale/shift | Separate C3 scope; does not demonstrate backbone SP |

The pipeline computes image-token count as `(height // down_factor) * (width // down_factor)`. With checkpoint `patch_size=2` and four VAE `ch_mult` stages (factor eight), 512×512 requires 1,024 sequential image tokens. Actual prefill length and stage fractions must be measured; no prefill speedup or meaningful E2E benefit is assumed from this count.

`StageProfileRunner` enables the repository's existing synchronized phase profiler only in a separate fresh TP2 process after the timed TP0 arm. It records `model.forward`, `model.image_head.sample` and `vae.decode`; the first backbone call is prefill and the subsequent calls are AR decode. A forward pre-hook records query shape, mask shape, current KV length and capacity at the first two calls and at 256-token intervals. These intrusive profiling latencies must not be included in the normal E2E speed table.

Before any runtime SP integration, require single-rank versus two-rank comparisons for short prefixes (including fewer tokens than ranks), non-divisible prefix lengths, padding, global positions, the first cached decode call, and A→B→A requests. Compare per-layer hidden states and complete KV, not only the final image. Empty query ranks must still participate in all collectives. Global head divisibility, group membership and cache ownership must be explicit. Profile evidence and a real distributed prefill proof are still pending.
