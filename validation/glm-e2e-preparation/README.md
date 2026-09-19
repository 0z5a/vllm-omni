# GLM-Image full E2E preparation

Baseline `698f716`; candidate `635df16`, including shared KL encode dependency `1885628`. Checkpoint `2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4` was downloaded and its shards/index verified. Five changed-file hashes match the immutable deployed source snapshot.

Both arms use the same two-stage placement on authorized physical GPUs 2/3: AR on logical device 0 with a 0.55 memory-utilization target; diffusion TP2 on logical devices 0/1. Stage 0 and the first diffusion rank therefore share a GPU. This is a planned allocation, not a fit guarantee. Both stages use BF16 and eager execution with prefix caching disabled. Only VAE degree changes from 1 to 2.

Deploy parsing passes for both ordinary and diagnostic configurations, including the stage-local diagnostic runner. The full benchmark and probe import successfully on CPU. Actual model loading, memory peaks, AR output equivalence and E2E remain pending.

The prepared A/P/P/A quartet runs 512² T2I, 768×512 edit, 1152×1024 T2I and 512² reentry, two warmups/five timed repeats, 50 diffusion steps, guidance 1.5 and seed 142. AR sampling defaults retain their stop token and 4,353-token ceiling; target dimensions are supplied through both processor metadata and AR extra arguments. The edit case uses the existing PIL-image API.

Separate untimed probes hash prior tokens and prompt embeddings presented to the transformer, VAE inputs/posteriors/decoded outputs, and actual tile execution counts. Degree-two KL decode returns complete output on rank zero and a dummy on other ranks, which must remain distinguished in diagnostic comparisons. Source images, raw timings and output PNG hashes are retained. The sequence runs after Helios when the current model controller finishes; no concurrent use of GPUs 2/3 is introduced.
