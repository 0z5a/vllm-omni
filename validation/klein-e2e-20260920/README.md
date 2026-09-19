# FLUX.2-klein-4B VAE E2E

The existing PR 7464 at `159202d6` is applied to `698f716`. Its implementation remains attributed to its original author; this evidence is prepared by 0z5a. Checkpoint revision: `e7b7dc27f91deacad38e78976d1f2b499d76a294`.

Both arms use two L20 GPUs (2/3), identical strict Ulysses-2, BF16, eager execution and native VAE tiling. The candidate additionally enables VAE degree 2. Four fresh processes in A/P/P/A order run 512² batch 1, 1152×1024 batch 1, 768×512 batch 2, and 512² reentry. Each case has two warmups and five timed repeats, four distilled steps, guidance 1 and seed 142.

[Speed and quality table](speed-comparison.md) separates full E2E from VAE-only timing. All 140 PNG files are retained with raw measurements. Repeated outputs within each arm, between matching arms and after reentry are exact. Baseline-versus-candidate large tiled outputs are exact; smaller halo-patch outputs differ as reported. This is not a held-out quality acceptance study.

The component gate fixes real checkpoint latents and separately invokes full-frame, native tiled, distributed tiled, degree-1 tiled within WORLD=2, and small-image halo patch. It covers square/rectangular grids, ragged boundary tiles, fewer tiles than ranks, idle-rank collectives, batch 2 and reentry. Native and distributed tiled equality is asserted at the output-owning rank; output shape/dtype/finite/repeat stability is asserted on both ranks. Raw per-rank samples, memory peaks, tile assignments, inputs and quality errors are retained.

The component benchmark holds both native and candidate modules resident; its absolute memory peaks are not full-pipeline residency measurements. Baseline component calls execute only on rank 0, whereas distributed calls participate on both ranks, with slowest-rank wall time reported. E2E telemetry is device-wide on a shared host. No isolated-device confidence interval is claimed. The archive deduplicates identical files as tar hardlinks while retaining every logical path.
