# SeedVR2 native C0 model contract

Owner: 0z5a. Implementation: native serving worktree on parent7739 head9a62e04 plus RoPE4dc1bfb and the native integration in this change. Reference source identity is in `reference-provenance.md`; frozen production source hashes are in `frozen-source-hashes.json`.

## Data and numerical contract

The released3B FP16 DiT, FP16 causal VAE and fixed `pos_emb.pt` are required. Only the checkpoint suffix `.rope.rope.freqs` is normalized to `.rope.freqs`; duplicate/missing/unexpected/shape-mismatched keys fail loading. No truncated model is accepted.

Input is RGB TCHW floating values in[0,1], PIL frames, or one persisted CFR video. Explicit output dimensions are multiples of16. Bicubic antialiased resizing clamps to[0,1]; the final frame is repeated until temporal length is4n+1, then model input is normalized to[-1,1]. Decode output is cropped to the original frame count, including6→9→6.

Causal VAE downsamples time by4 and space by8. For5×128×224 input: encoded latent is1×16×2×16×28. Posterior sampling uses a request-local generator. THWC condition is multiplied by0.9152; noise must preserve its tensor strides. Contiguous `randn(shape)` is a proven wrong substitution even with the same seed. Concatenating noise16,condition16,mask1 produces896×33 rows. Patch1×2×2 yields224 video tokens, hidden width2560. The fixed text embedding has width5120. There is one NaDiT call at timestep1000, velocity prediction, Euler update `noise - velocity`, inverse0.9152 scaling, native VAE decode and RGB clamping. Reference Euler materializes FP32 subtraction, then casts to FP16 before VAE scaling; native subtraction rounds directly to this FP16 boundary. Both five/six-frame cases have identical conditioning, velocity, FP16 decode-boundary latent and unencoded frames. The raw FP32 vs FP16 intermediate is not claimed bitwise equal.

Each block normalizes/modulates video and text, applies joint attention and residuals, then normalized/modulated MLP and residuals. `mm_layers=10` controls shared parameters; it does not stop text updates. The32-layer alternating schedule has31 layout transitions. Shifted windows clip boundaries rather than cyclically roll. Runtime ownership, global text-window reduction and inverse token routing are covered by production-path tests. VAE encode/decode currently execute independently on each SP rank; no single-rank VAE or memory scaling claim is made.

## Request and media lifetime

Prepared input, posterior/noise generator, runtime plans and final media belong to one request. RoPE tables cache geometric angles, not request-dependent text state. Repeated seed A/B/A engine calls are exact for A; B differs. Full SP1 matches reference frames exactly; SP2/4 pass the existing error gates. A three-block CPU counterexample (mm_layers=1) changes subsequent text by max0.23017 for different videos. Reusing the prior request's text after block1 corrupts output by max0.0040093/MAE0.0008901; normal A/B/A remains exact. This is a dependency counterexample, not a full-checkpoint quality threshold.

CFR rate and relative PTS are preserved, with output origin normalized to0. VFR and user-requested FPS changes are rejected. Mono/stereo audio is decoded at its source sample rate, aligned against video start, trimmed/padded to video duration, and carried through typed media/SHM to mux. AAC encoding is lossy: the five-frame fixture's valid-sample correlation is0.9967, with decoded AAC padding outside the0.2-second container duration. Audio compressed packets are not byte-preserved.

C0 does not silently implement WF-07 P0: batch5/overlap1, LAB, swap32 and tiled VAE are separate execution/algorithm choices. The B-line107-frame reference is therefore a distinct comparison, not a numeric parity denominator for whole-clip C0.

## Review and release checks

A contributor change must identify its exact source and checkpoint, preserved model invariant, regression entry point, actual backend, full-checkpoint result, and complete E2E timing boundary. Use the source/CPU/GPU/full-service evidence levels in `review-audit-7739.md`; microbenchmarks cannot establish service gains.

Before expanding support: run short and boundary video parity, A/B/A content/shape requests, SP1/2/4 where declared, audio/FPS/PTS checks, client-error recovery, cancellation and worker-fault cleanup, unsupported-option rejection, per-rank memory/stage profiling, and fresh-service paired validation. Record unavailable configurations honestly. Ownership of shared API changes stays with their existing maintainers; coordinate P0 work with MinhaoLi0318 before publishing an overlapping implementation. Two subsequent issue tasks remain unclaimed until the Track A work is complete.
