# Helios VAE integration audit

Pinned source: `698f716`. Existing checkpoint revision: `b991c0379a018f4de3227d95468237f56066f5bb`; the cached files and shard indexes were verified separately. No distributed Helios VAE or full E2E success is claimed here.

The pipeline loads Diffusers `AutoencoderKLWan` in FP32 and has real image and video encode paths. Image conditioning encodes the image and a fake video separately; video conditioning encodes its first frame and subsequent chunks. Each posterior is sampled with the request generator. Any encode integration must retain their order, frame counts, scale/shift and RNG consumption.

Both monolithic generation and step execution decode each chunk and concatenate decoded videos along time. The existing distributed Wan tiled decoder returns a full tensor only on rank zero and a dummy tensor elsewhere. A class-name substitution alone would therefore break the all-rank history concatenation contract. A pipeline-local adapter must preserve complete decoded results on participating ranks, or prove a correctly scoped output-ownership change in both callers. It must not change temporal partitioning or streaming policy.

The shared wrapper's distributed encode path already broadcasts posterior moments and clears temporal caches before and after tiled encoding. This is useful implementation evidence, not proof that real Helios input preprocessing, posterior sampling and chunk reentry are equivalent. J2-E and J2-D remain separate pending validations under V-E3D and V-D.

Ownership searches on 2026-09-20 found no result for `Helios distributed VAE`. The broader search found open PR 6010, which concerns compiling and prewarming VAE decode; it must be checked for interaction before a new implementation is submitted. The issue had no comments newer than 2026-09-19 in the fresh paginated API check. Search results alone do not establish correctness or completion.

## Implemented and validated component path

Commit `75e089b` (published) adds a small Helios-local adapter. It clears temporal caches on every rank and synchronizes the complete tiled decode result before either Helios caller appends it to video history. The native FP32 loader, encode ordering, scale/shift, and temporal partitioning are retained. PR 6010's compile/prewarm work touches different shared configuration/worker files; compilation is disabled for these initial comparisons.

Six CPU regressions pass. A real-weight two-process Gloo run first caught stale decoder cache on the nonzero rank; the failure log is retained. After cleanup, all eight rank/case records pass for 1→5→9→1 frames at 48×64. The subsequent actual two-L20 CUDA test passes all ten rank/case records at 320×448, including batch 2 and reentry. Both tests compare exact posterior parameters, generator-controlled samples, unchanged global RNG and exact native/distributed tiled decode. CUDA uses FP32, matching Helios, with matched 256-pixel tiles and 128-pixel stride.

These component results do not establish full-pipeline fit, VAE speed improvement, quality relative to untiled decoding, or full E2E. Those gates remain pending; shared weights are retained.

Draft implementation: [PR 7854](https://github.com/vllm-project/vllm-omni/pull/7854). Full E2E remains pending.

## Full E2E queue

The full benchmark and probe import successfully against an immutable source snapshot whose three changed-file hashes match `75e089b`. GPU runs are queued after the current model controller and ERNIE component gate. The shared checkpoint is already verified and retained.

Four timed A/P/P/A processes use identical TP2, BF16 transformer/text encoder, the pipeline's FP32 VAE, eager execution and native VAE tiling. They run 33-frame T2V, 33-frame I2V, 66-frame V2V, then T2V reentry at 640×384. Each case has two warmups and five timed requests. Both arms use the distilled pyramid path with 10/10/10 stages, guidance 1 and seed 142. Generated arrays are saved by content hash; selected frames provide inspection samples.

Separate untimed A/P probes record the ordered VAE inputs/outputs and actual tile-call counts. Two additional untimed runs cover step execution. These diagnostics are excluded from the speed samples. Both input types use the pipeline's existing tensor APIs; no new public image/video contract is introduced. Full execution and fit remain pending.
