# Helios VAE integration audit

Pinned source: `698f716`. Existing checkpoint revision: `b991c0379a018f4de3227d95468237f56066f5bb`; the cached files and shard indexes were verified separately. No distributed Helios VAE or full E2E success is claimed here.

The pipeline loads Diffusers `AutoencoderKLWan` in FP32 and has real image and video encode paths. Image conditioning encodes the image and a fake video separately; video conditioning encodes its first frame and subsequent chunks. Each posterior is sampled with the request generator. Any encode integration must retain their order, frame counts, scale/shift and RNG consumption.

Both monolithic generation and step execution decode each chunk and concatenate decoded videos along time. The existing distributed Wan tiled decoder returns a full tensor only on rank zero and a dummy tensor elsewhere. A class-name substitution alone would therefore break the all-rank history concatenation contract. A pipeline-local adapter must preserve complete decoded results on participating ranks, or prove a correctly scoped output-ownership change in both callers. It must not change temporal partitioning or streaming policy.

The shared wrapper's distributed encode path already broadcasts posterior moments and clears temporal caches before and after tiled encoding. This is useful implementation evidence, not proof that real Helios input preprocessing, posterior sampling and chunk reentry are equivalent. J2-E and J2-D remain separate pending validations under V-E3D and V-D.

Ownership searches on 2026-09-20 found no result for `Helios distributed VAE`. The broader search found open PR 6010, which concerns compiling and prewarming VAE decode; it must be checked for interaction before a new implementation is submitted. The issue had no comments newer than 2026-09-19 in the fresh paginated API check. Search results alone do not establish correctness or completion.
