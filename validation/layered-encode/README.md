# Qwen-Image-Layered encode validation

Author: 0z5a. Baseline `698f716`; candidate `58dfc80`.

The wrapper inherits Layered's vendored VAE and distributes the existing spatial encode tiles. It retains the original one-frame prefix/four-frame temporal chunks, per-tile feature-cache reset, posterior moments, blending and crop order. Decode and normalization implementations are inherited unchanged. This does not substitute the current Diffusers VAE or duplicate the Edit/Edit-2509 pipeline changes in PR #4933.

| Validation | Result |
|---|---|
| Native pipeline loader selection | Passed; vendored VAE subclass retained |
| CPU posterior/sample/RNG parity, serial and parallel | 6 cases passed |
| Batch 1/2, frames 1/5, rectangular A→B→A requests | Exactly equal |
| Reversed tile completion order and cache cleanup | Passed |
| Two-process Gloo executor, metadata packing/gather/merge/broadcast | 6 rank records exactly equal; RNG unchanged |
| Changed-file pre-commit checks, including mypy | Passed |

The Gloo probe uses real collectives and the real distributed executor with a test WORLD-group adapter. It uses a tiny randomly initialized vendored VAE on CPU, not real weights or GPU execution. Full checkpoint revision `8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf` has downloaded: all 33 repository files are present, and 10 safetensors files total 57,699,272,568 bytes. Every shard was opened and index keys were checked; see `layered-verified.json`. This is structural verification, not a full-file checksum or GPU inference test.

| Full-model metric | Native encode | Parallel encode | Speedup |
|---|---|---|---|
| VAE encode latency | Pending | Pending | Pending |
| Raw-image layered generation latency | Pending | Pending | Pending |

Real-weight two-rank CUDA parity, complete layered output correctness and E2E speed remain pending. The current evidence does not establish these claims.

The fresh-process E2E driver is prepared and deployed, with the same TP2, BF16, eager execution, native tiled/sliced decode and 50 denoising steps in both variants. It covers two/four RGBA layers, square/rectangular inputs, 640/1024 resolution buckets and a repeated first request; all layer images and raw input hashes are saved. Four arms run A/P/P/A, each with two warmups and five measurements per case. The CUDA component check precedes this queue. Syntax, Ruff and target-environment import checks passed; the GPU queue has not started. Candidate source is an immutable `698f716` archive plus the complete three-file `58dfc80` delta.
