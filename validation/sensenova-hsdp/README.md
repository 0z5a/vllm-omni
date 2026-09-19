# SenseNova HSDP candidate

Author: 0z5a. Baseline `698f716`; candidate `7db1da7`.

The language-model root now selects actual `SenseNovaU1DecoderLayer` instances for HSDP. Embedding-only, AR logits and denoising already enter the root through module calls. Two pipeline entrypoints previously forced inference mode, overriding the runner's HSDP `no_grad` context. They now disable gradients while retaining the caller's inference-mode setting. Native execution therefore retains the runner's inference mode, while HSDP can retain tensor version counters.

| Check | Baseline | Candidate |
|---|---|---|
| Pipeline/text entrypoints preserve HSDP execution mode | 2 failed | 2 passed |
| Pipeline/text entrypoints preserve native inference mode | 2 passed | 2 passed |
| Changed-file pre-commit, including mypy | Not rerun | Passed |
| Real two-rank embedding, AR-cache and denoising parity | Pending | Pending |
| Full U1/U1.5 checkpoint E2E and speed | Pending | Pending |

The CPU constructor probe stopped before model construction: the CUDA platform initializer attempted rank modulo zero after GPUs were hidden. Its failure log is retained; it is not a successful model-construction test or evidence of a model regression. The prepared CUDA probe covers embedding-first root entry, three cached AR decode steps, unchanged prefix KV during denoising, varying batch/sequence lengths and resharding between calls. It has not run yet.

Existing B3 prefix-identity and B5 TP work is separate. The recorded SenseNova HSDP PR search returned no matches, which is not a claim of maintainer-assigned ownership. No PR or full-model support claim follows from these CPU tests alone.
