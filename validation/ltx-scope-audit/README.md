# LTX scope audit

Audit date: 2026-09-20

## I1: LTX-2 family VAE encode

The execution stop condition is met by upstream PR
[#4831](https://github.com/vllm-project/vllm-omni/pull/4831). Its focused delta adds
distributed spatial encode to `DistributedAutoencoderKLLTX2Video`, preserves the
shared video VAE behavior, and does not change the audio path. The current PR is
open, non-draft, mergeable at `21cf0c9f6b283fa4880e46dadb41401430b578e2`,
with build, pre-commit, and DCO checks passing.

The PR includes component-level exactness tests and a full two-GPU I2V E2E. Its
reported speed comparison is:

| Model | Resolution | VAE patch parallel | E2E latency | Speedup |
| --- | ---: | ---: | ---: | ---: |
| LTX-2.5 | 576x576 | off | 6.30 s | 1.000x |
| LTX-2.5 | 576x576 | encode + decode = 2 | 6.29 s | 1.002x |
| LTX-2 | 1024x576 | off | 6.31 s | 1.000x |
| LTX-2 | 1024x576 | encode + decode = 2 | 5.44 s | 1.160x |

The LTX-2.5 result is neutral at the tested resolution. The LTX-2 result is about
14% lower latency. These are the PR author's 2xH100 measurements; this audit does
not relabel them as local L20 measurements.

Disposition: `EXISTING_PR_VALIDATED`. Do not create a duplicate implementation.

## I2: LTX TeaCache

Issue [#1217](https://github.com/vllm-project/vllm-omni/issues/1217#issuecomment-4648074990)
records `Sendoh-code` claiming TeaCache for LTX-2.3. The live matrix also names
that owner. Searches for `LTX TeaCache`, `LTX-2.3`, and `LTX2.3` found no TeaCache
PR, so there is no separate unowned implementation to start.

Disposition: `OWNER_COVERED_NO_DUPLICATE`. The document's owner gate and stop
condition are satisfied; any future implementation work stays with the existing
owner scope.
