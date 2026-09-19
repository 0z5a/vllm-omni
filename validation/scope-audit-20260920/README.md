# Remaining VAE scope and ownership audit

Author: 0z5a. Source baseline: `698f716`. GitHub issue body and all 93 comments were read on 2026-09-20; relevant comments and immutable PR heads are recorded alongside this note. Search results alone are not ownership clearance or validation evidence.

| Scope | Existing work | Implementation boundary | Remaining validation |
|---|---|---|---|
| A6 GLM-Image | [PR #3549](https://github.com/vllm-project/vllm-omni/pull/3549), head `1eeb8fb01f942352bf1f9631652188cbf4ba7e46`, separates AR, DiT and VAE into three stages | Stage placement differs from parallel VAE computation. Keep the default two-stage path and its image-conditioning encode semantics; do not duplicate the split pipeline | Real raw-prompt and raw-image E2E, stage-group membership, encode/decode parity and timing |
| F1/F2 Qwen editing | [PR #4933](https://github.com/vllm-project/vllm-omni/pull/4933), head `37e4407dcf10d49ab9215dfd4a545fb9f014b5c4`, implements distributed tiled encoding and loader wiring | Validate the existing author's implementation instead of duplicating it | Real-weight multi-GPU editing E2E; the PR's CPU results are not our verification |
| F3 Qwen Layered | #4933 explicitly excludes Layered | Layered imports the repository's vendored VAE. Preserve its normalization and temporal cache semantics; substituting the current Diffusers VAE would change the numerical baseline | Separate adapter/design and native-vendored parity, followed by layered E2E |
| I1 LTX encode | [PR #4831](https://github.com/vllm-project/vllm-omni/pull/4831), head `21cf0c9f6b283fa4880e46dadb41401430b578e2`, adds distributed tiled encoding | Retain known-metadata and fallback paths; audit the two-stage pipeline's nested VAE discovery separately | Real-checkpoint I2V and upsampler E2E across supported LTX variants |
| I2 LTX TeaCache | [Existing LTX-2.3 claim by Sendoh-code](https://github.com/vllm-project/vllm-omni/issues/1217#issuecomment-4648074990) | Avoid duplicate implementation while ownership remains claimed | Check existing implementation and coordinate scope before publishing overlapping code |

GLM's pinned default deployment places AR on device 0 and diffusion on device 1. VAE parallel size 2 requires two ranks inside the diffusion stage; the AR rank is not automatically a member of that group. A two-card experiment must explicitly account for stage placement and shared-device memory, or obtain a separate AR device. No two-card end-to-end topology is claimed validated here.

NextStep's vendored VAE has tiling-related attributes and blend helpers, but `decode` directly invokes the whole decoder. Global GroupNorm and spatial attention mean naive patch decoding changes the computation. C3 needs explicit numerical acceptance and restricted support before advertising patch parallelism. This is an audit finding, not completed C3 support.
