# Additional scope audit

Author: 0z5a. Baseline: `698f7160125d3071b8c0eef69b1b03fa8dfba766`.

| Scope | Actual runtime path | Disposition |
|---|---|---|
| G2 SenseNova-U1/U1.5 VAE parallelism | Registered `SenseNovaU1Pipeline` declares `_vae_modules = []` (pipeline line 491); its FM head predicts pixels, and `_run_denoising_loop` unpatchifies them directly (line 1509). Input images use vision preprocessing, not a VAE. | Not applicable to this pipeline. Documentation correction [PR #7845](https://github.com/vllm-project/vllm-omni/pull/7845), commit `29fa50b`; all changed-file checks pass. No VAE E2E or speedup claim. |
| A6-E GLM image encode | Diffusion-stage `pipeline_glm_image.py:653` calls `self.vae.encode(...).latent_dist` through `retrieve_latents(..., sample_mode="argmax")`. | Real encode path exists for conditioning images; cannot mark not applicable based on the AR visual tokenizer. Implementation and validation remain pending. |
| A6-D GLM image decode | Same diffusion pipeline calls `self.vae.decode` at line 901. Default deploy places AR on device 0 and diffusion on device 1. | Multi-rank VAE must use the diffusion worker WORLD, not combine independent AR and DiT stage devices. Complete multistage validation remains pending. |

G1 SenseNova HSDP remains independent and incomplete. G2's applicability finding does not certify any HSDP, cache, or other feature combination.
