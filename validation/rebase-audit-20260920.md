# Submitted-change replay audit

Upstream was refreshed to `5a93ec1b4e990f42973954d63809f932e699426a` on 2026-09-20. `git merge-tree --write-tree` completed cleanly for every candidate. Final rebases remain after GPU validation so the tested source SHA is not changed mid-run.

| Scope | Candidate SHA | Files changed by both candidate and upstream | Replay check |
| --- | --- | ---: | --- |
| ERNIE VAE | `e4a200fb` | 0 | clean |
| FLUX.1 VAE | `9054999b` | 0 | clean |
| GLM-Image VAE | `635df169` | 0 | clean |
| Helios VAE | `75e089b7` | 0 | clean |
| HiDream VAE | `f6098af8` | 0 | clean |
| Generic KL encode | `5413a40e` | 0 | clean |
| NextStep HSDP | `43c4e794` | 0 | clean |
| NextStep rectangular output | `8e43ef6d` | 0 | clean |
| OmniGen2 reference encode | `f2c1684b` | 0 | clean |
| OmniGen2 decode | `15840a66` | 0 | clean |
| Ovis HSDP evidence fix | `97d2a756` | 0 | clean |
| Qwen Layered encode | `58dfc80f` | 0 | clean |
| Qwen LoRA loading | `3ea51522` | 1 | clean; distinct hunks in `pipeline_qwen_image.py` |
| SD3.5 HSDP | `892df593` | 0 | clean |
| SenseNova HSDP | `e94dafe1` | 0 | clean |
| SenseNova RoPE | `fb3d6d54` | 0 | clean |
| SenseNova VAE scope | `29fa50b2` | 0 | clean |
| Step lifecycle | `bbd391d1` | 1 | clean; distinct hunks in `diffusion_worker.py` |

The Qwen upstream change controls CPU placement for the text encoder and VAE; the candidate adds the LoRA mixin to the pipeline class. The worker upstream change narrows custom-pipeline initialization; the candidate adds explicit step-request cleanup. Neither pair modifies the same hunk or contract.

All candidate commits retain author and committer `0z5a <dezhen.lu@student.uni-tuebingen.de>`. Each production commit includes the required sign-off.
