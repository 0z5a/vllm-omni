# Authenticated model prefetch

All four pinned Hugging Face repositories needed by A2, A4, A5, and H1 returned authenticated access from the official endpoint on 2026-09-20. The prior mirror-specific HTTP 403 result no longer blocks these scopes.

The credential is passed through hidden terminal input and retained only in downloader process memory. It is not written to a token file, login configuration, script, command line, or log. Downloads run one model at a time through the local proxy. Each selected Diffusers file is checked against the repository metadata and LFS SHA-256, transferred to the remote model directory, checked again on the server, then removed from the local staging directory. Full download verification and GPU E2E remain pending.

| Scope | Repository | Pinned revision | Access | Download state |
| --- | --- | --- | --- | --- |
| A2 | `black-forest-labs/FLUX.1-schnell` | `741f7c3ce8b383c54771c7003378a50191e9efe9` | Granted | In progress |
| A2 | `black-forest-labs/FLUX.1-dev` | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` | Granted | Queued |
| A4/A5 | `black-forest-labs/FLUX.1-Kontext-dev` | `24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d` | Granted | Queued |
| H1 | `stabilityai/stable-diffusion-3.5-medium` | `b940f670f0eda2d07fbb75229e779da1ad11eb80` | Granted | Queued |

The full-scope progress remains **5/33 E2E-complete (15.2%)**. Model access and harness preparation do not count as completed E2E work.
