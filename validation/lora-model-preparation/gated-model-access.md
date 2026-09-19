# Remaining full-checkpoint access (2026-09-20)

Fresh remote unauthenticated requests to `hf-mirror.com/{repository}/resolve/main/model_index.json` returned HTTP 403 for FLUX.1-dev, FLUX.1-schnell, FLUX.1-Kontext-dev and Stable-Diffusion-3.5-medium. Each response explicitly stated that the model is restricted and the requester is not authorized.

On the local host, the official Hugging Face FLUX.1-schnell file endpoint returned HTTP 401 requiring authentication and model access, while its public model metadata endpoint returned HTTP 200. Metadata visibility does not grant weight access. No configured Hugging Face token was present on either host; no token contents were printed or transmitted.

The user was asked to supply an existing authorized weight directory or configure an account credential with model access on the machine, without sending tokens in chat. Other experiments continue. These access-dependent scopes remain incomplete, and endpoint failures are not generalized to a claim that no authorized copy exists anywhere.
