# FLUX.2-klein-4B VAE validation preparation

The existing [PR 7464](https://github.com/vllm-project/vllm-omni/pull/7464), head `159202d6cff497b1c9e82e4d08a116dcee2f20c1`, is applied to baseline `698f716` for validation. Implementation ownership remains with its original author. Its loader test passes on CPU. No full GPU E2E result is claimed yet.

The prepared driver compares identical Ulysses-2 topologies with native versus distributed VAE decode. It uses BF16, eager execution, native tiling, seed 142, four distilled steps and guidance 1. Cases cover 512², 1152×1024, a two-output 768×512 batch, and 512² reentry, with two warmups and five timed requests each. A/P/P/A fresh processes and the real VAE component gate remain to be run after the current queue.

Model revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` is downloaded and its selected files and shard index verified. The prepared harness still needs GPU execution; no speed or output parity conclusion follows from the CPU loader test.

The prepared real-weight `vae_contract.py` now separates full-frame, explicitly forced native tiles, distributed tiles at degrees 2 and 1 within WORLD=2, and small-image halo patches. It records two warmups/five measurements, allocated/reserved peaks, local tile counts, saved latent tensors and errors against full-frame. Exact native/distributed tiled equality and request reentry are hard gates; halo error is reported without an invented acceptance threshold. CPU import and topology construction pass; GPU execution remains pending.

The subsequent controller waits for the current main controller to exit successfully through its K1 completion marker, then runs Klein's component gate and full quartet followed by the ERNIE component contract. It never overlaps those GPU jobs with the active NextStep/SenseNova/Layered/Edit/K1 sequence.

## Completed quartet

The component gate and all four fresh E2E arms now pass. All 140 PNG hashes were verified remotely. At 1152×1024, native/distributed tiled outputs are exact and the observed E2E speedup is 1.0718× (+7.18%); the corresponding large-latent VAE-only comparison is 1.456×. Small-image halo output is approximate: at 512², max error 25/255 and PSNR 35.88 dB; the initial +1.70% becomes +0.39% on reentry, so that small difference is not a robust gain. Batch-2 768×512 has max error 20/255 and PSNR 36.31 dB with +2.16% observed speed.

See `../klein-e2e-20260920/speed-comparison.md` and raw artifacts. Measurements are a single shared-host balanced quartet, not an isolated repeat-session confidence interval. Quality acceptance for halo approximation and a final requirement-by-requirement audit remain pending.
