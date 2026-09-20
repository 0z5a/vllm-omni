# Validation decision rule

Baseline `af59b66`; candidate causal temporal chunks and spatial VAE partitioning.
Full 3B FP16, five 848×480 frames, two L20s, window-SP degree 2 in both arms.
Only the candidate uses VAE patch degree 2 and tiling. The host is shared.

Correctness precedes screening: raw MAE≤0.02, max≤0.25, PSNR≥30 dB; exact
same-variant repeats; correct frame count, FPS, PTS and audio. Screening and A/A
data are separate from formal results. A/A drift limit: 15%; observed: 1.35%.

Five fresh APPA/PAAP quartets, five warmups and 24 measured requests per service.
The timer spans upload through MP4 download. Use each service median, geometric
means of the two same-arm service medians per quartet, then geometric aggregate
across the five independent quartets. Minimum effect: 1%; every quartet must
improve. No sample trimming and no exclusions from the completed formal run.

The descriptive whole-quartet bootstrap enumerates all 3,125 ordered resamples
of five quartets with replacement. Its 95% interval is supplementary; five units
on a shared host do not establish a tail SLA. Python/kernel source fingerprints,
worker warmup counts, media hashes and normal service exits are audited.
