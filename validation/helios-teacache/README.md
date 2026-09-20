# Helios TeaCache validation

Prepared for source `8aab7339feaaaaac2ee88562f984ab407b6734c6` and checkpoint
`BestWishYsh/Helios-Distilled` revision `b991c0379a018f4de3227d95468237f56066f5bb`.

The matrix uses the standard complete-request path with TP=2. It keeps the
checkpoint, prompt, seed, 384×640 output, 33 frames, BF16, and three two-step
pyramid stages fixed. Baseline and TeaCache arms run in separate processes. The
timed matrix uses two warmups and five retained iterations in forward and reverse
order. Untimed probes cover 33→66→33 frames, stage transitions, repeated requests,
full-compute fallback, real block skips, and per-rank decisions. `verify.py`
requires exact baseline/no-hit outputs and emits the requested Markdown
quality/speed table.

This directory records a prepared validation. No E2E result is claimed until the
matrix finishes and its raw records pass `verify.py`.
