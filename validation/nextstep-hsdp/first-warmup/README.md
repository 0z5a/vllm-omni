# NextStep first HSDP warmup: provisional evidence

Source `43c4e79`, BF16 eager, seed 142, 512×512, 20 FM steps, guidance 4.0. The TP2 reference is the first warmup in `../../nextstep-rectangular/corrected-TP0/`; HSDP2 uses the same request and source on GPUs 2,3.

| Sample | Request time (s) | Max RGB error versus TP2 | Mean absolute RGB error | PSNR (dB) |
|---|---:|---:|---:|---:|
| H0 first warmup | 2472.927 | 245 | 38.0378 | 13.6642 |

Both images contain the requested teapot and cups, but their arrangement differs visibly. Pixel equivalence is not established. These are different parallel strategies (TP2 versus HSDP2); this observation alone does not identify the source of numerical divergence or establish a quality regression. Repeated output stability and the completed quartet remain pending.

This sample is excluded from timed medians. No speed improvement or acceptance claim is made from one warmup. Initialization took 2109.094 seconds. Original PNG and full hashes are retained alongside the metrics.
