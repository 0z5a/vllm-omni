# NextStep H0 first timed request — incomplete arm

Source `43c4e79`, HSDP2 on GPUs 2,3, BF16 eager, seed 142, 512×512, 20 FM steps, guidance 4.0.

| Request | Classification | E2E seconds | GPU-seconds/output |
|---|---|---:|---:|
| 0 | Warmup, excluded from timing statistics | 2472.9275 | — |
| 1 | Warmup, excluded from timing statistics | 2472.7626 | — |
| 2 | First timed sample | 2530.6076 | 5061.2152 |

Only one timed sample is complete; the other H0 requests, H1, and TP1 remain pending. This is not a completed median or final speed comparison. Initialization was 2109.0936 seconds and is reported separately.

The three remote PNGs were hashed directly and all match the original retained at `../first-warmup/H0-512x512-0-0.png` (one canonical copy of identical bytes). The earlier TP2-versus-HSDP image difference remains: max RGB error 245, mean absolute error 38.0378, PSNR 13.6642 dB. Repeat stability within H0 does not establish cross-strategy equivalence or quality acceptance.
