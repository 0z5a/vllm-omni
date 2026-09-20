# NextStep H0 partial progress

Source `43c4e79`, GPUs 2/3, 512×512. Only 2 of 5 required timed samples in the first case are complete; this is not a final arm or quartet result.

| Iteration | Warmup | Seconds | GPU-seconds |
|---:|---|---:|---:|
| 0 | True | 2472.927457 | 4945.854915 |
| 1 | True | 2472.762553 | 4945.525106 |
| 2 | False | 2530.607596 | 5061.215192 |
| 3 | False | 2949.158997 | 5898.317994 |

All four remote PNGs have SHA-256 `935587c8b845c41262bc99adf1b437e5a75964c08786d7df7404a9492bb618f1`, matching the canonical image retained in `../first-warmup`. H0 still differs from TP0 (prior PSNR 13.664185 dB); repeat identity does not establish cross-mode quality acceptance. No final median or speedup is reported.
