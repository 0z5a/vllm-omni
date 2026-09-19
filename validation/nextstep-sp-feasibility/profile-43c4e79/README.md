# NextStep intrusive phase profile

Source `43c4e79`, TP2, full checkpoint, 512², seed 142. Startup, one warmup and one diagnostic request. These synchronized/logged phase timings are not E2E benchmark samples.

| Rank | Request | Prefill (s) | AR decode (s) | Flow head (s) | VAE (s) | Prefill share | Perfect-prefill upper bound |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | startup | 4.125039 | 46.185333 | 5.031784 | 0.188245 | 7.4284% | 1.080245× |
| 0 | warmup | 0.124518 | 45.092198 | 51.045214 | 0.046561 | 0.1293% | 1.001295× |
| 0 | diagnostic | 0.053191 | 44.759123 | 50.765098 | 0.046581 | 0.0556% | 1.000557× |
| 1 | startup | 4.125449 | 46.189791 | 5.032569 | 0.188069 | 7.4284% | 1.080245× |
| 1 | warmup | 0.124509 | 44.739956 | 51.403157 | 0.046582 | 0.1293% | 1.001294× |
| 1 | diagnostic | 0.053477 | 44.024192 | 51.498393 | 0.046604 | 0.0559% | 1.000560× |

The bound assumes zero prefill time and no added communication, using only the profiled phases. It is not an observed speedup. Startup uses nine prefill tokens; the measured prompt uses 24 with CFG batch two. Decode queries have length one while the measured prompt cache grows to 1,048 positions. Longer text and image-conditioned prefill remain unmeasured, so this result does not establish general SP feasibility. No SP implementation or distributed prefill correctness is claimed.
