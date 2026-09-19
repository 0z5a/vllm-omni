| Reference images | Serial encode E2E (s) | Parallel encode E2E (s) | Speedup | Speed change | Output |
|---|---:|---:|---:|---:|---|
| 1 (first) | 11.6420 | 11.6161 | 1.0022× | +0.22% | Exact |
| 2 | 17.8292 | 17.7910 | 1.0021× | +0.21% | Exact |
| 3 | 24.3400 | 24.3170 | 1.0009× | +0.09% | Exact |
| 1 (reentry) | 11.6210 | 11.5890 | 1.0028× | +0.28% | Exact |

All 112 PNG hashes checked; outputs are identical across arms and repeats for each reference set, including A→B→C→A reentry. Both arms use the same two-rank VAE decode and Ulysses topology. Only reference encoding changes; source f2c1684 is compared against its dependency 15840a6.

One balanced A/P/P/A quartet, two warmups and five timed requests per case and fresh process. Reported latency is the geometric mean of process medians. Worker-global posterior RNG is reset to 142 on both sides for controlled replay. Shared-host results; no confidence interval or isolated-device claim. Small differences do not establish a speed benefit.
