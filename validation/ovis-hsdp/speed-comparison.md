| Resolution | Baseline latency (s) | HSDP latency (s) | Speedup | Speed change | Max pixel difference |
|---|---:|---:|---:|---:|---:|
| 512×512 | 4.486 [4.450–4.643] | 32.789 [32.707–36.135] | 0.137× | -86.32% | 0 / 255 |
| 1024×768 | 12.110 [12.045–12.953] | 40.516 [40.294–65.764] | 0.299× | -70.11% | 0 / 255 |

Latency is the geometric mean of two fresh-process medians; brackets contain the range of all 10 measured requests per configuration/resolution. Speedup = baseline latency / HSDP latency. Negative speed change means slower. All 56 images (including warmups) are pixel-identical to their same-resolution baseline.
