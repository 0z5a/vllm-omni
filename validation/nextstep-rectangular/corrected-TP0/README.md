# Corrected NextStep TP2 first arm

Source `43c4e79`; full checkpoint, TP2, seed 142. This is only the first arm; HSDP comparisons remain pending. Two warmups and five timed requests per case.

| Case | Timed median (s) | Requests | Unique PNG hashes |
|---|---:|---:|---:|
| 512² initial | 97.2483 | 7 | 1 |
| 768×512 | 145.8993 | 7 | 1 |
| 512² reentry | 95.4569 | 7 | 1 |

Square reentry matches the initial square hash. All seven rectangular requests succeeded with identical output hashes. All 21 original PNGs are included and independently verified against remote SHA-256 manifests and worker result records, with RGB decoding, exact dimensions and iteration association checked. The archive SHA-256 is `d88ceb9b49f8b99582e8142e67762fd5e41d3e555e7da01fc9e915583ed71bbd`; `verify_images.py` reproduces the checks. No speedup is inferred from this single arm.

With two GPUs and one image per request, median GPU-seconds/image are 194.4966, 291.7986 and 190.9138 respectively. These are allocated GPU costs, not measured utilization-weighted compute. HSDP arms and the final TP2 arm are still pending, so no comparative acceleration is claimed.
