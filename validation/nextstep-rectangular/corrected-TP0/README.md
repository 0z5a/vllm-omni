# Corrected NextStep TP2 first arm

Source `43c4e79`; full checkpoint, TP2, seed 142. This is only the first arm; HSDP comparisons remain pending. Two warmups and five timed requests per case.

| Case | Timed median (s) | Requests | Unique PNG hashes |
|---|---:|---:|---:|
| 512² initial | 97.2483 | 7 | 1 |
| 768×512 | 145.8993 | 7 | 1 |
| 512² reentry | 95.4569 | 7 | 1 |

Square reentry matches the initial square hash. All seven rectangular requests succeeded with identical output hashes. Hashes here are worker result records; archive-level PNG verification is still required before the complete quartet report. No speedup is inferred from this single arm.
