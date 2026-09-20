# MOSS-VL-Realtime baseline

## Reference capture

| case | prompt tokens | generated | wall (s) | decode (tok/s) | backend | device |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 2.37 | 3.8 | sdpa | None |
| image | 13 | 37 | 2.286 | 15.75 | sdpa | None |
| short_video | 15 | 64 | 3.668 | 17.18 | sdpa | None |

## Native replay

| case | prompt | generated | prefill (s) | decode (s) | decode (tok/s) | wall (s) | reference wall (s) | speedup | tokens | logits max Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 0.067 | 0.41 | 21.96 | 0.477 | 2.37 | 4.97 | 10/10 | 0.1562 |
| image | 13 | 37 | 0.067 | 1.636 | 22.01 | 1.703 | 2.286 | 1.34 | 37/37 | 0.3438 |
| short_video | 15 | 64 | 0.071 | 2.96 | 21.29 | 3.031 | 3.668 | 1.21 | 32/64 | 0.584 |

## Paced realtime

| side | frames accepted | dropped | output chunks | first output (s) | session (s) | output |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 4 | 0 | 15 | 1.579 | 10.489 | <|silence|><|round_start|><|silence|><|response|><|response|>The color is altern |
| native | 4 | 0 | 1 | 2.45 | 2.598 | The color is red. |

## Loading

- `prompt_only`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `image`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `short_video`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
