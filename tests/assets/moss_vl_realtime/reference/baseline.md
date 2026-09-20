# MOSS-VL-Realtime baseline

## Reference capture

| case | prompt tokens | generated | wall (s) | decode (tok/s) | backend | device |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 1.984 | 4.54 | sdpa | None |
| image | 13 | 37 | 1.901 | 18.93 | sdpa | None |
| short_video | 15 | 64 | 3.22 | 19.57 | sdpa | None |

## Native replay

| case | prompt | generated | prefill (s) | decode (s) | decode (tok/s) | wall (s) | reference wall (s) | speedup | tokens | logits max Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 0.062 | 0.375 | 23.98 | 0.438 | 1.984 | 4.53 | 10/10 | 0.1562 |
| image | 13 | 37 | 0.058 | 1.429 | 25.18 | 1.487 | 1.901 | 1.28 | 37/37 | 0.3438 |
| short_video | 15 | 64 | 0.062 | 2.586 | 24.36 | 2.648 | 3.22 | 1.22 | 32/64 | 0.584 |

## Paced realtime

| side | frames accepted | dropped | output chunks | first output (s) | session (s) | output |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 4 | 0 | 19 | 1.301 | 17.415 | <|silence|><|round_start|><|silence|><|response|><|response|>The color is red.<| |
| native | 4 | 0 | 1 | 2.117 | 2.233 | The color is red. |

## Loading

- `prompt_only`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `image`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `short_video`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
