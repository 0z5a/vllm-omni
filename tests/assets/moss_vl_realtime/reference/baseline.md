# MOSS-VL-Realtime baseline

## Reference capture

| case | prompt tokens | generated | wall (s) | decode (tok/s) | backend | device |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 2.338 | 3.85 | sdpa | None |
| image | 13 | 37 | 2.281 | 15.78 | sdpa | None |
| short_video | 15 | 64 | 3.669 | 17.17 | sdpa | None |

## Native replay

| case | prompt | generated | prefill (s) | decode (s) | decode (tok/s) | wall (s) | reference wall (s) | speedup | tokens | logits max Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 0.062 | 0.4 | 22.51 | 0.462 | 2.338 | 5.06 | 10/10 | 0.1562 |
| image | 13 | 37 | 0.061 | 1.696 | 21.22 | 1.758 | 2.281 | 1.3 | 37/37 | 0.3438 |
| short_video | 15 | 64 | 0.071 | 2.842 | 22.17 | 2.913 | 3.669 | 1.26 | 32/64 | 0.584 |

## Paced realtime

| side | frames accepted | dropped | output chunks | first output (s) | session (s) | output |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 4 | 0 | 12 | 1.446 | 17.524 | <|silence|><|round_start|><|silence|><|response|><|response|><|response|>The col |
| native | 4 | 0 | 1 | 1.493 | 1.565 | The color is red. |

## Loading

- `prompt_only`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `image`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `short_video`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
