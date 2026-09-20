# MOSS-VL-Realtime baseline

> Repeat measurement, same host, same device, same commands as
> [`baseline.md`](baseline.md). It is kept because the two runs disagree in a way
> worth recording: the native decode rate is stable at 21-24 tok/s across both
> runs, while the reference's own rate moved from 15.8 to 24.3 tok/s on the image
> case with no change on either side of the comparison. The host is shared, so a
> speedup at longer sequences is a statement about that host at that moment; the
> short-sequence case is the one whose margin survives the variance.

## Reference capture

| case | prompt tokens | generated | wall (s) | decode (tok/s) | backend | device |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 1.461 | 6.16 | sdpa | None |
| image | 13 | 37 | 1.482 | 24.29 | sdpa | None |
| short_video | 15 | 64 | 2.432 | 25.9 | sdpa | None |

## Native replay

| case | prompt | generated | prefill (s) | decode (s) | decode (tok/s) | wall (s) | reference wall (s) | speedup | tokens | logits max Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 0.059 | 0.357 | 25.19 | 0.416 | 1.461 | 3.51 | 10/10 | 0.1562 |
| image | 13 | 37 | 0.067 | 1.587 | 22.68 | 1.655 | 1.482 | 0.9 | 37/37 | 0.3438 |
| short_video | 15 | 64 | 0.064 | 2.661 | 23.68 | 2.725 | 2.432 | 0.89 | 32/64 | 0.584 |

## Paced realtime

| side | frames accepted | dropped | output chunks | first output (s) | session (s) | output |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 4 | 0 | 12 | 1.276 | 17.387 | <|silence|><|round_start|><|silence|><|response|><|response|>The color is red.<| |
| native | 4 | 0 | 1 | 2.073 | 2.187 | The color is red. |

## Loading

- `prompt_only`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `image`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
- `short_video`: loaded 895 keys, 0 missing, 0 unexpected, peak 21.17 GiB
