# Paced realtime fixtures

Captured on RTX 5090 (SM120), BF16, greedy, checkpoint revision
`2cb8df5c2adae6b59653bbdd783dc580cf440175`.

| file | what it records |
| --- | --- |
| `events.jsonl` | reference session trace for the `timestamped_stream` timeline: frame acceptance, output chunks, lifecycle |
| `realtime-report.json` | reference session summary: 4 frames accepted, 15 output chunks, prompt-to-first-output |
| `steps.jsonl` | every reference forward: input tokens, the exact 3-axis positions it used, and its argmax |
| `native-paced-events.jsonl` | native paced driver trace, first version (frame-per-segment) |
| `native-paced-plan-events.jsonl` | native paced driver trace using the reference's publication plan |
| `batch-plan.json` | the publication plan that mirrors the reference's grouping |
| `realtime-step-parity.json` | native replay of the reference's steps: 12 of 13 argmax decisions agree |
| `timeline-comparison.json` | phase-aligned comparison of the two traces |

## What the reference protocol does

`steps.jsonl` shows the realtime contract directly:

- the prompt is prefilled as `system + user + assistant header` (72 tokens);
- the model answers `<|silence|>` to mean "nothing to report yet";
- when input arrives, the next segment is
  `<|silence|><|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
  followed by one `<|silence|>` and one frame segment per frame that arrived;
- a frame segment is
  `<|vision_start|><|time_start|>{seconds} seconds<|time_end|><|image_pad|><|vision_end|>`,
  and the `<|image_pad|>` token takes `running + max(eh, ew)` with the frame grid
  placed at the running position;
- `<|response|>` opens a text answer, `<|silence|>` returns to waiting.

## Native paced replay

`replay_paced.py --batch-plan batch-plan.json` reproduces the reference protocol:
prompt prefill, a silence decision while waiting, then a segment of
`[silence][im_end]\n[im_start]user\n{prompt}[im_end]\n[im_start]assistant\n[silence][frame segments]`
for each batch of input. With the plan above the native run reproduces the
reference's first two decisions exactly — silence on the prompt, then
`<|response|>` on the batch (segment argmax 151672, max logit 26.5 against the
reference's 26.625) — and then generates "The color is red." where the reference
generated "The color is alternating between red and blue.". The segment tokens
and the 3-axis positions are verified bit-identical to the reference's own
(`steps.jsonl`), so the remaining text difference is the same near-tie mechanism
seen in the offline cases, not a protocol error.

## Replaying the steps natively

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/refsite python tests/assets/moss_vl_realtime/replay_paced.py \
  --checkpoint /path/to/MOSS-VL-Realtime \
  --frame red.ppm --frame blue.ppm --frame red.ppm --frame blue.ppm \
  --reference-steps tests/assets/moss_vl_realtime/reference/realtime/steps.jsonl \
  --out artifacts/moss/step-parity
```

This feeds the reference's own tokens and positions through the native model,
publishing each frame when its `<|image_pad|>` token arrives and rebuilding the
per-row visibility the way the reference does. It is the comparison that
separates model parity from driver protocol: 12 of 13 argmax decisions agree,
including both silence decisions, the response marker and the first answer
tokens. The single disagreement (step 6) is a near-tie: the reference's margin
there is 26.5 against a 28.0 runner-up on the native side, and the reference
picks the native's token on the following step.
