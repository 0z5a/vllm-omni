# MOSS-VL-Realtime reference inputs (T01)

Tracking: [RFC #7890](https://github.com/vllm-project/vllm-omni/issues/7890),
task **T01**. Owner: @0z5a.

This draft establishes immutable source references and tiny deterministic input
fixtures. It does not provide reference-model captures or native model support.
T01 remains open until the capture checklist below is complete.

## Files and replay semantics

- `manifest.json` pins the original BF16 checkpoint, its tokenizer/processor and
  remote code, source repositories, and the bundled attention source tree.
  The checkpoint revision was resolved through the HF mirror API; verify it
  against the original repository when capturing tensors. These pins are a
  capture starting point, not a validated compatible runtime combination.
- `red.ppm` and `blue.ppm` are hand-constructed P6 RGB images, 32 by 32 pixels,
  containing exact constant RGB values. Their file hashes and raw tensor shape,
  dtype, and pixel values are in the manifest. They are synthetic test inputs,
  not model output or a visual-quality dataset.
- `cases.json` contains prompt-only, one-image, short-video, and timestamped-stream
  cases. The short video is an ordered frame sequence, avoiding codec variance.

Read PPM images as RGB without resizing before passing them to the pinned
processor. Processor resizing and resulting patch geometry must be captured;
these source dimensions do not assert a supported native-model resolution.

Events are ordered by their array index. `at_ms` is the arrival offset from the
start of replay; `media_timestamp_ms` is a frame's media time. Equal arrival
offsets preserve array order. Convert media time to seconds at the reference
adapter boundary. Do not sort, deduplicate, drop, or resample frames.

The stream includes a follow-up prompt and two distinct frames at the same media
timestamp. This is an input case whose model behavior still needs measurement,
not an assertion about interruption, silence, positions, or accepted timestamps.
A prompt scheduled during replay is not proof that generation was active then.
Replay with media-clock pacing for behavior comparisons; report unpaced runs
separately. This fixture format is test data, not a proposed serving wire API.

Use `do_sample=false` and `max_new_tokens=64` as the initial capture settings.
Record all effective session, processor, generation, and retention settings,
including defaults, before using a capture for parity. No expected text or
control decisions are fabricated in this draft.

## Reference capture

`capture_reference.py` runs the pinned HF implementation on these fixtures and
writes one artifact directory per case. It never imports the native executor;
the two sides meet only through the recorded tensors.

```bash
# Reference environment: transformers 4.57.1 with huggingface_hub 0.34 and
# kernels 0.11.7 (the checkpoint's remote code uses Qwen2VLImageProcessorFast,
# which later transformers releases removed). Installed as an overlay so the
# host environment is untouched:
pip install --target refsite --no-deps transformers==4.57.1 huggingface_hub==0.34.0 kernels==0.11.7

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/refsite python tests/assets/moss_vl_realtime/capture_reference.py   --checkpoint /path/to/MOSS-VL-Realtime --attn-impl sdpa   --out artifacts/moss/2cb8df5c-0e016ef7   --case prompt_only --case image --case short_video
```

Each run records `environment.json`, `inputs.json`, `load-report.json`,
`contract-report.json`, `numerical-report.json`, `events.jsonl`, `memory.json`,
`performance.json`, `result.md`, plus `inputs.safetensors` (processor tensors)
and `captured_tensors.safetensors` (positions, visibility, vision staging,
cross-attention shapes).

`vllm_omni/model_executor/models/moss_vl_realtime/parity.py` replays the same
inputs through the native executor and reports a per-stage delta, which is how
the reference/derived differences below were localised.

## Recorded reference run (RTX 5090, SM120, BF16)

`do_sample=false`, `max_new_tokens=64`, `attn_implementation=sdpa`.

| case | prompt tokens | frames | generated tokens | reference text |
| --- | --- | --- | --- | --- |
| prompt_only | 11 | 1 (processor blank-image fallback) | 10 | "Hello! How can I assist you today?" |
| image | 13 | 1 | 37 | solid uniform red square, no objects or text |
| short_video | 15 | 2 | 64 (cap) | explanation of why a colour changes |

`reference/manifest.json` carries per-file SHA-256 and sizes;
`reference/tolerances.json` carries the measured native-vs-reference absolute
deltas and the list of stages compared exactly.

## Remaining T01 acceptance

- [x] Verify and download the pinned original checkpoint and remote code; record
  the resolved files and hashes. Do not substitute the converted SGLang weights.
- [x] Add a reference capture/replay adapter using the pinned HF implementation.
- [x] Capture processor tensors, patch order, text/visual positions, and
  query-specific visible-prefix boundaries for every applicable case.
- [x] Capture bounded intermediate tensors, logits, control tokens, and event
  traces; distinguish valid silence, response completion, and cancellation.
- [x] Record hardware, Python, Torch, Transformers, backend build, and effective
  runtime settings. Validate dependency compatibility for the pinned revisions.
- [x] Set per-tensor absolute/relative tolerances from measured reference runs;
  use exact comparisons for integer positions and visibility metadata.
- [x] Add CPU contract checks and reference replay checks alongside the adapter.
- [ ] Capture the timestamped-stream case through the paced session path.

The checkpoint files were verified against the pinned revision before capture;
the converted SGLang checkpoint has a separate unresolved entry and is excluded
from every comparison until conversion equivalence is established.
