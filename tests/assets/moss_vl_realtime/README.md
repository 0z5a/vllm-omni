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

## Remaining T01 acceptance

- [ ] Verify and download the pinned original checkpoint and remote code; record
  the resolved files and hashes. Do not substitute the converted SGLang weights.
- [ ] Add a reference capture/replay adapter using the pinned HF implementation.
- [ ] Capture processor tensors, patch order, text/visual positions, and
  query-specific visible-prefix boundaries for every applicable case.
- [ ] Capture bounded intermediate tensors, logits, control tokens, and event
  traces; distinguish valid silence, response completion, and cancellation.
- [ ] Record hardware, Python, Torch, Transformers, backend build, and effective
  runtime settings. Validate dependency compatibility for the pinned revisions.
- [ ] Set per-tensor absolute/relative tolerances from measured reference runs;
  use exact comparisons for integer positions and visibility metadata.
- [ ] Add CPU contract checks and reference replay checks alongside the adapter.

Neither GPU inference nor numerical parity has been run for this input-only
seed. The converted checkpoint has a separate unresolved entry and is excluded
from comparisons until conversion equivalence is established.
