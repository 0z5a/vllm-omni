# MOSS-VL-Realtime

MOSS-VL-Realtime (`MossVLForConditionalGeneration`, `model_type: moss_vl`) is an
11B BF16 vision-language model that keeps text generation running while frames
continue to arrive. Vision never enters the text embedding stream: the text
carries one `<|image_pad|>` token per image and per video frame, and the vision
tokens reach the language model only through cross-attention at 12 of the 48
text layers.

Tracking: [RFC #7890](https://github.com/vllm-project/vllm-omni/issues/7890).

## Task status

Named after the tasks in [RFC #7890](https://github.com/vllm-project/vllm-omni/issues/7890).

| Task | State on this branch | Evidence or gap |
| --- | --- | --- |
| T01 reference revisions and fixtures | captured | `tests/assets/moss_vl_realtime/reference/`, capture and paced-capture adapters |
| T02 placement and interface audit | not here | owned by another contributor; this branch consumes its contract |
| T03 configuration, registration, loading | done for the offline path | 895/895 keys, refusal matrix, native resolver, and a `moss_vl` family entry in the shared text-output registry (stop-token contract); the worker-model registry entry stays with T02's placement audit |
| T04 processor and vision encoding | not here | processor consumed from the pinned checkpoint; owned by another contributor |
| T05 text executor and positions | done | positions exact against the capture; prefill/decode consistency checks |
| T06 cross-attention correctness | done for the eager path | visibility matrix, GQA mapping, ragged media, additive-mask parity |
| T07 native offline parity | done for the offline path | `runner.py`/`parity.py`; no served endpoint yet |
| T08 timestamped video input | proposal only | `duplex.py` states and validates the frame-only payload (bounded size, event identity, media-timestamp ordering, accepted ≠ visible) as a reviewable proposal; the shared command vocabulary and wire names stay with #6592 |
| T09 non-audio duplex capability and plugin | model-local session implemented, engine plugin not wired | `session.py` owns the token stream, the paced cursor, publication and the turn budget for one conversation, and exposes no loop of its own, so the engine stays the only scheduler; the engine-side plugin and its capability extension are still T09's to add |
| T10 incremental visual state | initial contract, paced driver and session path | `append_frames` publishes at a step boundary; `replay_paced.py --via-session` drives the real checkpoint through the session and `compare_paced_paths.py` records that both native paths produce the same decisions |
| T11 text output, silence, interruption | protocol implemented, text diverges on near-ties | the paced driver carries the silence decision into the next segment, renders the user turn and reports text separately from silence; the answer text still differs from the reference on near-ties |
| T12 admission limits and lifecycle | native budgets implemented | vision-extent, text-context and per-append frame budgets are declared per session and refused with `BudgetExceededError` instead of silently dropping frames; the engine session lifecycle is not wired |
| T13 paced single-session validation | partial | 12 of 13 teacher-forced step decisions agree; the paced driver reproduces the reference's first two decisions and generates text, and `timeline.py` compares the traces phase by phase; the driver and the session path agree decision for decision on the real checkpoint |
| T14 baseline evidence | partial | baseline tables for the offline cases; the realtime metrics are recorded but not yet aggregated |
| T23 SM120 qualification | done | `sm120_probe.py` executes the operators on RTX 5090 and measures error |
| T24 tests, examples, docs | partial | 89 CPU checks, this page, the recipe and one example; no CI wiring yet |

## Current capability

| Item | State |
| --- | --- |
| Checkpoint loading (BF16, 5 shards, 895 keys) | Validated; no key is skipped silently |
| Text execution, three-axis mRoPE, cross-attention visibility | Validated against a captured reference |
| Offline native execution (prompt-only, image, two-frame sequence) | Validated on RTX 5090, greedy BF16 |
| Vision encoder and processor | Consumed from the pinned checkpoint; the native split is tracked as T04 |
| Paged attention, continuous batching, TP, quantization | Not implemented |
| Serving entry point (chat completions) | Not implemented |
| Paced realtime session, incremental visual KV, silence/interruption | Model-local session and its contract checks; the engine-side session, and therefore a served duplex endpoint, are not implemented |

The implementation lives in
[`vllm_omni/model_executor/models/moss_vl_realtime`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/model_executor/models/moss_vl_realtime).
It is a single-device eager executor: it reads the checkpoint's `config.json`
and safetensors directly and does not execute checkpoint-provided Python.

## Loading a checkpoint

```python
from vllm_omni.model_executor.models.moss_vl_realtime import load_native_model

model, report = load_native_model("/path/to/MOSS-VL-Realtime", device="cuda:0")
print(report.loaded_keys, report.missing_keys, report.unexpected_keys)
```

`report.loaded_keys` counts every tensor copied from the shards;
`missing_keys` and `unexpected_keys` are empty for the published checkpoint.
The two non-persistent RoPE frequency buffers are recomputed and listed under
`recomputed_buffers`.

## Capturing a reference run

`tests/assets/moss_vl_realtime/capture_reference.py` runs the pinned HF
implementation and writes one artifact directory per case containing processor
tensors, positions, visibility metadata, vision staging, per-step logits, memory
and an event trace:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/assets/moss_vl_realtime/capture_reference.py \
  --checkpoint /path/to/MOSS-VL-Realtime --attn-impl sdpa \
  --out artifacts/moss/2cb8df5c-0e016ef7 \
  --case prompt_only --case image --case short_video
```

The fixture README documents the reference environment (transformers 4.57.1
with `huggingface_hub` 0.34 and `kernels` 0.11.7, installed as an overlay).

## Replaying natively and comparing

```bash
python -m vllm_omni.model_executor.models.moss_vl_realtime.runner \
  --checkpoint /path/to/MOSS-VL-Realtime \
  --run-dir artifacts/moss/2cb8df5c-0e016ef7/image/<run-id> \
  --out artifacts/moss/native/image --compare --max-new-tokens 64

python -m vllm_omni.model_executor.models.moss_vl_realtime.parity \
  --checkpoint /path/to/MOSS-VL-Realtime \
  --run-dir artifacts/moss/2cb8df5c-0e016ef7/image/<run-id>
```

The runner warms up before timing, reports prefill and decode separately, and
compares greedy token ids, per-step argmax decisions and prefill logits with the
capture. The parity probe reports a per-stage delta so a mismatch localises to
one tensor instead of a final text difference.

## Measured evidence (RTX 5090, SM120, BF16)

Committing to a real device rather than an architecture macro:

| case | positions and visibility | vision embeddings max Δ (scale) | prefill logits max Δ (scale) | greedy tokens |
| --- | --- | --- | --- | --- |
| prompt_only | exact | 2.00 (250) | 0.156 | 10 / 10 |
| image | exact | 1.25 (233) | 0.344 (23.1) | 37 / 37 |
| short_video | exact | 11.375 (231) | 0.584 (18.5) | 32 / 64 |

| case | reference wall (s) | native wall (s) | wall speedup | native decode (tok/s) |
| --- | --- | --- | --- | --- |
| prompt_only | 2.338 | 0.462 | 5.06x | 22.5 |
| image | 2.281 | 1.758 | 1.30x | 21.2 |
| short_video | 3.669 | 2.913 | 1.26x | 22.2 |

A repeat measurement on the same host and device is committed as
`tests/assets/moss_vl_realtime/reference/baseline-repeat.md`. It records the part
of the comparison that moves: the native decode rate stayed at 21-24 tok/s in both
runs, while the reference's own rate on the image case went from 15.8 to 24.3
tok/s with nothing changed on either side. The host is shared, so at longer
sequences the speedup is a statement about that host at that moment (0.9x to
1.3x); the short-sequence case is the one whose margin survives the variance.
The native path amortises its advantage differently: it scores logits for the new
position only, while the reference offline path projects the full sequence every
step, which is why the margin shrinks as the generated sequence grows.

Device memory for the image case, from the run report: weights
21622.41 MiB, text cache 6.89 MiB, vision cache 0.23 MiB, peak allocated
21676.98 MiB. The caches are negligible at the validated bounded sizes; the model
weights are the entire footprint, which is why capacity work belongs to weight
residency rather than cache tuning.

The hardware qualification probe
(`python -m vllm_omni.model_executor.models.moss_vl_realtime.sm120_probe`)
executes the operators this model uses on the target device and reports measured
error against float32; all four SDPA backends are available on RTX 5090. At the
checkpoint's own shapes the additive-visibility cross-attention (32 query heads,
8 KV heads, head_dim 128) measures 0.0029 max absolute error, the vision tower's
fused qkv 0.0078, a per-frame attention segment 0.0021, and the three-axis mRoPE
is exact. The full report is committed as
`tests/assets/moss_vl_realtime/reference/sm120-qualification.json`.

## Session contract

`moss_vl_realtime.session.MossVLNativeSession` is the model-local half of a
duplex session: it owns one conversation's token stream, paced position cursor,
frame publication, silence bookkeeping and budgets. It deliberately owns no loop
and no registry — `append()` and `step()` are the only entry points, so the engine
stays the only scheduler and the only lifecycle.

```python
from vllm_omni.model_executor.models.moss_vl_realtime.session import MossVLNativeSession

session = MossVLNativeSession(model, frame_encoder=encode, detokenize=decode)
session.start(prompt_ids)                      # one prefill
session.append(frames=[(path, ts_ms)], prompts=["What changed?"], prompt_ids_for=render)
stats = session.segment_decision               # why the model answered as it did
text, silent = session.step()                  # one bounded turn
session.poll_output()                          # text queued for the transport
session.close()                                # releases caches, reports what ran
```

Rules the implementation follows, and the checks that hold it to them:

| Rule | Why | Check |
| --- | --- | --- |
| Publication happens at a step boundary and the append result is the visible extent | no attention step may observe a half-written frame | `test_publication_happens_before_the_step_that_consumes_it` |
| An append is all or nothing: budgets are checked before anything is published or consumed | a refusal must not drop the carried silence marker or leave an unreferenced frame in the cache | `test_a_refused_append_publishes_nothing`, `test_a_refused_append_keeps_the_carried_silence_marker` |
| A silence decision is carried into the next segment and consumes no position | feeding it twice desynchronises the stream | `test_silence_ends_the_turn_without_consuming_a_position`, `test_silence_is_replayed_at_the_head_of_the_next_segment` |
| Visibility is rebuilt per step from that step's tokens, with the frame index spanning every published frame | this is the reference rule the step parity was measured against | `test_the_frame_mask_spans_every_published_frame` |
| The session exposes no loop of its own | one scheduler, one lifecycle in the process | `test_a_session_exposes_no_loop_of_its_own` |

## Paced run (timestamped stream)

Two comparisons are recorded under
`tests/assets/moss_vl_realtime/reference/realtime/`.

**Step parity (teacher forced).** Feeding the reference's own realtime tokens
and positions through the native model, publishing each frame when its
`<|image_pad|>` token arrives and rebuilding the per-row visibility, reproduces
**12 of 13 argmax decisions**, including both silence decisions, the
`<|response|>` marker and the first answer tokens. The one disagreement is a
near-tie whose token the reference itself picks on the following step.

**Paced replay.** Driving the reference's publication protocol from the fixture
timeline — silence carried into the next segment, the user turn rendered through
the chat template, one silence marker, one segment per frame in the batch —
reproduces the reference's first two decisions exactly (silence on the prompt,
then `<|response|>` with segment argmax 151672 and max logit 26.5 against the
reference's 26.625). The segment tokens and the 3-axis positions are
bit-identical to the reference's recorded steps; the generated answer text
differs ("The color is red." against "The color is alternating between red and
blue.") through the same near-tie mechanism as the offline cases.

| side | frames accepted | output chunks | first decision | answer |
| --- | --- | --- | --- | --- |
| reference session | 4 | 15–19 | `<|silence|>`, `<|silence|>`, `<|response|>` | "The color is alternating between red and blue." on one run, "The color is red." on another |
| native paced driver (plan) | 4 | 1 | `<|silence|>`, `<|response|>` | "The color is red." |
| native paced session (plan) | 4 | 1 | `<|silence|>`, `<|response|>` | "The color is red." |

**Session path.** `replay_paced.py --via-session` drives the same timeline through
`MossVLNativeSession` — the contract a duplex engine consumes — and
`compare_paced_paths.py` records the two native traces against each other:
identical event sequences, identical per-segment decisions (argmax 151672 at max
logit 26.5, then 151671 at 24.75), identical published vision extents and
identical position cursors. The session adds the fields an engine needs to report
(`path`, the prefill decision, and a close report), and nothing else differs.

The reference answer is not stable across runs: two paced captures on the same
device and timeline produced two different sentences, and the native driver's
answer matches one of them. Paced behaviour therefore differs by *timing* on both
sides, which is why the step-level comparison above — same tokens, same
positions, 12 of 13 decisions — is the claim that is actually checkable.
`tests/assets/moss_vl_realtime/run_evidence.sh` reproduces the whole set.

## Tolerance basis

Two independent reference captures are bit-identical: all 31 compared tensors per
case, the token ids and the decoded text match exactly
(`tests/assets/moss_vl_realtime/reference/reference-repeatability.json`). The
tolerances therefore describe how close a *different implementation* can be, not
reference noise, and the checks in
`tests/model_executor/models/moss_vl_realtime` enforce that claim on CPU.

## Notes for contributors

- Cross-attention visibility is an additive mask, not a window. A query with no
  visible frame contributes nothing to the residual stream, and the reference
  re-gates both the attention and the MLP branch with that row mask.
- The `<|image_pad|>` text token consumes a whole mRoPE block: the next text
  position advances by `max(h, w) + 1`.
- A sequence that never produces vision states has no cached rope delta, and the
  reference then restarts decode positions at zero. The native path mirrors that
  rule instead of silently correcting it, so captures stay comparable.
- The published checkpoint sets `vision_seq_pad_multiple = 1`; a padded vision
  sequence is rejected explicitly until it is implemented.
