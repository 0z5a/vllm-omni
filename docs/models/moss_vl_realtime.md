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
| T03 configuration, registration, loading | done | 895/895 keys, refusal matrix, `registry.py` |
| T04 processor and vision encoding | not here | processor consumed from the pinned checkpoint; owned by another contributor |
| T05 text executor and positions | done | positions exact against the capture; prefill/decode consistency checks |
| T06 cross-attention correctness | done for the eager path | visibility matrix, GQA mapping, ragged media, additive-mask parity |
| T07 native offline parity | done for the offline path | `runner.py`/`parity.py`; no served endpoint yet |
| T08 timestamped video input | not started | needs the shared duplex input extension |
| T09 non-audio duplex capability and plugin | not started | needs the shared engine plugin contract |
| T10 incremental visual state | initial contract | `append_frames` publishes at a step boundary; reference-timeline segment splicing is not implemented |
| T11 text output, silence, interruption | partial | the paced driver treats `<|silence|>` as a turn end; the full control/state mapping is not implemented |
| T12 admission limits and lifecycle | not started | needs the engine session lifecycle |
| T13 paced single-session validation | partial | paced capture, native paced driver and the timeline comparator exist; no end-to-end paced acceptance run yet |
| T14 baseline evidence | partial | baseline tables for the offline cases; the realtime metrics are recorded but not yet aggregated |
| T23 SM120 qualification | done | `sm120_probe.py` executes the operators on RTX 5090 and measures error |
| T24 tests, examples, docs | partial | 34 CPU checks, this page, the recipe and one example; no CI wiring yet |

## Current capability

| Item | State |
| --- | --- |
| Checkpoint loading (BF16, 5 shards, 895 keys) | Validated; no key is skipped silently |
| Text execution, three-axis mRoPE, cross-attention visibility | Validated against a captured reference |
| Offline native execution (prompt-only, image, two-frame sequence) | Validated on RTX 5090, greedy BF16 |
| Vision encoder and processor | Consumed from the pinned checkpoint; the native split is tracked as T04 |
| Paged attention, continuous batching, TP, quantization | Not implemented |
| Serving entry point (chat completions) | Not implemented |
| Paced realtime session, incremental visual KV, silence/interruption | Not implemented |

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
| prompt_only | 2.374 | 0.440 | 5.40x | 24.5 |
| image | 2.448 | 1.560 | 1.57x | 23.9 |
| short_video | 3.794 | 2.710 | 1.40x | 23.9 |

The hardware qualification probe
(`python -m vllm_omni.model_executor.models.moss_vl_realtime.sm120_probe`)
executes the operators this model uses on the target device and reports measured
error against float32; on RTX 5090 all four SDPA backends are available and the
additive-visibility cross-attention path measures 0.0019 max absolute error at
BF16.

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
