# MOSS-VL-Realtime (native offline executor)

## Summary

- Model: `OpenMOSS-Team/MOSS-VL-Realtime`, revision `2cb8df5c2adae6b59653bbdd783dc580cf440175`
- Task: text generation over text, images, and timestamped video frames
- Hardware validated: one RTX 5090 (SM120, 32 GB), CUDA 13.0, torch 2.13.0+cu130
- Precision: BF16, greedy decoding
- Interface: Python offline executor (`vllm_omni.model_executor.models.moss_vl_realtime`)

This recipe covers the **offline single-device executor**. It is not a served
endpoint: paged attention, continuous batching, tensor parallelism, quantization,
and the duplex realtime session are not implemented yet. See
[docs/models/moss_vl_realtime.md](../../docs/models/moss_vl_realtime.md) for the
exact capability boundary.

## Checkpoint

The published checkpoint ships remote code and a converted SGLang variant. Only
the original BF16 checkpoint is supported; the executor refuses a checkpoint
whose weights carry no cross-attention gates instead of loading something that
would silently not match.

```bash
export HF_ENDPOINT=https://hf-mirror.com   # when huggingface.co is not reachable
hf download OpenMOSS-Team/MOSS-VL-Realtime \
  --revision 2cb8df5c2adae6b59653bbdd783dc580cf440175 \
  --local-dir ckpt/MOSS-VL-Realtime

# Verify before using the weights.
(cd ckpt/MOSS-VL-Realtime && sha256sum -c <<'EOF'
3fe2d46a92e3c036e8dfbb2519f65415f8f1ab4f731ffdcb9c8bbff4e66fc3d6  model-00001-of-00005.safetensors
18f3579907933bad721053b3ac405c51045db4539fedcc7ee3c8db79e79f70e5  model-00002-of-00005.safetensors
cecf1e02afe5c7250bb43d6b6e99c7cb430a02afc07a2d697ab1dc4b5f784567  model-00003-of-00005.safetensors
57d794a266a1283d209d054114d9888a6a6b36a0af87fbf314c92b76c89dc294  model-00004-of-00005.safetensors
34009172ba732eb447caf1293126bba35eb934a622677ac7c24bb2d6af462b0d  model-00005-of-00005.safetensors
EOF
)
```

## Preparing inputs

The checkpoint processor targets transformers 4.57.1; later releases removed the
image processor it imports. Install the reference stack as an overlay so the
host environment is untouched:

```bash
pip install --target refsite --no-deps transformers==4.57.1 huggingface_hub==0.34.0 kernels==0.11.7

PYTHONPATH=$PWD/refsite python tests/assets/moss_vl_realtime/prepare_inputs.py \
  --checkpoint ckpt/MOSS-VL-Realtime \
  --prompt "Describe the image." --image tests/assets/moss_vl_realtime/red.ppm \
  --out artifacts/moss/native-inputs/red
```

`prepare_inputs.py --case <id> --verify-against <captured-run>` proves the
preparer reproduces a captured reference run bit-exactly, which is how the
replayable-input claim is checked.

## Running

```bash
CUDA_VISIBLE_DEVICES=1 python examples/offline_inference/moss_vl_realtime/offline_native.py \
  --model ckpt/MOSS-VL-Realtime \
  --inputs artifacts/moss/native-inputs/red \
  --max-new-tokens 64
```

Replay a captured reference run and compare:

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.model_executor.models.moss_vl_realtime.runner \
  --checkpoint ckpt/MOSS-VL-Realtime \
  --run-dir artifacts/moss/<rev>/image/<run-id> \
  --out artifacts/moss/native/image --compare --max-new-tokens 64
```

## Validated results

Reference and native runs on the same RTX 5090, BF16, greedy, warm-up before
timing:

| case | prompt tokens | generated | reference wall (s) | native wall (s) | speedup | native decode (tok/s) | greedy tokens matching |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 11 | 10 | 2.374 | 0.440 | 5.40x | 24.5 | 10 / 10 |
| image | 13 | 37 | 2.448 | 1.562 | 1.57x | 23.9 | 37 / 37 |
| short_video | 15 | 64 | 3.794 | 2.713 | 1.40x | 23.9 | 32 / 64 |

Positions, visibility metadata, patch embedding and the vision rotary embedding
match exactly; prefill logits agree to 0.16–0.58 absolute on a logit scale of
18–23, and the one divergence (`short_video`, token 33) starts on a near-tie.
Loading reports 895/895 keys, 0 missing, 0 unexpected, 21.17 GiB peak device
memory.

## Notes

- The executor consumes vision states through cross-attention only: one
  `<|image_pad|>` token per image and per video frame, and the `<|image_pad|>`
  position block advances text positions by `max(h, w) + 1`.
- A query with no visible frame contributes nothing, and both the attention and
  MLP branches are re-gated with the row mask.
- `sm120_probe.py` qualifies the operators on the target device and reports
  measured error against float32 instead of relying on an architecture macro.
