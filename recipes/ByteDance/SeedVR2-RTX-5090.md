# SeedVR2 video restoration — RTX 5090

## Summary

- Vendor: ByteDance
- Model: SeedVR2 3B
- Task: restore or upscale an uploaded video
- Mode: native offline/HTTP pipeline, one Euler step
- Hardware: NVIDIA GeForce RTX 5090, 32 GiB per device
- Maintainer: 0z5a

## When to use this recipe

Run short, constant-frame-rate videos with the native SeedVR2 pipeline. This
recipe covers the whole-clip path. Long-video batching, overlap, and color
correction from external applications are separate execution semantics.

## Supported model contract

| Property | Contract |
| --- | --- |
| Model directory | See [checkpoint files and hashes](../../docs/models/seedvr2.md#model-directory) |
| Input | One uploaded video; blank prompt; positive dimensions divisible by 16 |
| Output | MP4 at requested geometry, original frame count and source frame rate |
| Audio | First mono/stereo track, aligned by PTS; re-encoded as AAC |
| Sampling | One Euler step, CFG=1, per-request seed |
| Precision | 3B DiT and VAE FP16 |
| Parallelism | Specialized USP via `ulysses_degree`; replicated model weights |
| Frame padding | Internal 4n+1 padding, cropped back; five returns five, six returns six |

## References

- [Canonical SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [Model guide](../../docs/models/seedvr2.md)
- [Integration RFC](https://github.com/vllm-project/vllm-omni/issues/7723)

## Hardware

RTX 5090 with 32 GiB per GPU, PCIe, one GPU for the default command or two/four
for USP. Each GPU must fit a complete model copy and its activations. Host RAM
must accommodate checkpoint staging per rank. Short-clip validation does not
establish the maximum frame count or output resolution.

## Software environment

Linux x86-64, Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.0, vLLM 0.29.0, and
PyAV in the existing environment. Use the vLLM-Omni branch containing this recipe.

## Command

Set `MODEL_DIR` to the authorized checkpoint directory documented in the model
guide; it must also contain `model_index.json` selecting `SeedVR2Pipeline`.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --num-gpus 1 --host 127.0.0.1 --port 8098
```

For USP=2, expose two GPUs and replace the GPU count with:

```bash
--num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2}}'
```

Use degree 4 and four visible GPUs for USP=4. Combine any feature-specific
settings below into the same stage-overrides object; do not repeat that flag.

## Verification

```bash
curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' \
  -F 'guidance_scale=1' -F 'seed=7723' --output restored.mp4
ffprobe -v error -count_frames -show_streams -of json restored.mp4
ffmpeg -v error -i restored.mp4 -f null -
```

Expect HTTP 200 and a completely decodable 224×128 video. Check the frame count,
FPS, timestamps, and audio against the input. Repeat the same seed and compare
decoded frames; changing the seed should change the restored video. Inspect
matching first/middle/last frames at the same display scale. AAC is re-encoded,
so input/output audio packet hashes need not be identical.

For transformer-only SP=1/2/4 parity, set `VLLM_TEST_SEEDVR2_MODEL` to the 3B
safetensors file and run:

```bash
python -m pytest -o addopts='' -v tests/diffusion/models/seedvr2/test_seedvr2_e2e.py
```

This checkpoint-gated test does not validate the HTTP server or an optional
optimization. Feature validation must use its enabled configuration, full model
outputs, and the actual backend selected by the worker.

## Supported features

| Feature | Status |
| --- | --- |
| [USP](../../docs/models/seedvr2.md) | Model-local regular/shifted window attention |
| CPU offload, LoRA, compiled execution, CFG/TP/PP | Unsupported |
| VFR or multichannel audio | Unsupported |
| Quantization / VAE tiling | Only when explicitly added by the feature branch below |

## Measurement scope

Compare identical inputs, seeds, dimensions, GPU counts, and warmups. Separate
correctness from performance: use repeated interleaved HTTP measurements before
claiming a stable gain. Keep raw logs, tensors, and generated media outside the
source diff; link selected visuals from the PR's test results.
