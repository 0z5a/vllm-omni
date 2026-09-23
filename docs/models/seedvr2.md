# SeedVR2 3B video restoration

The native `SeedVR2Pipeline` restores an input video using the released 3B FP16
NaDiT and s8/c16/t4 causal VAE. It uses fixed checkpoint conditioning, CFG=1,
and one Euler step. Text prompts do not change conditioning.

## Model directory

Place these files together, retaining their names:

| File | SHA-256 of the validated release |
| --- | --- |
| `seedvr2_ema_3b_fp16.safetensors` | `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304` |
| `ema_vae_fp16.safetensors` | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |
| `pos_emb.pt` | `fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799` |

Add `model_index.json` containing:

```json
{"_class_name": "SeedVR2Pipeline"}
```

The loader checks every checkpoint tensor, including RoPE buffers, and rejects
missing, duplicate, unexpected, or incompatible tensors. Install PyAV for video
decoding. The validated environment uses PyAV 18.1, PyTorch 2.13/CUDA 13, and L20 GPUs.

## Serve and restore a video

```bash
vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --num-gpus 1 --host 127.0.0.1 --port 8098

curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' \
  -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' \
  -F 'guidance_scale=1' -F 'seed=7723' \
  --output restored.mp4
```

For two-rank USP, expose two GPUs and replace `--num-gpus 1` with:

```bash
--num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2}}'
```

Use degree 4 and four visible GPUs for SP4. Each rank runs the whole VAE, so
USP does not reduce VAE memory requirements.

The multipart API requires a prompt field; a single space supplies a blank
prompt. Nonblank text is rejected. Omit `fps` to retain the source frame rate;
an explicit rate must match the source. Output dimensions are explicit multiples
of 16. Input frames are resized with antialiased bicubic interpolation. The final
frame is repeated internally to reach 4n+1, and the decoded output is cropped back
to the original frame count. Five frames remain five; six frames are internally
padded to nine and return six.

## Current integration scope

| Property | Contract |
| --- | --- |
| Weights / dtype | Released 3B DiT and VAE, FP16 |
| Reference semantics | C0 whole clip, no color correction |
| Video input | One uploaded file, or offline TCHW RGB floats / PIL frames |
| Timing | Constant frame rate, increasing PTS; normalize the video origin to zero |
| Audio | First mono/stereo track, aligned by source PTS, cropped to the video interval, re-encoded as AAC |
| Randomness | Per-request generator; preserve reference latent strides when sampling noise |
| Sequence parallelism | Native engine SP1/2/4 correctness verified on five-frame L20 cases |
| VAE placement | Replicated on participating ranks in this integration |
| Unsupported | VFR, multichannel audio, 7B, other sampling schedules, quantization, cache acceleration, VAE tiling/slicing, CPU offload, CFG/TP/PP/VAE parallelism, compiled execution, LoRA |

Unsupported engine modes are rejected before process hooks and worker creation.
Use `ulysses_degree=N` for specialized Ulysses window attention. Video tokens
remain sequence-sharded outside attention; each head shard attends within the
reference regular/shifted windows. Uneven head shards support 20 heads on eight
ranks. Ring, AllGather-KV, advanced-UAA, and permute mode are unsupported.

The SP measurements below describe the earlier window-aligned implementation;
they are not validation results for the new Ulysses path.

Native SP1 five-frame output is bitwise equal to the reference fixture. SP2/SP4
frame error is MAE 0.000171566 and max 0.00387573 against SP1, below the declared
0.02/0.25 gates. These shared-host checks establish correctness, not scaling or
stable service speedup.

The practical P0 reference's five-frame batches, overlap, LAB correction,
32-block CPU swapping, and VAE tiling are separate execution semantics and are
not reproduced by this whole-clip path. A 107-frame 448×256 input targeting 896×512 exceeds a 44 GiB L20 during
whole-clip VAE encoding (43.14 GiB allocated, an additional 11.92 GiB required).
Do not infer large-video capacity from the short-clip SP checks.

Invalid inputs return 400. A model OOM fails that request; a subsequent short
request succeeds. Cancelling an in-progress video job removes it through the
existing DELETE endpoint; in-flight GPU work may drain before memory is reusable.
A lost SP worker fails the request and makes health return 503. Restart the
service to restore availability; automatic rank recovery is not provided.

## Reproducible deployment

See the [RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ByteDance/SeedVR2-RTX-5090.md) for the
input/output contract, complete serving command, and media checks. The local
checkpoint test in `tests/diffusion/models/seedvr2/test_seedvr2_e2e.py` checks
3B transformer USP parity; it is not an HTTP or optional-feature benchmark.
