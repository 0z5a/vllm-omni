# HunyuanVideo 1.5

> Text-to-video with the HunyuanVideo-1.5 Diffusers checkpoints

## Summary

- Vendor: Tencent (community Diffusers ports)
- Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`
- Task: Text-to-video
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to generate a short clip with HunyuanVideo-1.5 on a memory-
constrained card, trading output size for capacity.

## References

- Model card:
  <https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v>
- Shared runnable example:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)

## Hardware Support

This recipe documents one validated Intel XPU configuration. Extend it with
more hardware sections as community validation lands.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

BF16 weights with model-level CPU offload at a reduced 320x240, 17-frame
shape. The 480p and 720p T2V and I2V checkpoints all use the same command; only
`--model` changes.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 240 --width 320 --num-frames 17 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --output hunyuan_video_15_output.mp4
```

#### Verification

Confirm `hunyuan_video_15_output.mp4` decodes and that the sampled frames match
the prompt.

#### Notes

- Memory usage: 18.3 GiB loaded, 25.7 GiB peak, about 39 s per clip.
- Known limitations: the native 720p geometry does not fit a 32 GB card with
  CPU offload. Online serving is out of scope for this profile.

## NVIDIA: text encoder online FP8

The HunyuanVideo-1.5 T2V pipeline supports opt-in online FP8 for its Qwen text encoder:
`--quantization-config '{"text_encoder":{"method":"fp8"}}'`.
Attention and MLP projections use native vLLM FP8 linear layers with dynamic
activation quantization. Embeddings, norms, the second ByT5 encoder, video
transformer and VAE retain their original precision.

Use an unquantized BF16 or FP16 checkpoint and an SM89+ NVIDIA GPU.
Static activation scales and pre-quantized FP8 text encoder checkpoints are
not supported. The encoder remains replicated under sequence parallelism.

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled \
  --model-class-name HunyuanVideo15Pipeline \
  --quantization-config '{"text_encoder":{"method":"fp8"}}' \
  --ulysses-degree 2 --enable-layerwise-offload --vae-use-tiling \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --height 480 --width 832 --num-frames 33 --num-inference-steps 50 \
  --guidance-scale 1.0 --seed 42 --enforce-eager \
  --prompt "A golden retriever walks through a grassy meadow in gentle sunlight." \
  --output hunyuan15_encoder_fp8.mp4
```

To measure the baseline, omit `--quantization-config` and keep every other
setting identical. Compare complete video requests after warmup, recording
encoder time, E2E time, peak memory and output quality. Weight compression
does not by itself imply lower E2E latency.

## NVIDIA L20 validation

The official distilled 480p T2V checkpoint was tested with two L20 GPUs,
Ulysses degree 2, eager execution, layerwise DiT and encoder offload, and VAE
tiling. Each mode used one warmup followed by six measured 33-frame 480x832
requests with 50 steps, guidance 1, and seed 42.

| Metric | BF16 | Encoder FP8 | Observed comparison |
| --- | ---: | ---: | ---: |
| E2E mean +/- sample SD | 191.935 +/- 40.159 s | 331.371 +/- 67.215 s | 0.579x |
| Text encoder mean | 1.190 s | 2.696 s | 0.441x |
| Rank-0 peak allocated | 13.311 GiB | 13.091 GiB | 0.220 GiB lower |
| Rank-1 peak allocated | 13.311 GiB | 13.091 GiB | 0.220 GiB lower |
| Rank-0 peak reserved | 19.207 GiB | 18.879 GiB | 0.328 GiB lower |

Raw measured E2E samples were
`[136.949, 161.368, 172.495, 221.498, 231.814, 227.484]` seconds for BF16
and `[293.869, 300.022, 291.823, 289.738, 353.248, 459.528]` seconds for
encoder FP8. Other GPU and storage workloads changed during the two runs, so
these timing ratios are shared-host observations and do not show an isolated
quantization speedup.

All 14 videos decoded to the expected shape and repeated prompts produced
identical frame hashes within each mode. Mean paired frame SSIM was 0.8807 for
the dog prompt, 0.8474 for the sailboat prompt, and 0.8756 for the two-duck
prompt. Runtime inspection found 196 FP8 Qwen projections on both ranks; the
second text encoder, video transformer, and VAE retained their original
precision.
