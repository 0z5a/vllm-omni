# HunyuanVideo-1.5 text encoder online FP8

HunyuanVideo-1.5 supports opt-in online FP8 for its Qwen text encoder:
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
  --height 480 --width 832 --num-frames 33 --num-inference-steps 50 \
  --guidance-scale 1.0 --seed 42 --enforce-eager \
  --prompt "A golden retriever walks through a grassy meadow in gentle sunlight." \
  --output hunyuan15_encoder_fp8.mp4
```

To measure the baseline, omit `--quantization-config` and keep every other
setting identical. Compare complete video requests after warmup, recording
encoder time, E2E time, peak memory and output quality. Weight compression
does not by itself imply lower E2E latency.
