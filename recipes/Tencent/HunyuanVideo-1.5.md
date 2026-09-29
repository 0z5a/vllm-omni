# HunyuanVideo-1.5 T2V text encoder online FP8

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


### RTX 5090 CUDA activation quantization E2E (2026-09-29)

Full-model A/P/P/A on gj5090 GPU 7, pinned
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled`
revision `1abb14f06518f37448dcf3a6917dd086dd7045c7`.
All 32 required files were verified against official SHA/Git-blob hashes.
All arms use encoder-only online FP8; A disables CUDA activation quantization,
P enables a temporary sm120 BF16 gate for exactly `(1108,3584)` and
`(1108,18944)`. The existing block kernel is unchanged. These inputs exceed
the default 512-row gate and normally fall back to the reference kernel.

Workload: 384×640, 33 frames, 50 steps, guidance 1, 24 fps, seed 42;
eager execution, layer offload for `dit` and `text_encoder` (both encoders),
VAE tiling. Four fresh processes, three warmups and six measured full requests
each. Loading and output export are excluded from timing.

| Arm | CUDA path | E2E mean ± sample SD (s) | Peak allocated (GiB) |
| --- | --- | ---: | ---: |
| A1 | Reference | 27.753 ± 0.020 | 13.14 |
| P1 | Temporary narrow gate | 27.799 ± 0.020 | 13.14 |
| P2 | Temporary narrow gate | 27.792 ± 0.100 | 13.14 |
| A2 | Reference | 27.802 ± 0.027 | 13.14 |

Pooled A/P **0.999×**, A2/A1 drift 1.002× and P2/P1 drift 1.000×:
no reliable E2E benefit. The temporary Python gate was reverted and its SHA
checked against the original backup. No model-specific CUDA candidate is
included; model support and shared #22 routing remain.

**12/12 raw uint8 videos are bit-exact.** Census: 196 FP8 modules,
6,525,288,448 weight elements. P warmups confirm 196 CUDA calls/request
(168 at width 3584 and 28 at width 18944); A has zero. Counting is removed
before measured requests. Independent seed 0/1/42 checks including zero rows
match reference FP8 payload and scale bit-for-bit for both shapes.

Evidence: `results/hunyuan15/abba_gpu7`, `hunyuan15_e2e_hotpath.py`,
`run_hunyuan15_abba.sh`, `summarize_hunyuan15_abba.py`,
`hunyuan15-1108-correctness.json`. A test-source-only platform compatibility
overlay is identical in all arms; the installed environment is unchanged.
Initial configuration failure and shape audit are retained separately and
excluded from performance evidence. All four processes exited naturally.
Earlier BF16/FP8 model comparisons above are independent of this CUDA test.
