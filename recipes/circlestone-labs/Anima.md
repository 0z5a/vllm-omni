# Anima

> Native single-file diffusion text-to-image

## Summary

- Vendor: circlestone-labs
- Model: [`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)
- Task: text2img
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

Use this recipe to run Anima via vLLM-Omni's native `AnimaPipeline`. Anima is a
Cosmos-style diffusion transformer text-to-image model distributed as a
single-file transformer checkpoint, not as a normal Hugging Face model
directory.

The native path reads the Anima transformer checkpoint directly, converts
original Cosmos transformer keys when needed, and loads the Cosmos transformer
and text conditioner into native vLLM-Omni modules. Non-denoiser components
such as `text_encoder`, `tokenizer`, `t5_tokenizer`, `vae`, and optionally
`scheduler` must be supplied through a Diffusers-layout components directory.

Native Anima currently supports baseline single-GPU execution. Cache-DiT,
TeaCache, CPU offload, layer-wise offload, TP/SP, CFG parallel,
HSDP, and step execution are not supported by `AnimaPipeline` yet.
Dynamic online FP8 is supported for transformer attention and feed-forward
projections using `--quantization fp8` in the offline example, or
`--diffusion-quantization-config '{"transformer":{"method":"fp8"}}'`
when serving.

## References

- Offline example:
  [`examples/offline_inference/text_to_image/README.md`](../../examples/offline_inference/text_to_image/README.md)
- Supported model entry:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- HuggingFace model page:
  [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- Diffusers-layout components:
  [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers)

## Hardware Support

## ROCm

### 1x AMD MI300X

#### Environment

- OS: Ubuntu 22.04.5 LTS (x86_64)
- Python: 3.12.13
- Driver / runtime: ROCm 7.2.53211, HIP runtime 7.2.53211, MIOpen 3.5.1
- PyTorch: 2.10.0+git8514f05 built with ROCm 7.2.53211
- GPU: 1x AMD MI300X
- vLLM version: 0.23.0
- vLLM-Omni version or commit: 0.1.dev2002+g704724675.rocm

Recommended environment variables:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ROCM_USE_AITER=0
export PYTORCH_ROCM_ARCH=gfx942
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

#### Command - prepare assets

Download the official transformer checkpoint and the Diffusers-layout component
directory:

```bash
mkdir -p /path/to/models/anima-official
mkdir -p /path/to/models/anima-components

hf download circlestone-labs/Anima \
    split_files/diffusion_models/anima-base-v1.0.safetensors \
    --local-dir /path/to/models/anima-official

hf download circlestone-labs/Anima-Base-v1.0-Diffusers \
    --local-dir /path/to/models/anima-components

export ANIMA_CHECKPOINT=/path/to/models/anima-official/split_files/diffusion_models/anima-base-v1.0.safetensors
export ANIMA_COMPONENTS=/path/to/models/anima-components
```

The `ANIMA_COMPONENTS` directory must be in Diffusers `from_pretrained()`
layout. Raw auxiliary files such as `qwen_3_06b_base.safetensors` and
`qwen_image_vae.safetensors` are converter inputs; they are not accepted
directly as `components_path`.

#### Command - text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model "$ANIMA_CHECKPOINT" \
    --model-class-name AnimaPipeline \
    --custom-pipeline-args "{\"components_path\":\"$ANIMA_COMPONENTS\"}" \
    --prompt "A cinematic close-up of a glass teapot on a wooden table." \
    --seed 42 \
    --guidance-scale 4.0 \
    --num-inference-steps 50 \
    --height 1024 \
    --width 1024 \
    --output /tmp/anima_output.png
```

Use 1024x1024, 50 denoising steps, `max_sequence_length=512`, one image per
prompt, empty negative prompt, and CFG scale 4.0 when matching Anima's default
Diffusers settings.

#### Verification

Check that `/tmp/anima_output.png` exists and contains a generated image.

#### Notes

- Key flags: `--model-class-name AnimaPipeline` selects the native Anima path;
  `--custom-pipeline-args` supplies `components_path`.
- Keep the default diffusion load format. No deploy config is required when a
  local checkpoint and `--model-class-name AnimaPipeline` are provided.
- Start with `max-concurrency=1` for correctness and latency validation.
- Keep requests at the same resolution when comparing runs.
- Do not enable parallelism, cache acceleration, offload, or quantized
  checkpoint formats for Anima; use dynamic online FP8 instead.

## Online Serving

Anima supports text-to-image generation through the OpenAI-compatible image
generation API.

### Launch

```bash
vllm serve "$ANIMA_CHECKPOINT" \
    --omni \
    --port 8099 \
    --model-class-name AnimaPipeline \
    --custom-pipeline-args "{\"components_path\":\"$ANIMA_COMPONENTS\"}"
```

### Send requests

```bash
curl http://localhost:8099/v1/images/generations \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$ANIMA_CHECKPOINT\",
      \"prompt\": \"A cinematic close-up of a glass teapot on a wooden table.\",
      \"size\": \"1024x1024\",
      \"num_inference_steps\": 50,
      \"guidance_scale\": 4.0,
      \"seed\": 42
    }"
```

The same generation knobs used by other text-to-image recipes apply:
`num_inference_steps`, `seed`, `height` / `width` through `size`, and optional
negative prompting.

## RTX 5090 CFG graph and conditioning-cache E2E

These measurements were collected on the fork CUDA-routing stack (fork PR #32,
base PR #22), before this upstream port. They are historical evidence for the
combined optimization, not a new benchmark of this upstream branch. This port
uses the existing vLLM FP8 operators and adds no custom quantization extension.

The official checkpoint passed SHA256 verification before a full text-to-image
A1/P1/P2/A2 comparison on one RTX 5090 (GPU 5). Each arm used a fresh process,
three warmups, and six measured requests. Requests generated one 1024×1024 image
with 50 steps, CFG 4.0, seed 42, an empty negative prompt, and
`max_sequence_length=512`. The run used BF16, eager execution, no offload, and
the same three positive prompts in every arm. Model loading, first-use CUDA
compilation, and graph capture were excluded from measured requests.

| Arm | Online FP8 route | Full request mean ± SD (s) | Speedup vs pooled A | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| A1 | Existing dynamic FP8 CUDA path | 8.913 ± 0.008 | 1.000× | 7.86 |
| P1 | sm_120 CFG CUDA graphs + request K/V + empty-negative conditioning cache | 8.454 ± 0.020 | 1.055× | 8.25 |
| P2 | Same optimized route | 8.425 ± 0.008 | 1.058× | 8.25 |
| A2 | Existing dynamic FP8 CUDA path | 8.920 ± 0.010 | 1.000× | 7.86 |
| Pooled A / P | Existing / optimized route | 8.916 / 8.440 | **1.056×** | 7.86 / 8.25 |

The pooled means give **1.0565×** full-request speedup; the pooled median ratio
is **1.0570×**. All 12 paired measured images were pixel-exact. Every arm loaded
280 dynamic-FP8 denoiser linears. The optimized warmups captured 336 custom CUDA
quantizer calls at shape 512×1024; the measured graph replays do not invoke the
Python quantizer. The 4096-row denoiser quantizations remain on vLLM's CUDA
reference path. The gain comes from replaying the CFG pair, sharing its common
prefix and request K/V, and reusing deterministic conditioning for the default
empty negative prompt. A separate graph-only comparison reached 1.048×; the
4096×2048 custom quantizer candidate did not improve full-request latency and
is excluded. The cache is used in evaluation mode for the empty negative prompt;
other negative prompts use regular encoding. PyTorch was 2.13.0+cu130 and vLLM
was 0.29.0.

### Torch fallback

Set `VLLM_OMNI_ANIMA_FP8_CUDA_GRAPH=0` to use the original Torch eager
Anima diffusion path. `VLLM_OMNI_FP8_ONLINE_DISABLE=1` also disables graph
installation for compatibility with the benchmark switch. The graph path is
limited to BF16 online FP8 on sm_120, one 1024×1024 image, CFG, and 512-token
conditioning. Unsupported inputs use the original Torch path. A graph setup
or capture failure restores all temporarily replaced forwards and disables
further graph attempts for that pipeline. Errors during actual diffusion are
propagated, so a partially advanced scheduler is never retried. Standard BF16
and the existing vLLM FP8 implementation remain available.

### Calculated combined improvement

Combining the earlier BF16 → online-FP8 factor (1.125×) with the measured
FP8 graph/cache factor (1.0565×) gives **1.125 × 1.0565 = 1.1885625×**,
or approximately **1.189× speedup / 15.9% lower latency** at fixed workload.
This is a calculated composition assuming the factors transfer between the
configurations, not a directly measured BF16 → optimized-FP8 result: the first
comparison used 20 steps and medians, while the graph/cache comparison used
50 steps and pooled means on the fork CUDA-routing stack. The separate measured
results above remain the reproducible evidence.
