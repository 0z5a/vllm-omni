# ERNIE-Image

> Text-to-image online serving (ERNIE-Image 8B)

## Summary

- Vendor: Baidu
- Model: `baidu/ERNIE-Image` / `baidu/ERNIE-Image-Turbo`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`baidu/ERNIE-Image` or `baidu/ERNIE-Image-Turbo` with vLLM-Omni for
high-quality text-to-image generation.

ERNIE-Image is an 8B-parameter Diffusion Transformer (DiT) model that achieves
state-of-the-art performance among open-weight text-to-image models. Key
strengths include:

- **Text rendering:** Excellent for dense, long-form, and layout-sensitive
  text — ideal for posters, infographics, and UI-like images.
- **Instruction following:** Reliably handles complex prompts with multiple
  objects, detailed relationships, and knowledge-intensive descriptions.
- **Structured generation:** Effective for posters, comics, storyboards, and
  multi-panel compositions where layout matters.
- **Style coverage:** Supports realistic photography, design-oriented imagery,
  and stylized aesthetic outputs.
- **Practical deployment:** Can run on consumer GPUs with 24GB VRAM.

Two model variants are provided:

1. **ERNIE-Image** — The SFT model with stronger general-purpose capability
   and instruction fidelity, typically using 50 inference steps.
2. **ERNIE-Image-Turbo** — Optimized by DMD and RL for faster speed and
   higher aesthetics, requiring only 8 inference steps.

## References

- Upstream model card: <https://huggingface.co/baidu/ERNIE-Image>
- Upstream model card (Turbo): <https://huggingface.co/baidu/ERNIE-Image-Turbo>
- GitHub: <https://github.com/baidu/ERNIE-Image>
- Blog: <https://ernie-image.github.io/>

## Online FP8

Enable dynamic FP8 for both the DiT and Mistral language decoder with an
unquantized ERNIE-Image or ERNIE-Image-Turbo checkpoint:

```bash
vllm serve baidu/ERNIE-Image-Turbo --omni \
  --quantization-config '{"transformer": {"method": "fp8"}, "text_encoder": {"method": "fp8"}}' \
  --port 8091
```

Text encoder FP8 quantizes attention and MLP projections in
`text_encoder.language_model.layers`. Token embeddings, normalization,
the vision tower, multimodal projector, optional Prompt Enhancer and VAE
retain their original precision. A global `--quantization fp8` continues to
apply to the DiT only. Text encoder FP8 requires BF16 or FP16 source weights
and a GPU supported by vLLM's FP8 backend; prequantized checkpoints and static
activation scales are unsupported for this component.

Use `ignored_layers` within the `text_encoder` configuration to retain selected
projections at their original precision. Layer names use the complete prefix,
for example `text_encoder.language_model.layers.0.self_attn.q_proj`.

## Hardware Support

This recipe currently documents tested configurations for CUDA GPU serving.
Add more sections for other hardware as community validation lands.

## GPU

### 1 x RTX 4090 (Single GPU, 24GB VRAM, Minimum Recommended)

#### Command

**ERNIE-Image (full model):**

```bash
vllm serve baidu/ERNIE-Image --omni \
  --enable-layerwise-offload \
  --port 8091
```

**ERNIE-Image-Turbo (distilled, faster):**

```bash
vllm serve baidu/ERNIE-Image-Turbo --omni \
  --enable-layerwise-offload \
  --port 8091
```

### 2 x RTX 4090 (Multi-GPU, 24GB VRAM)

#### Command

**ERNIE-Image (full model):**

```bash
vllm serve baidu/ERNIE-Image --omni \
  --tensor-parallel-size 2 \
  --enable-cpu-offload \
  --port 8091
```

**ERNIE-Image-Turbo (distilled, faster):**

```bash
vllm serve baidu/ERNIE-Image-Turbo --omni \
  --tensor-parallel-size 2 \
  --enable-cpu-offload \
  --port 8091
```

#### Verification

After the server is ready, test with a simple request:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of a cat sitting on a laptop keyboard, digital art style.",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

For ERNIE-Image-Turbo, reduce the inference steps:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of a cat sitting on a laptop keyboard, digital art style.",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

#### Notes

- **Memory usage:** The 8B model requires significant VRAM. Use
  `--enable-cpu-offload` to reduce GPU memory footprint by offloading
  components to CPU when not in use.
- **Key flags:**
    - `--omni` — enables vLLM-Omni diffusion serving.
- **Advanced features:**
    - **TP (Tensor Parallelism):** `--tensor-parallel-size <N>` — distribute
    model weights across N GPUs.
    - **SP (Sequence Parallelism):** `--usp <N>` (Ulysses SP) and `--ring <N>`
    (Ring SP) for long-sequence workloads.
    - **HSDP:** `--use-hsdp` enables Hybrid Sharded Data Parallelism; use
    `--hsdp-shard-size` and `--hsdp-replicate-size` for fine-grained control.
    - **Cache-DiT:** `--cache-backend cache_dit` caches DiT intermediate outputs
    for faster generation; configure via `--cache-config`.
    - **Layer offload:** `--enable-layerwise-offload` offloads DiT layers to CPU
    for memory-constrained scenarios.
- **Prompt Enhancer (PE):** ERNIE-Image includes an optional 3B-parameter
  Prompt Enhancer model that expands brief user inputs into richer structured
  descriptions, improving output quality. Set `use_pe=False` in the request
  body to disable it if you prefer direct prompt processing or want to use
  larger LLMs (e.g., Gemini, ChatGPT) for prompt enhancement instead.
- **Recommended settings:**
    - ERNIE-Image: `num_inference_steps=50`, `guidance_scale=4.0`
    - ERNIE-Image-Turbo: `num_inference_steps=8`, `guidance_scale=1.0`


### RTX 5090 CUDA activation quantization E2E (2026-09-29)

Full-model A/P/P/A on gj5090 GPU 7 with `baidu/ERNIE-Image-Turbo`
revision `bc68c81e2a1730a394d5fc9fae70713dee940140`.
All 13 required files were verified against revision SHA/Git-blob hashes.
Prompt enhancement is disabled; enhancer weights are excluded.
All arms use combined transformer and text-encoder online FP8.
A disables CUDA activation quantization; P enables the existing shared route.
No model-specific gate or kernel changes were introduced.

Workload: 1024×1024, 8 steps, guidance 1, seed 42, eager execution,
VAE tiling, no offload. Four fresh processes, three warmups and six measured
full requests each, cycling three prompts. Loading and output export are
excluded from timing; call counters are removed before measured requests.

| Arm | CUDA path | E2E mean ± sample SD (s) | Peak allocated (GiB) |
| --- | --- | ---: | ---: |
| A1 | Reference | 2.317 ± 0.015 | 14.65 |
| P1 | Shared route | 2.324 ± 0.011 | 14.65 |
| P2 | Shared route | 2.322 ± 0.011 | 14.65 |
| A2 | Reference | 2.320 ± 0.007 | 14.65 |

Pooled A/P **0.998×**, A2/A1 drift 1.002×, P2/P1 drift 0.999×:
no reliable CUDA E2E benefit. No model-specific CUDA candidate is included;
model support and shared #22 routing remain.

**12/12 raw uint8 images are bit-exact.** Census: 252 DiT FP8 modules
(7,851,737,088 weight elements) and 182 text FP8 modules
(3,026,190,336 weight elements). P warmups confirm 182 CUDA calls/request:
130 at width 3072, 26 at width 4096, and 26 at width 9216, with BF16
inputs of 15/16/19 rows across the prompts. A has zero CUDA calls.
The full-request shape audit also observes DiT inputs `(4115,4096)` and
`(4115,12288)` falling back under the existing row gate.

Evidence: `results/ernie/abba_gpu7`, `ernie_e2e_hotpath.py`,
`run_ernie_abba.sh`, `summarize_ernie_abba.py`, and the separate
`results/ernie/shape_audit_gpu7` audit. The test-source-only platform
compatibility overlay is identical in all arms; the installed environment
is unchanged. All four processes exited naturally. Earlier PRO 6000
BF16/FP8 comparisons above are independent of this CUDA test.
