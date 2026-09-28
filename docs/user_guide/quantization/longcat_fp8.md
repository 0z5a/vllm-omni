# LongCat text encoder FP8

These pipelines opt in to dynamic online FP8 with an explicit `text_encoder`
entry. A global `quantization="fp8"` does **not** enable their HF encoder.

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="meituan-longcat/LongCat-Image",  # or meituan-longcat/LongCat-Image-Edit
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

Only the Qwen2.5-VL language decoder's attention and MLP linear layers are
quantized. The vision tower, embeddings, normalization and output head retain
their original precision. This setting does not enable DiT or VAE quantization.
For Image-Edit it also applies to language layers processing image-conditioned
features; for Image it also affects prompt rewriting when enabled.

Use an unquantized checkpoint and dynamic activations. Serialized FP8
checkpoints, static activation scales and other encoder quantization methods
are rejected. Hugging Face loads the original encoder before conversion, so
this reduces resident language weights, not the initial encoder loading peak.
The encoder remains replicated; this option does not add encoder tensor
parallelism. Compare image quality on your prompts before enabling it in
production. For controlled comparisons, disable prompt rewriting in both runs
with `OmniDiffusionSamplingParams(extra_args={"enable_prompt_rewrite": False})`.


## RTX 5090 CUDA hot-path E2E (2026-09-28)

LongCat-Image revision `d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8`;
all required checkpoint files were verified against their official hashes.
Single RTX 5090, eager execution, text-encoder online FP8, no cache,
512×512, 50 steps, guidance 4.5, seed 42, prompt rewrite disabled.
Each A/P/P/A arm used three warmup requests and six measured full requests
(three prompts repeated twice). Counting instrumentation was removed before timing.
All arms used the same test-source IrOpPriorityConfig compatibility overlay;
the installed environment was unchanged.

The existing 512-row gate falls back for this model's BF16 activations:
553×3584 and 553×18944. A temporary SM120-only gate extension was tested.
Both shapes matched reference FP8 payloads and scales exactly for three random seeds.
The candidate made 392 CUDA quantizer calls per warmup request; reference arms made zero.

| Arm | Quantizer | Full request mean ± SD (s) | Peak allocated (GiB) |
| --- | --- | ---: | ---: |
| A1 | Reference | 7.350 ± 0.599 | 21.91 |
| P1 | Candidate CUDA | 7.999 ± 0.430 | 21.91 |
| P2 | Candidate CUDA | 7.583 ± 0.604 | 21.91 |
| A2 | Reference | 7.191 ± 0.383 | 21.91 |

Pooled reference/candidate ratio: **0.933×**. All **12/12 paired images**
were pixel-identical. A2/A1 was 0.978× and P2/P1 was 0.948× on this shared host.
The candidate did not show an E2E benefit and was withdrawn; this PR retains
reference fallback for these shapes. The earlier incomplete GPU3 run is excluded.
LongCat-Image-Edit full E2E is pending.


### DiT compilation comparison

A separate same-GPU A/P/P/A comparison changed only `enforce_eager`:
A uses eager execution, P uses compilation. CUDA activation routing was
**disabled in all four arms**; these results are not CUDA quantizer speedups.
The same 512×512/50-step configuration used three warmups and six measured
full requests per arm. Download writers cooperatively waited during measurement.

| Arm | Execution | Full request mean ± SD (s) |
| --- | --- | ---: |
| A1 | Eager | 7.661 ± 0.301 |
| P1 | Compiled | 6.457 ± 0.058 |
| P2 | Compiled | 6.542 ± 0.109 |
| A2 | Eager | 7.987 ± 0.340 |

The pooled configuration ratio is **1.204×** (7.824 s / 6.500 s).
A2/A1 drift is 1.043×; P2/P1 is 1.013×. Peak allocated memory was 21.91 GiB.
**0/12 cross-configuration image pairs were pixel-identical**. Mean absolute
pixel error was 1.207 on the 0–255 scale, with maximum channel error 189.
This is a speed/rounding tradeoff, not an output-equivalent replacement or a
quality validation. Keep eager execution when exact eager output is required.
