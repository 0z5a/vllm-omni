# Ovis-Image text encoder FP8

Ovis-Image supports online FP8 for the Qwen3 text encoder through an explicit
component configuration:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="ATH-MaaS/Ovis-Image-7B",
    dtype="bfloat16",
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

This converts attention and MLP projections under `text_encoder.layers`.
Embeddings, normalization, the diffusion transformer and VAE retain their
original precision. A global `"fp8"` setting does not opt this encoder in.
Only unquantized BF16/FP16 checkpoints with dynamic activation scaling are
supported. Use full projection prefixes in `ignored_layers`, for example
`text_encoder.layers.0.self_attn.q_proj`.

## Reproduce the image comparison

Use a local copy of the official checkpoint in `OVIS_MODEL`. Set `OVIS_CFG=2`
and expose two GPUs to repeat with CFG parallelism. Each mode performs two
warmups and six measured requests, with identical seeds and 30 denoising steps.

```bash
OVIS_MODEL=/path/to/Ovis-Image-7B OVIS_CFG=1 CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os
import statistics
import time
from pathlib import Path

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

prompts = [
    "A red ceramic teapot and two blue cups on a wooden table, soft window light, detailed product photograph.",
    'A storefront with a clear sign reading "OPEN", a bicycle parked beside the door, watercolor illustration.',
    "Three yellow rubber ducks floating in a turquoise pool, overhead photograph, realistic ripples.",
]
for mode in ("bf16", "fp8"):
    engine = Omni(
        model=os.environ["OVIS_MODEL"], dtype="bfloat16", enforce_eager=True,
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=int(os.environ["OVIS_CFG"])),
        quantization_config={"text_encoder": {"method": "fp8"}} if mode == "fp8" else None,
        enable_diffusion_pipeline_profiler=True,
    )
    output = Path(f"ovis-{mode}")
    output.mkdir(exist_ok=True)
    timings = []
    for index in range(8):
        prompt = prompts[0 if index < 2 else (index - 2) % 3]
        params = OmniDiffusionSamplingParams(
            width=768, height=512, num_inference_steps=30,
            guidance_scale=4.0, seed=142, num_outputs_per_prompt=1,
        )
        start = time.perf_counter()
        result = engine.generate(
            {"prompt": prompt, "negative_prompt": "blurry, distorted", "modalities": ["image"]},
            sampling_params_list=[params],
        )
        elapsed = time.perf_counter() - start
        extract_images_from_outputs(result)[0].save(output / f"{index}.png")
        if index >= 2:
            timings.append(elapsed)
    print(mode, "mean", statistics.mean(timings), "stdev", statistics.stdev(timings))
    engine.close()
PY
```

## L20 validation

The official checkpoint at revision `41be1c5821a92c970d63d7eb595a2fd3fe32b22e`
completed the comparison above on one L20 and on two L20s with CFG parallelism.
The rank-0 snapshot verified 196 FP8 decoder projections and BF16 DiT/VAE
parameters. Memory figures below report rank 0, not the maximum across ranks.

| Configuration | BF16 rank-0 peak allocated | FP8 rank-0 peak allocated |
| --- | --- | --- |
| 1×L20 | 18.008 GiB | 16.690 GiB |
| 2×L20, CFG=2 | 17.999 GiB | 16.680 GiB |

Three prompt pairs retained the requested objects, counts and OPEN text; paired
image SSIM was 0.9667, 0.6998 and 0.9380. The storefront sign layout changed.
Single-card and CFG-parallel PNG files matched exactly within each precision mode.
These samples establish smoke coverage, not general quality equivalence.

The text encoder was slower with FP8 at this request size. Shared GPU workloads
caused substantial E2E timing variation, so these runs establish memory savings
and functional coverage without a reliable latency speedup claim.

## RTX 5090 online FP8 CUDA hot path E2E

The official checkpoint at revision `41be1c5821a92c970d63d7eb595a2fd3fe32b22e`
was verified against the 19 required files' LFS SHA256 values. On one RTX 5090,
the same checkpoint ran in A/P/P/A order with the CUDA activation quantizer
disabled (A) or enabled (P). Both modes used online FP8 for the Qwen3 text
encoder; the diffusion transformer and VAE remained at their original precision.
Each arm used three warmups and six timed full image requests, cycling three
prompts at 768×512, 30 denoising steps, guidance scale 4, and seed 142.
Execution was eager, with one GPU and no CFG parallelism. Model loading was
excluded from the request timings.

| Arm | Activation quantizer | Full request mean ± SD | Speedup vs pooled A | Peak allocated |
| --- | --- | ---: | ---: | ---: |
| A1 | reference | 6.448 ± 0.046 s | 1.000× | 16.77 GiB |
| P1 | sm_80+ CUDA | 6.434 ± 0.082 s | 1.002× | 16.77 GiB |
| P2 | sm_80+ CUDA | 6.479 ± 0.011 s | 0.995× | 16.77 GiB |
| A2 | reference | 6.451 ± 0.026 s | 1.000× | 16.77 GiB |

The pooled full-request speedup was **0.999×**. A2/A1 timing drift was 1.000×
and P2/P1 was 1.007×. The text encoder contained 196 online FP8 linear
modules. Instrumentation recorded zero CUDA activation quantizer calls per A
request and 392 per P request. All 12 paired A/P RGB images matched pixel for
pixel. These measurements establish correct CUDA path execution and output
equivalence at this workload size, with no measurable full-request speedup.

With the existing regional `torch.compile` path enabled (`enforce_eager=False`),
the same GPU, checkpoint, prompts and A/P/P/A protocol produced:

| Arm | Activation quantizer | Full request mean ± SD | Speedup vs pooled A | Peak allocated |
| --- | --- | ---: | ---: | ---: |
| A1 | reference | 6.178 ± 0.073 s | 1.004× | 16.77 GiB |
| P1 | sm_80+ CUDA | 6.194 ± 0.054 s | 1.001× | 16.77 GiB |
| P2 | sm_80+ CUDA | 6.185 ± 0.076 s | 1.003× | 16.77 GiB |
| A2 | reference | 6.226 ± 0.067 s | 0.996× | 16.77 GiB |

The compiled run's pooled CUDA-path speedup was **1.002×**. Its A2/A1 drift
was 1.008×; all 12 paired A/P images remained pixel-identical. Relative to the
separate eager run above, regional compilation reduced pooled full-request time
from about 6.45 s to 6.20 s (approximately **1.04×**); this comparison was not
interleaved across compilation modes. Compiled and eager images are not pixel
identical to each other, so compare outputs within the same execution mode.
The CUDA activation calls observed in a shape audit were BF16 tensors of
284×2048 and 284×6144; every observed call satisfied the CUDA fast-path gate.

The run used PyTorch `2.13.0+cu130`, vLLM `0.29.0`, vLLM-Omni `0.28.0`,
Transformers `5.10.4`, and Diffusers `0.40.0`. Its runtime source overlay used
the existing CUDA platform compatibility adjustment from #27 for the installed
vLLM `IrOpPriorityConfig`; the same overlay was used in all four arms and is
not part of this model diff.
