# LTX-2 Gemma3 text encoder FP8 (experimental)

Enable online FP8 explicitly for the Gemma3 text encoder:

```python
from vllm_omni import Omni

omni = Omni(
    model="Lightricks/LTX-2",
    dtype="bfloat16",
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

This converts only the Gemma3 language-model decoder's attention and MLP
projections. Embeddings, norms, the language-model output head, vision tower,
connectors, DiT and VAEs retain their original precision. The diffusion loader
finalizes FP8 weights through its existing post-load path.

The initial implementation accepts BF16/FP16 source weights and dynamic online
FP8 only. Global DiT quantization does not opt the encoder in. Gemma4-based
LTX-2.5 models reject this explicit encoder setting.

Tiny-Gemma3 probes cover stacked hidden-state outputs consumed by LTX-2,
not official-model video/audio quality. Full-model quality, memory, encoder TP,
offload combinations and steady-state latency remain to be validated.
