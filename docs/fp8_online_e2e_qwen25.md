# Qwen2.5-Omni online FP8 CUDA E2E

Full text-to-speech requests were measured with the three-stage
`Qwen/Qwen2.5-Omni-3B` checkpoint (ModelScope revision
`f0fe4326288865534ad09d70aa435ff7d1611541`) on three RTX 5090 GPUs
(SM120). The Thinker used dynamic online FP8 weights on MLP gate/up projections;
Talker and Code2Wav ran in their original precision. Both arms used the same
FP8 weights. `VLLM_OMNI_FP8_ONLINE_DISABLE=1` selected vLLM's current CUDA
activation quantizer (A), and `0` selected the sm_80+ CUDA hot path (P).

Each arm started in a fresh process, ran three warmup requests and six measured
requests, and used the same 1,978-token text prompt, seed 42, and full audio
generation. A/P/P/A order reduced the effect of time drift. `enforce_eager=True`
was necessary to exercise `QuantFP8.forward_cuda`: with the installed vLLM
0.29.0, `torch.compile` decomposes `QuantFP8.forward_native` and bypasses this
CUDA dispatch hook. Torch was 2.13.0+cu130.

| Arm | CUDA activation quant | Full E2E mean ± SD (s) | Thinker completion mean ± SD (s) | Full speedup vs pooled A |
|---|---|---:|---:|---:|
| A1 | current CUDA | 13.059 ± 0.231 | 1.079 ± 0.162 | 0.995× |
| P1 | sm_80+ hot path | 12.488 ± 0.351 | 0.924 ± 0.022 | 1.040× |
| P2 | sm_80+ hot path | 12.531 ± 0.127 | 0.982 ± 0.022 | 1.037× |
| A2 | current CUDA | 12.927 ± 0.483 | 1.044 ± 0.149 | 1.005× |

The pooled full-request speedup was **1.039×**; Thinker completion was
**1.113×** faster. A2/A1 latency drift was 0.990×, and P2/P1 drift was 1.004×.
All 24 measured A/P/P/A requests returned text and a finite 10.24-second audio tensor.

Output consistency requires care: A2, P1, and P2 produced the same 27-token
text and exact audio for each paired request. A1 produced a 28-token variant,
which a separate A3 current-CUDA run also reproduced. Thus only 6/12 A/P pairs
had exact text and audio. A separate hot-path diagnostic run also produced the
28-token variant, showing that both paths can reach both outputs. In that full
three-stage diagnostic request, the CUDA extension and reference quantizer
returned bit-identical FP8 activations and scales for the observed BF16 shapes
`(16, 2048)` and `(1, 2048)`. The alternate text branch prevents attributing
the audio difference to the vocoder or quantizer alone.
