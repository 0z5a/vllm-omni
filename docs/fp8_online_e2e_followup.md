# Online FP8 CUDA hot-path follow-up

## Selecting an implementation

The custom CUDA extension is **opt-in**, because the full-model measurements
show workload-dependent regressions. Set flags before starting the process:

- `VLLM_OMNI_FP8_ONLINE_ENABLE=1`: enable the experimental CUDA route for
  validated eager workloads. The kernel keeps the original 512-row gate.
- `VLLM_OMNI_FP8_ONLINE_DISABLE=1`: retain the original vLLM activation
  quantizer. This takes precedence over ENABLE.
- `VLLM_OMNI_FP8_FORCE_TORCH=1`: explicitly use `QuantFP8.forward_native` for
  intercepted CUDA calls; this takes precedence over both flags. SeedVR2 uses
  that native implementation for activations when explicitly requested.

`forward_native` itself is never replaced. Compiled calls bypass the custom
extension and use the original vLLM implementation. Static/group quantization,
unsupported inputs, and unavailable extension builds retain the reference path.
The reference CUDA kernel and Torch FP8 casting differ in midpoint rounding;
selecting Torch preserves the native option, not bitwise equivalence to CUDA.
BF16 without online quantization remains available through model configuration.

## Measured full-model results

These are existing RTX 5090 ABBA results from the fork experiment branches,
not fresh benchmarks of the upstream ports. Each compares fixed online-FP8
weights with reference CUDA versus the experimental route, except Anima, whose
candidate combines CFG graphs and caches. The new opt-in and fallback guards
were CPU-tested; model weights were not downloaded again.

| Model | Scope | Speedup | Correctness and limitation |
| --- | --- | ---: | --- |
| Anima | 1024 square, 50 steps, CFG graph and caches | 1.0565× | 12/12 identical images; see upstream #8052 |
| Qwen2.5-Omni-3B | Full text-to-speech, eager | 1.039× | Thinker 1.113×; 6/12 exact text/audio pairs, both routes produced 27/28-token variants; observed quantizer payload and scale bit-exact |
| OmniGen2 | 512 square, 28 steps, layer offload, five warmups | 1.031× | 12/12 exact images; earlier one-warmup runs unstable |
| SeedVR2 | 5 frames, 192×112, HTTP video/audio | 1.045× | 28/28 decoded videos exact including warmups; short-clip result only |
| SeedVR2 | 24 frames, 768×1344 | 0.963× | 16/16 decoded videos exact including warmups; slower, use original route |

Anima's calculated composition is `1.125 × 1.0565 = 1.1885625×`, approximately
**1.189× / 15.9% lower latency**. This assumes transfer between the earlier
20-step BF16/FP8 median comparison and the 50-step graph/cache pooled-mean
comparison. It is not directly measured total speedup. Qwen's compiled FP8 gain
cannot be presented as a combined working mode with the eager-only CUDA hook.
No reliable BF16-relative speed ratio was established for the other candidates
that could support a combined performance claim.

The unchanged kernel sources were previously tested against reference CUDA
payload bytes and scales on sm_120. Other listed architectures have compile
coverage only. The unsuccessful 1108/2048/4096-row candidates are excluded.
LongCat, Helios, LTX-2, HunyuanVideo-1.5, ERNIE-Image, Ovis, Kontext, Wan2.2,
and Klein have no accepted additional CUDA hot-path gain in this campaign.

## Evidence

- [Qwen fork #25](https://github.com/0z5a/vllm-omni/pull/25)
- [OmniGen2 fork #26](https://github.com/0z5a/vllm-omni/pull/26)
- [SeedVR2 fork #30](https://github.com/0z5a/vllm-omni/pull/30)
- [Anima fork #32](https://github.com/0z5a/vllm-omni/pull/32)
