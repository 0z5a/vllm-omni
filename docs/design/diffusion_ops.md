<!-- markdownlint-disable MD013 -->

# RFC: Establish an operator boundary for vLLM-Omni diffusion

> **Status:** Draft for discussion
>
> **vLLM-Omni snapshot:** [`92534733`](https://github.com/vllm-project/vllm-omni/commit/925347332f8190d172dfa8ac0c307a5cfa2320cb)
>
> **Upstream vLLM snapshot:** [`fa2a2589`](https://github.com/vllm-project/vllm/commit/fa2a2589bd2a1fce0851df7fd42ffb54b6195f04)

## Summary

vLLM has an operator boundary, most visibly through `_custom_ops.py` and its
kernel-provider interfaces. Model code can use optimized kernels without
depending on native extension names, packed layouts, allocation details, or
platform-specific dispatch.

vLLM-Omni already reuses much of that machinery. However, diffusion support has
also accumulated local Triton kernels, `torch.library` custom ops, platform
calls, third-party providers, and compatibility patches. These cover semantics
that are sometimes different from autoregressive inference:

- N-dimensional and non-contiguous activations;
- routed mixture-of-transformers (MoT) computation;
- diffusion-specific attention and quantized attention;
- platform-specific microscaling formats; and
- stateful diffusion cache operations.

This RFC asks whether vLLM-Omni should define an explicit diffusion operator
boundary. “Independent” means independently defined semantics, not independent
reimplementation: generic kernels should continue to come from upstream vLLM.

Approval would only establish ownership principles. It would not approve a
refactor, public API, file layout, or migration plan.

## Motivation

### Diffusion needs a stronger tensor contract

Diffusion linears commonly receive tensors shaped `(..., K)` and must return
`(..., N)`, preserving all leading dimensions. Inputs may also be
non-contiguous.

The current code adapts upstream FP8 paths by:

- making ScaledMM activations contiguous before a two-dimensional `view`; and
- restoring leading dimensions when the FlashInfer provider returns a
  two-dimensional result.

Several quantization methods independently perform similar flatten/restore
logic. These are provider-contract requirements, not new GEMM algorithms.

### Some operations are genuinely diffusion-specific

Examples include:

- MoT kernels that route text and VAE tokens to different weights inside one
  fused GEMM;
- diffusion attention with its own layouts, masks, parallel strategies, and
  optional quantization;
- NPU FP8 attention that rotates and block-quantizes Q/K/V on each denoising
  step; and
- experimental paged attention that writes into a process-owned diffusion KV
  pool.

These semantics do not map cleanly to generic dense-linear or autoregressive
attention interfaces.

### Quantization mixes policy, checkpoint handling, and kernels

Current quantization code combines:

1. deciding which stages and layers to quantize;
2. loading, remapping, and packing weights and scales; and
3. selecting and invoking a hardware kernel.

The first two are often Omni-specific. The third is frequently generic and
already supported upstream. Without a clear boundary, it is difficult to tell
when Omni is extending vLLM and when it is duplicating it.

## Current status

The audited snapshot has no project-owned C++, CUDA, or HIP extension sources.
Its optimized operator surface is Python- and provider-driven:

| Mechanism | Current status |
| --- | --- |
| Upstream vLLM ops/providers | Default path for generic quantization, GEMM, normalization, and related primitives |
| Diffusion `CustomOp` | Construction-time platform dispatch for norm, RoPE, AdaLayerNorm, and MoT RMSNorm |
| Attention registry | Nine default diffusion attention backends with platform overrides |
| Local Triton | Seven decorated functions, primarily MoT plus one model-specific fusion |
| `torch.library` | Three custom ops: SageAttention 3, ROCm MXFP4, and experimental paged write-attention |
| External providers | Direct use of `torch_npu`, AITER, FlashInfer, SageAttention, BitsAndBytes, and Quack |
| Compatibility patches | FP8 contiguity/output shape, NVFP4 scale sanitization, Quack selection, and scoped provider forcing |

### Quantization snapshot

| Path | Kernel source | Omni-specific behavior |
| --- | --- | --- |
| FP8 / ModelOpt / NVFP4 | Upstream vLLM | Checkpoint adaptation, diffusion shape fixes, HSDP weight views, scale sanitization |
| CUDA INT8 | Upstream vLLM | Configuration and online/offline weight handling |
| NPU INT8 and MXFP8 | `torch_npu` | Layouts, scales, bias/dtype handling, and shape restoration |
| XPU MXFP8 | Upstream vLLM | Local N-dimensional wrapper |
| NPU MXFP4 | `torch_npu` | Single-scale and diffusion-oriented DualScale formats |
| ROCm MXFP4 | Local AITER custom op | Overlaps with an AITER provider now present upstream |
| BitsAndBytes W4 | BitsAndBytes | Load-time quantization and shape adaptation |
| Quack FP8 | Quack CuteDSL | Installed by replacing the selected FlashInfer method |
| MoT FP8/INT8 | Local Triton plus upstream FP8 activation quantization | Routed text/VAE semantics |

Upstream now has provider registries for FP8, INT8, NVFP4, MXFP8, and MXFP4
across supported platforms, plus `register_linear_kernel()`. This should remain
the preferred home for model-agnostic linear kernels.

### Problems exposed by the current structure

- Tensor shape, contiguity, dtype, bias, packing, and scale contracts are not
  expressed uniformly across providers.
- Global patches solve real problems but make provider ownership, ordering, and
  version compatibility difficult to inspect.
- FakeTensor, `torch.compile`, fallback, mutation, and aliasing behavior varies
  by integration.
- Platform capability information is duplicated across code and documentation
  and has already drifted.
- Tests cover configuration and weight layouts well, but quantized MoT, Quack
  execution, real ROCm MXFP4, and cross-provider compile behavior remain thin.

## Direction proposed for discussion

```text
diffusion model or layer
        |
        v
diffusion semantic contract, only where upstream is insufficient
        |
        +----> upstream vLLM provider
        +----> Omni-owned diffusion provider
        +----> platform or out-of-tree provider
```

Suggested principles:

1. **Reuse upstream by default.** Do not wrap or fork a generic kernel whose
   upstream contract already works for diffusion.
2. **Fix generic contracts upstream.** Shape preservation, non-contiguous input
   handling, generic packing, and provider registration should live in vLLM
   when they benefit more than diffusion.
3. **Own diffusion semantics in Omni.** MoT routing, quantized diffusion
   attention, and diffusion cache behavior are reasonable Omni responsibilities.
4. **Separate policy from execution.** Stage selection, ignored layers,
   checkpoint adaptation, and quality-sensitive BF16 fallback are not kernel
   responsibilities.
5. **Treat hardware integrations as providers.** Model code should not depend
   directly on vendor ABI, packed layouts, or optional package internals.
6. **Make compilation and state explicit.** Opaque kernels need deliberate fake
   behavior, mutation/aliasing semantics, and fallback rules.
7. **Prefer scoped selection over global replacement.** Active providers and
   fallbacks should be inspectable and limited to the intended operation or
   stage.
8. **Do not require one monolithic file.** The boundary could be a small package
   or extensions to existing registries; copying upstream `_custom_ops.py` is
   not the goal.

Likely upstream-owned:

- generic dense-linear quantization kernels and activation quantization;
- provider-independent shape/dtype contracts; and
- generic provider registration and selection.

Likely Omni-owned:

- routed MoT operators;
- dynamically quantized diffusion attention;
- diffusion-specific cache operations; and
- formats or semantics not represented upstream, such as the current NPU
  MXFP4 DualScale path.

## Non-goals

- Defining a detailed refactor or migration sequence.
- Choosing final module names or public Python signatures.
- Reimplementing vLLM's generic custom ops or linear kernels.
- Adding an Omni native extension loader without a demonstrated need.
- Changing supported hardware or claiming unvalidated quantization support.
- Removing compatibility patches before replacement behavior is validated.

## Open questions

1. Is an explicit diffusion operator boundary needed, or are the current layer
   and backend interfaces sufficient?
2. Should the boundary remain diffusion-only or eventually cover other
   non-autoregressive Omni stages?
3. What minimum shape, layout, dtype, bias, and compilation contract should
   every provider satisfy?
4. Which current adaptations are generic upstream fixes versus
   diffusion-specific behavior?
5. Should quantized diffusion attention be an attention backend, an operator
   provider, or both?
6. Should platform providers live in Omni core, platform projects, or
   out-of-tree packages?
7. Is upstream `register_linear_kernel()` sufficient for provider priority,
   diagnostics, explicit selection, and fallback?
8. How should stateful operators declare process-global mutation to eager mode
   and `torch.compile`?
9. What validation should be required before an experimental provider becomes
   a supported default?

## Feedback requested

Feedback is requested from upstream kernel, diffusion, quantization, platform,
attention, and `torch.compile` maintainers. Particularly useful comments would
identify:

- an operation that clearly belongs upstream or clearly belongs in Omni;
- a missing provider contract;
- a compatibility patch that should remain local or move upstream; or
- a platform integration that needs an out-of-tree extension point.

If there is agreement on the ownership principles, a follow-up design can
compare concrete API options. If not, this RFC should document why the existing
abstractions are sufficient.

## References

- [vLLM `_custom_ops.py`](https://github.com/vllm-project/vllm/blob/fa2a2589bd2a1fce0851df7fd42ffb54b6195f04/vllm/_custom_ops.py)
- [vLLM linear-kernel registry](https://github.com/vllm-project/vllm/blob/fa2a2589bd2a1fce0851df7fd42ffb54b6195f04/vllm/model_executor/kernels/linear/__init__.py)
- [vLLM-Omni quantization factory](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/quantization/factory.py)
- [Diffusion attention registry](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/diffusion/attention/backends/registry.py)
- [Diffusion `CustomOp`](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/diffusion/layers/custom_op.py)
- [Compatibility patches](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/patch.py)
- [MoT GEMM](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/diffusion/layers/mot/ops/mot_gemm.py)
- [NPU FP8 diffusion attention](https://github.com/vllm-project/vllm-omni/blob/925347332f8190d172dfa8ac0c307a5cfa2320cb/vllm_omni/platforms/npu/quant/kv_quant_npu.py)
