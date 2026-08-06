# Production-grade NVIDIA Blackwell support in vLLM-Omni

> **Status:** Draft for discussion
>
> **Proposed issue title:** `[RFC]: Production-grade NVIDIA Blackwell support in vLLM-Omni`
>
> **vLLM-Omni snapshot:** [`273fc8eb`](https://github.com/vllm-project/vllm-omni/commit/273fc8eb1de084e20a23d03586dec33ac1cbe5ad)
>
> **vLLM baseline:** `v0.26.0`
>
> **Initial scope:** datacenter Blackwell `sm_100` / `sm_103` and
> workstation or consumer Blackwell `sm_120` / `sm_121`

## Summary

vLLM-Omni already has substantial Blackwell enablement:

- the vLLM `v0.26.0` CUDA 13 runtime provides the autoregressive CUDA
  foundation;
- [diffusion attention recognizes `sm_100`, `sm_103`, `sm_120`, and
  `sm_121`](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/vllm_omni/platforms/cuda/platform.py#L64-L89);
- datacenter Blackwell can use TRTLLM-generated diffusion attention when its
  shape and mask constraints are satisfied;
- cuDNN and FlashInfer provide safe dense diffusion-attention paths for both
  Blackwell families; and
- FP8, INT8, ModelOpt, NVFP4, SageAttention 3, and Quack integrations cover
  selected models and hardware.

The remaining problem is not initial bring-up. It is the lack of a precise,
continuously validated support contract. An importable dependency can still
lack a Blackwell kernel, current CUDA CI has no Blackwell resource, and several
quantized paths are documented more broadly than they are exercised on native
Blackwell hardware.

This RFC proposes:

1. architecture-aware kernel capability checks with safe, explicit fallbacks;
2. separate validation for datacenter `sm_10x` and workstation `sm_12x`;
3. a small mandatory correctness matrix for attention, sequence parallelism,
   representative Omni workloads, and documented quantization paths;
4. a reproducible dependency and release qualification contract; and
5. a distinction between **baseline supported**, **optimized**, and
   **experimental** features.

Optimized-kernel parity between `sm_10x` and `sm_12x` is not required. The two
families expose different hardware features, and a correct cuDNN, FlashInfer, or
SDPA fallback is acceptable for baseline support.

## Motivation.

### “Blackwell supported” is currently ambiguous

The current platform selector contains explicit Blackwell routing and the user
guide publishes
[`sm_120` end-to-end measurements](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/docs/user_guide/diffusion/attention_backends.md#L170-L181).
This is strong manual evidence that common BF16 diffusion workloads run.

However, a broad support statement could mean any of the following:

- the package imports and one model starts;
- common BF16 workloads run on one Blackwell SKU;
- every documented backend fails safely or executes correctly;
- multi-GPU sequence parallelism works;
- documented FP8 or NVFP4 checkpoints use native Blackwell kernels; or
- every Blackwell family has continuous correctness, accuracy, and performance
  validation.

This RFC defines a support claim narrowly enough to be testable.

### A current failure demonstrates the binary-capability gap

[`fa3-fwd==0.0.3`](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/requirements/cuda.txt#L1-L3)
imports on Blackwell, but its extension contains only SM80 and SM90a cubins.
Ring attention treats importability as availability and can therefore launch a
kernel that does not exist for the current device.

This is reported in
[#5611](https://github.com/vllm-project/vllm-omni/issues/5611).
[#5617](https://github.com/vllm-project/vllm-omni/pull/5617) adds an
architecture-aware SDPA fallback and reports real two-rank validation on
B300/SM103. The defect is narrow, but the lesson is general: importing a Python
module does not prove that an optional extension contains a compatible device
kernel.

### Current CI cannot continuously validate Blackwell behavior

CUDA
[test resources currently accept only L4 and H100](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/tests/helpers/mark.py#L11-L18).
The
[Buildkite hardware presets](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/.buildkite/common/ci_mirror_hardwares.yml#L20-L190)
likewise have no Blackwell entry on `main`.
[#5543](https://github.com/vllm-project/vllm-omni/pull/5543) is adding B200
hardware selection, but it is still under review and does not establish an
`sm_120` lane or an architecture-specific pytest contract.

The
[Qwen3-Omni NVFP4 regression test](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/tests/e2e/offline_inference/test_qwen3_omni_modelopt_nvfp4_w4a4.py#L14-L18)
explicitly states that its H100 Marlin fallback does not exercise the native
Blackwell FP4 path and leaves a `TODO(B200)` for that coverage.

### Datacenter and workstation Blackwell are not interchangeable

`sm_100` / `sm_103` and `sm_120` / `sm_121` share the Blackwell generation but
do not expose identical kernels:

- TRTLLM-generated diffusion attention is intentionally limited to
  datacenter `sm_10x`;
- Quack's `tcgen05` FP8 path is intentionally enabled only on datacenter
  `sm_10x`;
- workstation `sm_12x` uses cuDNN or FlashInfer for those workloads; and
- optional FlashInfer, SageAttention, FA4, and low-precision features have
  provider-specific architecture matrices.

Testing B200 alone is therefore insufficient evidence for `sm_120`, while lack
of one optimized `sm_10x` kernel on `sm_120` is not itself a correctness bug.

## Scope and ownership

### Initial architecture scope

| Family | Representative validation target | Initial support intent |
| --- | --- | --- |
| `sm_100` | B200 or GB200 | Continuously validated |
| `sm_103` | B300 or GB300 | Release or periodic compatibility validation |
| `sm_120` | RTX PRO 6000 or equivalent | Continuously validated |
| `sm_121` | Available `sm_121` runner | Compatibility validation when hardware is available |

This RFC does not silently extend the support claim to `sm_101`, `sm_110`,
future Blackwell variants, Windows, or Jetson deployments. They may reuse the
same capability framework, but each needs an explicit owner and validation
target before being listed as supported.

### Upstream vLLM versus vLLM-Omni ownership

| Area | Primary owner | RFC requirement |
| --- | --- | --- |
| Autoregressive LLM attention, CUDA compilation, generic CUDA kernels | vLLM | Consume a Blackwell-capable vLLM release and run a representative Omni smoke test |
| Diffusion attention routing and fallbacks | vLLM-Omni | Validate every default and explicit Blackwell route |
| Ring/Ulysses diffusion sequence parallelism | vLLM-Omni | Provide correct multi-GPU behavior or an actionable early rejection |
| Model-specific audio, vision, talker, tokenizer, and diffusion stages | vLLM-Omni | Validate representative end-to-end paths |
| Generic quantized linear providers | Prefer upstream vLLM | Reuse upstream providers and validate Omni checkpoint/model adaptation |
| Optional provider binaries such as FlashInfer, FA, SageAttention, and Quack | Shared | Declare and test architecture capability before dispatch |

Generic provider or CUDA defects should be fixed upstream when possible.
vLLM-Omni remains responsible for safe selection, model adaptation, and the
fallback behavior exposed to its users.

## Goals

1. Define what vLLM-Omni means when it lists a Blackwell architecture as
   supported.
2. Prevent an importable but architecture-incompatible extension from reaching
   a CUDA launch.
3. Continuously test representative `sm_100` and `sm_120` behavior.
4. Restore correct ring sequence parallelism on Blackwell, even if the initial
   implementation uses SDPA rather than an optimized native kernel.
5. Validate the native Blackwell execution path for every quantization method
   documented as supported on Blackwell.
6. Split hardware documentation by `sm_10x` and `sm_12x` where behavior differs.
7. Record enough environment and dispatch information to reproduce a
   Blackwell-only failure.

## Non-goals

- Requiring identical performance or kernel selection on `sm_10x` and
  `sm_12x`.
- Making FA4, quantized diffusion attention, SageAttention 3, or Skip-Softmax a
  prerequisite for baseline BF16 support.
- Implementing CUDA-native diffusion MXFP8 or MXFP4 in the first phase.
- Reimplementing generic vLLM, cuDNN, or FlashInfer kernels in vLLM-Omni.
- Validating every model and every quantization combination on every
  Blackwell SKU.
- Treating manual benchmark results as a replacement for recurring CI.
- Expanding the initial claim to unlisted Blackwell variants without hardware
  validation.

## Support contract

### Support levels

**Baseline supported**

- the documented installation resolves to a compatible vLLM, PyTorch, CUDA,
  cuDNN, and provider set;
- the runtime selects only a kernel known to support the current architecture;
- an unsupported explicit backend fails during configuration or warmup with an
  actionable error;
- a documented safe fallback produces correct output;
- representative eager and compiled paths pass recurring hardware CI; and
- applicable multi-GPU and documented quantization smokes pass.

**Optimized**

- the architecture uses an intentional accelerated provider rather than the
  conservative fallback;
- accuracy is compared with the architecture-local BF16 baseline; and
- performance is monitored against a per-SKU baseline.

**Experimental**

- the feature is opt-in;
- its architecture or model matrix may be incomplete; and
- it is not part of the release-blocking baseline.

### Proposed compatibility matrix

| Capability | `sm_100` / `sm_103` | `sm_120` / `sm_121` | Required RFC exit state |
| --- | --- | --- | --- |
| AR/LLM CUDA path | Inherited from vLLM; not directly covered by current Blackwell CI | Same | Representative Omni AR-stage smoke on each continuously validated family |
| Dense BF16 diffusion attention | TRTLLM when compatible; otherwise cuDNN/FlashInfer/SDPA | cuDNN/FlashInfer/SDPA | Default route and fallback tested on real hardware |
| Ring sequence parallelism | B200 failure reported in #5611 | Same import-only selection risk is unverified | Correct SDPA fallback plus real two-rank Blackwell test |
| Ulysses sequence parallelism | Expected to use the selected local backend | Same | Existing multi-GPU smoke rerun on B200; failures must identify unsupported shapes |
| TRTLLM diffusion attention | Supported only on datacenter `sm_10x` with provider, shape, and mask constraints | Unsupported by design | Positive `sm_10x` test and early `sm_12x` rejection test |
| FP8 and INT8 linear paths | Documented support; native path not broadly covered | Documented support; provider differs | One representative diffusion checkpoint per family |
| ModelOpt NVFP4 | Documented Blackwell path; native regression coverage incomplete | Broad Blackwell claim is not separately validated on `sm_12x` | Native `sm_10x` smoke; validate `sm_12x` separately or narrow its documentation |
| Diffusion-native MXFP8 / MXFP4 | Unsupported on CUDA today | Unsupported on CUDA today | Documentation says unsupported; configuration fails early |
| Quack FP8 | Optimized on datacenter `sm_10x` | Unsupported by hardware; FlashInfer fallback | Positive `sm_10x` and negative `sm_12x` selection tests |
| SageAttention 3, FA4, and quantized attention | Experimental or in flight | Experimental or provider-dependent | Not release-blocking until promoted through a follow-up RFC or support update |

## Proposed Change.

### 1. Centralize architecture and provider capability checks

Introduce one internal CUDA architecture classification used by platform
routing, attention backends, ring attention, quantization providers, and tests.
It should distinguish at least:

- datacenter Blackwell `sm_10x`;
- workstation or consumer Blackwell `sm_12x`;
- explicitly recognized minor architectures; and
- unknown or unvalidated variants.

Every optional provider selected by vLLM-Omni should expose or be wrapped by a
capability predicate that answers whether it can execute the requested
operation on the actual device. The predicate may use:

- provider-published architecture metadata;
- a known contract for an exactly pinned provider version;
- an upstream vLLM provider capability API; or
- a bounded startup probe when metadata is unavailable and probing cannot
  terminate the process.

Importability alone is insufficient.

Selection must have one of three observable outcomes:

1. select a compatible provider;
2. select a documented safe fallback and log the reason once; or
3. reject an explicitly requested unsupported provider before the first real
   request.

The runtime log should identify the device capability, selected provider, and
fallback reason. It must not claim that a fallback provider is the optimized
path.

The first implementation should:

- land the architecture-aware ring fallback from #5617;
- audit `vllm_omni.diffusion.attention.backends.utils.fa`;
- audit direct `flash_attn`, `fa3_fwd_interface`, and
  `flash_attn_interface` imports in model-specific code; and
- make the FA3 Hub and explicit FlashAttention gates match their provider
  contracts.

### 2. Add architecture-specific CI resources

Extend the CUDA resource helpers and Buildkite presets with architecture-honest
names. A test running on B200 must not continue to advertise itself as an H100
test merely to reuse an existing marker.

Recommended representative resources:

- `B200`, with one-, two-, and four-GPU presets as infrastructure permits; and
- `RTXPRO6000` or another stable `sm_120` product name, initially single-GPU.

The `MIRROR_HW` mechanism proposed by #5543 can choose the physical preset, but
pytest selection should still express the architecture actually required by a
test.

#### CI levels

| Level | Hardware | Required coverage | Blocking policy |
| --- | --- | --- | --- |
| L1/L2 | CPU plus existing CUDA | Architecture classifier, provider-capability, routing, early-error, and fallback unit tests | Required on affected PRs |
| L3 | One or two B200 GPUs, change-selected | Startup/kernel smoke; two-rank ring when relevant; native quantization smoke when relevant | Required pre-merge for Blackwell-sensitive PRs through the existing `merge-test` trigger; main-branch runs provide detection and recovery |
| L4 | B200 and `sm_120` representative | End-to-end BF16, compiled path, two-rank B200 ring, representative Omni AR stage, and documented FP8/NVFP4 paths—or an explicitly narrowed support status where qualification is incomplete | Nightly release signal |
| L5/release | B300/SM103 and other available variants | Compatibility smoke, extended models, accuracy, and performance | Required before expanding or retaining the published variant matrix |

Infrastructure failures should be reported separately from product failures.
An architecture should not be promoted to baseline supported until its required
jobs complete seven consecutive scheduled runs without a product failure.

Blackwell-sensitive source paths should automatically request, or be required
by project policy to use, the existing `merge-test` trigger. A label that can be
silently omitted is not a merge gate. Required Blackwell jobs must execute with
zero unexpected skips or xfails; a green job in which the architecture-specific
test never ran is not evidence of support.

### 3. Use a small representative model matrix

The recurring matrix should cover behavior classes rather than every model:

- one mask-free video DiT that can exercise the datacenter TRTLLM route, such
  as Wan2.2;
- one mask-using image or video DiT that exercises the cuDNN path, such as
  Qwen-Image or HunyuanVideo-1.5;
- one multi-stage or Omni workload that reaches an upstream vLLM AR stage and
  at least one Omni-owned non-AR component;
- one audio or TTS integration canary that exercises a model-specific encoder
  or attention adapter;
- one native Blackwell NVFP4 checkpoint smoke.

Tiny or random-weight models should cover routing, shapes, compilation, and
collective behavior in lower levels. Full checkpoints should be reserved for
nightly accuracy and integration validation.

Large Omni checkpoints do not need to fit on a single workstation GPU to call
the `sm_120` family supported. The `sm_120` lane must instead validate
representative components and at least one end-to-end workload that fits the
target resource, while documenting memory-driven model exclusions.

### 4. Make quantization claims path-specific

Quantization documentation and tests should distinguish:

- generic upstream vLLM FP8, INT8, and NVFP4 providers;
- ModelOpt checkpoint adaptation in vLLM-Omni;
- online versus pre-quantized checkpoints;
- diffusion versus AR/thinker stages; and
- datacenter versus workstation provider selection.

Each Blackwell-supported quantization row needs:

1. proof that the intended quantized provider is selected;
2. evidence that weights remain in the intended representation or reduce
   memory as expected;
3. a finite and non-degenerate output assertion; and
4. an accuracy comparison with a same-architecture BF16 baseline.

The open report in
[#4202](https://github.com/vllm-project/vllm-omni/issues/4202) should be
reproduced or closed against the RFC baseline. Because that report used
vLLM-Omni `0.20.0`, it is evidence requiring current reproduction, not proof
that `0.26.0` is still broken.

CUDA-native diffusion MXFP8 and MXFP4 should be marked unsupported until a
separate implementation and validation effort lands. “Not verified” is too
ambiguous when
[MXFP8](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/vllm_omni/quantization/mxfp8_config.py#L120-L147)
and
[MXFP4](https://github.com/vllm-project/vllm-omni/blob/273fc8eb1de084e20a23d03586dec33ac1cbe5ad/vllm_omni/quantization/mxfp4_config.py#L142-L168)
explicitly raise `NotImplementedError` on CUDA.

### 5. Qualify the dependency and release environment

vLLM-Omni owns no in-tree CUDA or C++ extension for these paths; its Blackwell
binary support is supplied by vLLM, PyTorch, cuDNN, FlashInfer, FA packages,
SageAttention, Quack, and other optional providers.

Release qualification should therefore record:

- GPU product and compute capability;
- driver, CUDA runtime, PyTorch, and cuDNN versions;
- vLLM and vLLM-Omni versions;
- optional provider package versions;
- provider-reported or known compiled architectures; and
- the selected attention and quantized-linear providers.

The supported installation path must resolve to a tested combination. Possible
implementations include an exact vLLM dependency, a generated compatibility
constraint, or a startup error for an unsupported major/minor combination.
The current warning-only behavior is insufficient for an ABI-critical mismatch.

CUDA 13 driver requirements and the tested container tag should be stated in
the installation guide. Published containers should use an immutable base
reference or otherwise record the resolved base-image digest in release
artifacts.

### 6. Publish an architecture-specific support table

The user guide should avoid a single “Blackwell SM100+” row when provider
selection differs by family. It should list:

- supported architecture families and representative tested GPUs;
- the tested vLLM/CUDA/driver baseline;
- default and fallback attention providers;
- explicit unsupported combinations;
- quantization status by model stage and provider; and
- whether each item is baseline supported, optimized, experimental,
  unsupported, or unverified.

Release notes should link the exact CI or validation artifact used to retain a
support claim.

## Implementation phases and exit criteria

### Phase 0: Correctness floor

- [ ] Merge an architecture-aware ring fallback for #5611.
- [ ] Add unit tests for `sm_100`, `sm_103`, `sm_120`, `sm_121`, and unknown
      architectures.
- [ ] Audit import-only FA capability checks.
- [ ] Make unsupported explicit backends fail before request execution.
- [ ] Mark CUDA diffusion MXFP8/MXFP4 unsupported in code-facing documentation.

**Exit criterion:** every known Blackwell route either selects a compatible
provider, takes a tested safe fallback, or fails early with an actionable
message.

### Phase 1: Datacenter Blackwell baseline

- [ ] Land B200 Buildkite presets and honest pytest markers.
- [ ] Add real one-GPU default-routing and two-GPU ring tests.
- [ ] Add BF16 eager and compiled diffusion smokes.
- [ ] Add one representative Omni AR-stage smoke.
- [ ] Add native FP8 and NVFP4 smokes with provider and output assertions.
- [ ] Establish per-B200 accuracy baselines rather than reusing H100 thresholds.

**Exit criterion:** required B200 jobs pass seven consecutive scheduled runs
without a product failure or unexpected architecture-specific skip.

### Phase 2: Workstation Blackwell baseline

- [ ] Add an `sm_120` hardware preset and marker.
- [ ] Validate the cuDNN default and FlashInfer/SDPA fallbacks.
- [ ] Assert that TRTLLM and Quack are not auto-selected.
- [ ] Run a fitting end-to-end workload in eager and compiled modes.
- [ ] Validate representative FP8 and NVFP4 paths or narrow their documented
      support.
- [ ] Establish per-`sm_120` accuracy baselines.

**Exit criterion:** required `sm_120` jobs pass seven consecutive scheduled
runs without a product failure or unexpected architecture-specific skip.

### Phase 3: Optimization and expanded coverage

- [ ] Add stable per-SKU performance monitoring.
- [ ] Evaluate FA4 without making it a baseline requirement.
- [ ] Evaluate Blackwell quantized diffusion attention and typed SAGE paths.
- [ ] Add B300/SM103 periodic or release validation.
- [ ] Add further models and `sm_121` when hardware and owners are available.

**Exit criterion:** individual optimized features are promoted only after they
have architecture-specific correctness, accuracy, and performance evidence.

## Correctness and testing plan

### Routing and binary compatibility

Table-driven tests should cover:

- every recognized Blackwell architecture family;
- cuDNN above and below the supported threshold;
- FlashInfer present, absent, partially installed, and missing the required
  symbol;
- an FA Python module that imports while its version contract excludes the
  current architecture;
- explicit versus automatic backend selection;
- mask-free versus mask-using models;
- supported and unsupported head dimensions; and
- datacenter-only providers on `sm_12x`.

### Multi-GPU

At least one real two-rank NCCL test must exercise:

- ring fallback selection;
- finite BF16 outputs;
- output shape and shard ownership;
- comparison with a single-GPU or non-ring reference within a documented
  tolerance; and
- eager and compiled behavior where supported.

Ulysses and hybrid Ulysses/Ring should retain their existing shape constraints
and fail clearly when an input is unsupported.

### Quantization

Quantization tests should assert provider selection and representation, not
only that a request returns successfully. Accuracy comparisons should use the
same architecture, prompt, seed, dimensions, and scheduler settings as the BF16
reference.

Architecture-specific thresholds are allowed. They must be derived from
measured numerical variation and may not be loosened merely to make a new
runner pass.

### Performance

Performance jobs should initially run in monitor-only mode until enough samples
exist to estimate variance. A later gate can use a threshold based on both a
fixed practical floor and observed noise, for example the larger of 10% or
three standard deviations.

Performance regression on an optimized provider is not automatically a
baseline-correctness failure if the documented safe fallback remains correct.
It blocks promotion or retention of the **optimized** label.

## Compatibility and rollout

No public API change is required for the initial phases. Users may observe:

- a previously crashing backend falling back to SDPA;
- an explicit unsupported backend failing earlier;
- more precise startup logs;
- CUDA MXFP8/MXFP4 documentation changing from unverified to unsupported; and
- narrower quantization claims for combinations without native hardware
  evidence.

These are intentional correctness changes. Any fallback that can materially
change latency should emit one clear warning and be included in release notes.

CI rollout should be additive. Existing H100 and L4 coverage remains necessary;
Blackwell jobs validate architecture-specific behavior and must not replace all
pre-Blackwell regression coverage.

## Dependencies and risks

| Risk | Mitigation |
| --- | --- |
| Scarce or unstable Blackwell runners | Keep lower-level routing tests hardware-free; separate infrastructure failures; use targeted L3 and recurring L4/L5 jobs |
| Third-party wheels change embedded cubins or ABI | Pin or record exact versions, declare architecture contracts, and verify provider selection at startup |
| B200 results are incorrectly generalized to `sm_120` | Maintain separate family rows, runners, and accuracy baselines |
| Full-model CI becomes too expensive | Use tiny models for routing/kernel tests and a minimal representative full-model matrix nightly |
| Cross-GPU numerical drift creates false accuracy failures | Use architecture-local baselines and thresholds derived from repeated measurements |
| Omni duplicates upstream vLLM work | Fix generic provider contracts upstream and keep only routing/model adaptation in Omni |
| Safe fallback hides a performance regression | Log selected providers, distinguish baseline from optimized support, and monitor per-SKU performance |

## Alternatives considered

### Rely only on upstream vLLM Blackwell support

Rejected. Upstream owns the AR CUDA foundation, but vLLM-Omni owns diffusion
attention routing, ring parallelism, model-specific stages, and checkpoint
adaptation. The current ring defect exists above the upstream boundary.

### Validate only B200 and call all Blackwell supported

Rejected. Datacenter `sm_10x` and workstation `sm_12x` intentionally use
different providers and tensor-core features.

### Wait for FA4 before declaring baseline support

Rejected. cuDNN, FlashInfer, and SDPA already provide viable correctness paths.
FA4 is an optimization and should be promoted independently.

### Require full kernel and model parity before any support claim

Rejected. It would make support unattainable and would ignore legitimate
hardware differences. A narrow tested baseline plus explicit experimental
features is clearer.

### Keep using manual benchmark evidence

Rejected as the only mechanism. Manual results are useful for bring-up and
optimization decisions but do not prevent regressions in routing, dependencies,
or optional binary packages.

## Open questions

1. Should `sm_103` be continuously tested, or is a B300 release qualification
   sufficient while B200 remains the recurring `sm_10x` representative?
2. Which stable `sm_120` product and runner pool can the project commit to?
3. Which mask-using diffusion and fitting Omni workloads should form the
   smallest representative matrix?
4. Should vLLM-Omni encode an exact vLLM dependency, a generated compatibility
   range, or a hard startup check?
5. Should provider capability metadata become an upstream vLLM interface?
6. Is a correct SDPA ring fallback sufficient for baseline support, with a
   native Blackwell ring kernel tracked separately?
7. Should native CUDA diffusion MXFP8/MXFP4 remain out of scope, or should their
   current configuration names be rejected earlier at CLI/config parsing?
8. What stability window and performance threshold should be required before
   promoting an experimental provider to optimized?

## Feedback Period.

Feedback is requested for two weeks after publication.

## CC List.

TBD: CUDA platform, diffusion attention, sequence parallelism, quantization,
CI infrastructure, and upstream vLLM kernel maintainers.

## Any Other Things.

### Suggested contribution split

The work can land as small, independently reviewable changes:

1. ring fallback and architecture-contract tests;
2. shared CUDA architecture classification;
3. FA/provider capability audit;
4. B200 resource markers and CI presets;
5. B200 routing, ring, and quantization smokes;
6. `sm_120` resource and CI lane;
7. architecture-specific documentation and release manifest; and
8. optional performance-provider follow-ups.

### References

- [vLLM GPU installation and Blackwell CUDA requirement](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [#3079: CUDNN_ATTN and FLASHINFER_ATTN Blackwell auto-routing](https://github.com/vllm-project/vllm-omni/pull/3079)
- [#3015: SageAttention 3 Blackwell diffusion backend](https://github.com/vllm-project/vllm-omni/pull/3015)
- [#4025: Qwen3-Omni NVFP4 W4A4 serving on Blackwell](https://github.com/vllm-project/vllm-omni/pull/4025)
- [#4241: Quack FP8 on datacenter Blackwell](https://github.com/vllm-project/vllm-omni/pull/4241)
- [#4817: Quack gating for `sm_10x` versus `sm_12x`](https://github.com/vllm-project/vllm-omni/pull/4817)
- [#5288: TRTLLM diffusion-attention roadmap](https://github.com/vllm-project/vllm-omni/issues/5288)
- [#5283: TRTLLM diffusion attention with Skip-Softmax](https://github.com/vllm-project/vllm-omni/pull/5283)
- [#5611: B200 ring-attention failure](https://github.com/vllm-project/vllm-omni/issues/5611)
- [#5617: architecture-aware ring fallback](https://github.com/vllm-project/vllm-omni/pull/5617)
- [#5543: B200 Buildkite hardware selection](https://github.com/vllm-project/vllm-omni/pull/5543)
- [#4202: ModelOpt FP8 checkpoint behavior on Blackwell](https://github.com/vllm-project/vllm-omni/issues/4202)
- [#1959: NVFP4 quantization support for diffusion models](https://github.com/vllm-project/vllm-omni/issues/1959)
- [#4858: FA4 diffusion attention](https://github.com/vllm-project/vllm-omni/pull/4858)
- [#5344: Blackwell QK16/V8 FlashInfer attention](https://github.com/vllm-project/vllm-omni/pull/5344)
- [#5509: typed SAGE for TRTLLM diffusion attention](https://github.com/vllm-project/vllm-omni/pull/5509)
- [#3340: consolidate Omni/TTS FlashAttention dispatch through vLLM](https://github.com/vllm-project/vllm-omni/pull/3340)
