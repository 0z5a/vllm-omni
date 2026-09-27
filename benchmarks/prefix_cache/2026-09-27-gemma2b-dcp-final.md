# Real Gemma 2B DCP E2E: final matched comparison

## Result

**Correctness passed with the same explicit FlashInfer runtime correction applied to every arm.** All eight runs completed naturally. All 200 requests produced exactly the same 32 token IDs as the TP=1 reference; cache-on warm requests reused 928/960 prompt tokens, and cache-off reused zero.

| Round | Original off p50 | Original on p50 | Optimized off p50 | Optimized on p50 | Original-on / optimized-on | Optimized off/on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.992512 s | 1.102165 s | 1.026203 s | 0.853840 s | **1.291×** | **1.202×** |
| 2 | 0.892461 s | 0.879329 s | 1.040965 s | 0.842747 s | **1.043×** | **1.235×** |

Both orders show lower optimized cache-on p50 than original cache-on (1.043–1.291×), and lower optimized cache-on p50 than its own cache-off (1.202–1.235×). Original cache-on/off behavior varies substantially with run order and shared-host load. These are two matched screening rounds, not a guaranteed production speedup.

## Conditions and code identity

- Real pretrained `google/gemma-2b`, both safetensor SHA-256 values independently verified (see original diagnostic report).
- Original Omni code: `ffdb0df43793b3cc1d527757928a134e0a6c2777` (#8066); optimized Omni code: `ea311af2e2119e264d2dd32abc35d57067ef09ed` (PR #36 before evidence-only commits). All 1698 files under `vllm_omni/` matched GitHub blob hashes in both source directories.
- Same RTX 5090 GPUs 4 and 7; CPU affinity 32–39,56–63,96–103,120–127; torch 2.13.0+cu130, vLLM 0.29.0; existing 0z5a environment.
- TP=2/DCP=2; FlashInfer; block size 16; BF16; eager execution; async scheduling disabled; GPU memory utilization 0.4; 960 prompt tokens; 32 output tokens; temperature 0; seed 42.
- Each arm has one cold request plus 24 measured requests. Round 1 order: original off, original on, optimized off, optimized on. Round 2 reverses that order.
- The supervisor checks for free GPUs before each arm. Earlier arms on other device pairs are excluded from this table. No process was terminated to obtain resources.

## Runtime prerequisite — separate from the Omni change

Unmodified installed FlashInfer DCP metadata handling produced anomalous text and cache-on/off token mismatch in both original and optimized Omni, and in native vLLM with both `ag_rs` and `a2a`. TP=1 and TP=2/DCP=1 controls were correct.

The builder computed its NumPy length view and KV block counts before converting lengths to the local DCP context. The [scoped runtime patch](flashinfer-dcp-local-metadata.patch) clones CPU lengths before mutation and recomputes the view and block counts after the adjustment. The [process-local diagnostic overlay](diagnostic_dcp_metadata.py) applies exactly that correction to both comparison arms. It does not edit installed packages. Corrected native vLLM passed all eight control requests and matched the TP=1 reference.

**The Omni PR optimization does not itself repair this vLLM runtime bug.** The table above requires this shared runtime correction. The uncorrected timing table remains invalid as correctness-qualified evidence and is retained in [the diagnostic report](2026-09-27-gemma2b-dcp.md).

## Reproduction and raw evidence

- [All per-request timings/token IDs, deployment YAML, and natural-exit markers](2026-09-27-gemma2b-dcp-final-results.json).
- [E2E harness](probe_e2e_metadata.py), [shared runtime overlay](diagnostic_dcp_metadata.py), [eight-arm supervisor](run_real8066_gemma2b_metadata.py). The supervisor records the original remote paths; adapt only the work/model paths to reproduce on another host.
- [Earlier native/TP diagnostic controls](2026-09-27-gemma2b-diagnostics.json).

Engine shutdown uses an idle-core exit request, waits for worker/core exits, and asserts zero exit codes before writing completion markers. No kill/terminate signals are used. Shutdown can emit an EngineDeadError after the core has exited and before orchestration stops; all final arm completion markers and exit-code assertions passed.

## Cleanup

After all eight runs completed and live process/file references were checked, this task's Gemma model directory on gongji was removed (5,034,246,144 allocated bytes). A100 task-owned model copies had already been removed and their absence checked earlier; the final SSH recheck was refused, and no files were downloaded to A100 during this final test series. Model weights were never downloaded to the local Mac. Source, logs and test evidence are retained.
