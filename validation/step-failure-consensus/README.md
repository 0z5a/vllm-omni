# Step preparation failure consensus

Candidate `17d38375713d04bd3092b43c8ce026bac4db0973` is independent on pinned base `698f716`. It is a local candidate, not a full-model validation or published PR.

The original helper imports `get_dit_group`, which does not exist, catches the ImportError, and returns only the local failure flag. A peer with successful preparation can therefore continue into denoising while another rank drops the request.

The fix uses the existing world coordinator CPU group. This scope follows the actual step RPC: `MultiprocDiffusionExecutor.execute_step` broadcasts the same scheduler output with `exec_all_ranks=True` and returns rank 0's response. Step workers do not apply the request-mode DP splitting path. A regression assertion preserves that dispatch contract. No new process group, platform-specific device allocation, or fallback catch is added.

| Validation | Baseline | Candidate | Interpretation |
|---|---|---|---|
| Real Gloo single/two-process failure consensus | 1 passed, 1 failed, 39.58 s | Included in passing suite | Two-process baseline fails the remote-failure assertion |
| Related CPU step suite | Narrow reproducer above | 35 passed, 3 hardware cases deselected, 39.64 s | Includes genuine Gloo collectives, not a mocked helper |
| Combined cancellation + failure fixes (`bbd391d`) | Separate candidates above | 186 passed, 11 hardware cases deselected, 33.50 s | Five related CPU suites; seven source hashes verified |
| Final executor dispatch contract | Not separately timed | 1 passed, 2.94 s | Same scheduler wave reaches all workers, only rank 0 replies |
| Changed-file mypy | 4 diagnostics in earlier completed baseline run | 3 existing diagnostics | Missing accessor error removed; no new diagnostic |
| Full GPU E2E speed improvement | Unmeasured | Unmeasured | Pending; test durations are not model latency |

The real Gloo tests use one and two CPU processes, report failures from every rank in turn, verify the following no-failure wave resets correctly, and call the real runner preparation path with a last-rank failure followed by a valid request. All ranks drop the failed state and retain the valid state. CPU platform device selection is patched solely to construct the real coordinator without allocating GPU memory.

This cannot recover failures occurring inside a pipeline collective before every rank reaches the consensus call. It does not validate full-model HSDP/offload, GPU failure recovery, or cross-rank cancellation. Those remain part of K6. The request lifecycle cleanup candidate is tracked separately at `463eb88`.

Final source hashes match `/dev/shm/0z5a-step-failure-17d3837`. Runtime: Python 3.12, torch 2.13, vLLM 0.29; CUDA_VISIBLE_DEVICES empty and OMP_NUM_THREADS=1. Applicable pre-commit hooks other than the existing mypy failures pass. The fresh baseline pre-commit run also completed: the same four baseline diagnostics versus three candidate diagnostics, with no new diagnostics.

Reproduce locally or in a CI CPU environment with the repository runtime and pytest dependencies:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q -m cpu tests/diffusion/test_diffusion_step_pipeline.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q tests/diffusion/test_diffusion_step_pipeline.py -k failure_consensus_gloo
uv tool run pre-commit run --files vllm_omni/diffusion/worker/diffusion_model_runner.py tests/diffusion/test_diffusion_step_pipeline.py
```

Combined validation source: `/dev/shm/0z5a-step-lifecycle-bbd391d`, corresponding to local integration commit `bbd391d` (cleanup `463eb88` plus this failure-propagation fix). The full model K6 harness is still pending.
