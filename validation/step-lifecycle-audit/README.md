# K6 request-state ownership audit

Baseline: `698f716`. Local candidate: `9aa4241` (not submitted as a PR). Remote immutable source: `/dev/shm/0z5a-step-cleanup-9aa4241`; both changed files match local SHA256.

The real step runner, driven by the existing CPU test pipeline, removes terminated requests from `state_cache` but retains their `StepRequestState` through `input_batch.states`. This is bounded retention of the last batch, not evidence of an unbounded leak. Model-private state can remain reachable even after a finished-request cleanup wave. No GPU byte count is inferred from this CPU reproduction.

The four-line candidate drops the cached input batch when no live state remains, both after terminal execution and after an empty/failed preparation wave. An empty wave with live state preserves its batch and request object. The reproduction records weak-reference ownership, not just dictionary membership.

| Check | Baseline | Candidate |
|---|---|---|
| Normal completion plus cleanup: request object retained | Yes | No |
| Cancellation cleanup wave: request object retained | Yes | No |
| New release regression cases | 2 failed | Pass in full CPU suite |
| CPU step and input-batch suites | Prior baseline recorded separately | 36 passed, 3 hardware tests deselected |
| Changed-file mypy | 4 errors | Same 4 errors, no new diagnostic |
| Other applicable changed-file pre-commit hooks | Pass | Pass |
| Full GPU lifecycle and E2E speed | Pending | Pending |

The baseline regression run preceded adding the empty-wave preservation assertion; the candidate suite includes that assertion. `mypy-comparison.json` normalizes line numbers and verifies the four diagnostics are unchanged. Pre-commit is **not fully passing**. One baseline error exposes an import of nonexistent `get_dit_group`; the surrounding ImportError fallback suppresses cross-rank failure coordination. That requires a separate, correctly scoped group/lifecycle fix and real distributed checks.

The runner fix does not establish that the engine delivers cancellation cleanup promptly when its final request is aborted. The additional `reproduce_engine_abort.py` drives the real synchronous engine and runner together to inspect that path. Observed result on the candidate: the engine returns `aborted=true`, the scheduler has no requests, yet the worker retains `abort-last` in both state cache and input batch. This confirms a separate cleanup-delivery gap in the synchronous path. Asynchronous service cancellation remains a separate required gate. K6 is not complete.

Logs retain one command that named a nonexistent input-batch test file and one engine-audit invocation before its upload; neither is counted as a passing test. The corrected invocations have separate logs. Model weights are shared with pending full E2E tasks and remain in place.
