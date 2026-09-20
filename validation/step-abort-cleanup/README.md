# Step request abort cleanup

Candidate: `463eb88a25fd4e7c2c556307af3081aa3adb7558`, based on `698f716` through `9aa4241`. Local candidate only; no PR or full GPU validation yet. Author and committer: 0z5a.

The engine now retires step state on every worker when aborting a request, without requiring a later denoise step. The background loop delivers terminal outputs for cancelled running and waiting requests, including when cancellation empties the queue. The runner releases the last input batch; worker LoRA request state is retired alongside it. Duplicate IDs are deduplicated and existing terminal outcomes are preserved.

| Check | Before | Candidate | Result |
|---|---|---|---|
| New synchronous/background abort regressions | 2 failed on `9aa4241` | Included in 183 passing CPU tests | Cleanup and output delivery corrected |
| Five related CPU suites | See raw initial fixture failures | 183 passed, 11 hardware cases deselected, 16.32 s | CPU coverage only |
| Final engine cleanup suite, including cancellation during execution | Added running + waiting cancellation case | 18 passed, 1.63 s | Final parameterized test verified |
| Main changed-file mypy diagnostics | 56 | 56, no new normalized diagnostics | Existing failures remain |
| Additional model-runner test mypy diagnostics | 8 | 8, no new normalized diagnostics | Existing failures remain |
| Final cleanup-file mypy diagnostics | 1 | Same existing diagnostic | Other applicable hooks passed |
| Full GPU E2E speed improvement | Not measured | Not measured | Pending; no speedup claim |

Durations are test-run wall times, not model latency or speedup measurements. `fixed-suite.log` precedes the final additional asynchronous variant; `background-abort-final.log` covers the final cleanup file. Earlier failed attempts are retained: `fixed.log` used incomplete test fixtures and real GPU peak-memory calls in CPU tests; `background-abort.log` exposed a missing fixture admission setting. Fixtures now initialize the fields used by the real control loop and mock only the pre-existing CPU peak-memory platform boundary.

The SHA-256 manifest matches all seven final source files on the remote immutable snapshot `/dev/shm/0z5a-step-abort-463eb88`. CPU tests used CUDA_VISIBLE_DEVICES empty, Python 3.12, torch 2.13 / vLLM 0.29 runtime. This does not prove distributed cancellation, paged KV, full-model pause/resume/offload, or cross-rank failure handling. The separate existing missing DiT-group accessor remains under investigation.
