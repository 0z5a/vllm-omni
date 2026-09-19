# HSDP × TeaCache full-model validation harness

Prepared against upstream `698f716`; GPU execution is pending. The driver reuses the existing ZImage checkpoint at revision `f332072aa78be7aecdf3ee76d5c247082da564a6`. Shared weights remain owned by the earlier task.

All four configurations use BF16, eager execution, two-rank Ulysses, nine denoising steps, seed 142 and identical raw prompts. Baseline, TeaCache, HSDP, and TeaCache+HSDP each run twice in fresh processes in a balanced forward/reverse order. Each process runs square → rectangular → square requests, with two warmups and five measurements per case. No output or speedup claim is available yet.

Separate untimed probe processes replace only the TeaCache hook with an observing subclass. The original decision method still executes unchanged. The probe records decisions, actual first-block invocations, first-step cache reset, and post-request DTensor parameters. It asserts that block invocation counts match the decisions and that HSDP parameters remain sharded. Rank agreement, nonzero cache hits, output quality, repeated-request output parity and timing analysis still require the actual run and audit.

Syntax, Ruff and CPU import checks passed in the target environment. Do not run this queue concurrently with the current GPU 2/3 E2E chain. Instrumented probe latency must not be used in the speed table.

| Configuration | E2E latency | Speedup vs baseline | Actual cache hits | Output comparison |
|---|---|---|---|---|
| Baseline | Pending | 1× | N/A | Pending |
| TeaCache | Pending | Pending | Pending | Pending |
| HSDP | Pending | Pending | N/A | Pending |
| TeaCache + HSDP | Pending | Pending | Pending | Pending |

## Forced no-hit control

A separate diagnostic driver supplies the supported coefficients `[0,0,0,0,1]` with threshold 0.2. The constant rescaled distance forces complete computation without changing the production decision method; threshold zero is invalid. Three CPU requests through the real observed hook each produce 9/9 compute decisions with clean resets. Complete-model cache-only and cache+HSDP no-hit probes are queued after DBCache validation and excluded from timing. Their rank records and PNG comparisons must still be audited. The normal timing driver and calibrated coefficients remain unchanged.
