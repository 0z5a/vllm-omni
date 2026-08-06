<!-- markdownlint-disable MD013 -->

# RFC: Align vLLM-Omni error handling with vLLM

> **Status:** Draft for discussion  
> **Feedback period:** Two weeks after publication  
> **vLLM-Omni snapshot:** [`1b718f7b`](https://github.com/vllm-project/vllm-omni/commit/1b718f7b1529bf7b59450de2dba0f820ef676dc8)  
> **Upstream vLLM snapshot:** [`fa2a2589`](https://github.com/vllm-project/vllm/commit/fa2a2589bd2a1fce0851df7fd42ffb54b6195f04)  
> **Upstream direction:** [vLLM RFC #48227](https://github.com/vllm-project/vllm/issues/48227) and [vLLM PR #49665](https://github.com/vllm-project/vllm/pull/49665)

## Motivation

The error fragmentation described in upstream vLLM also exists in vLLM-Omni
and is amplified by multi-stage execution and additional serving transports.
In the audited snapshot, `vllm_omni/` contains:

- 2,147 `raise ValueError` sites, including 171 in `entrypoints/`;
- 166 `raise TypeError` sites;
- 460 `raise RuntimeError` sites;
- no direct `raise VLLMValidationError` site; and
- one direct `raise GuardrailViolationError` site.

These counts do not imply that every raw exception should become public. They
show that raw Python types currently represent request validation, internal
invariants, configuration errors, model failures, and control flow, while
semantic classification is rare.

This causes four concrete problems:

1. **Internal failures can be reported as HTTP 400.** Several entrypoints treat
   every `ValueError` as invalid user input:

   ```python
   except OmniClientError as exc:
       raise HTTPException(exc.status_code, exc.message) from exc
   except ValueError as exc:
       raise HTTPException(400, str(exc)) from exc
   except Exception as exc:
       raise HTTPException(500, str(exc)) from exc
   ```

   A `ValueError` from parsing, model execution, output conversion, or
   configuration can therefore be reported as `400 Bad Request` instead of
   `500 Internal Server Error`. The safer default is that only an explicit
   client-error type becomes 4xx.

2. **Ownership is guessed after crossing stage boundaries.** Current receivers
   reconstruct a client error when the serialized status is in the 4xx range
   and otherwise construct an engine error. This derives semantic ownership
   from a status chosen earlier, so a wrong classification cannot be recovered.
   Parameter information and nonfatal stage context are also not consistently
   preserved.

3. **The same failure has different public contracts.** HTTP routes, SSE,
   WebSocket, speech helpers, and asynchronous video jobs convert failures
   independently. [vLLM-Omni PR #3969](https://github.com/vllm-project/vllm-omni/pull/3969)
   already fixed one result of this fragmentation: voice endpoints returning
   an error object with HTTP 200.

4. **Prometheus status metrics are fragile.**
   [vLLM PR #44303](https://github.com/vllm-project/vllm/pull/44303) showed that
   a client can receive 4xx while `http_requests_total` records 5xx when an
   exception crosses the Prometheus middleware before reaching an outer
   catch-all handler. Omni adds its own exceptions and handlers on top of the
   upstream application, so relying on per-type registration can reproduce the
   mismatch. Returning an error model with default HTTP 200 creates the inverse
   problem: failed requests are recorded as 2xx.

## Proposed change

Adopt upstream vLLM's semantic exception hierarchy for in-process errors and
add one Omni-owned serializable record for stage/process transport:

```text
semantic VLLMError
    -> normalize to OmniErrorDetails
    -> stage/process transport
    -> one boundary renderer
    -> HTTP / SSE / WebSocket / async job / offline Python
```

The supported vLLM baseline must first contain
[vLLM PR #49665](https://github.com/vllm-project/vllm/pull/49665), including
commit `58f96593971b6287cdc8867024a6dfb9985c545c`. vLLM-Omni must import, not copy,
the upstream hierarchy.

### 1. Use semantic exception ownership

Generic failures use upstream classes directly:

| Failure | Exception | Status |
| --- | --- | ---: |
| Invalid or incompatible request parameter | `VLLMValidationError` | 400 |
| Missing model, LoRA, voice, job, or artifact | `VLLMNotFoundError` | 404 |
| Permanently inaccessible or undecodable request media | `VLLMUnprocessableEntityError` | 422 |
| Recoverable per-request engine failure | `EngineGenerateError` | 500 |
| Unrecoverable engine failure | `EngineDeadError` | 500 |
| Unknown server failure | `VLLMServerError` or the `Exception` safety net | 500 |

Omni adds only types with concrete Omni behavior:

| Omni type | Parent | Status | Use |
| --- | --- | ---: | --- |
| `GuardrailViolationError` | `VLLMClientError` | 400 | Guardrail rejects request content |
| `OmniConflictError` | `VLLMClientError` | 409 | Existing async job is in an incompatible state |
| `OmniNotImplementedError` | `VLLMServerError` | 501 | Invoked server operation is not implemented |
| `OmniServiceUnavailableError` | `VLLMServerError` | 503 | Required handler, pipeline, or stage is unavailable |
| `OmniStageTimeoutError` | `VLLMServerError` | 504 | Stage or generation deadline expires |
| `OmniEngineGenerateError` | `EngineGenerateError` | 500 | Nonfatal stage failure with `stage_id` |
| `OmniEngineDeadError` | `EngineDeadError` | 500 | Fatal stage failure with `stage_id` |

Each new subtype lands with its first production raise site. Existing 403, 413,
and 429 transport behavior remains compatible; dedicated types are added only
when a concrete use case and retry policy are defined.

`OmniClientError` remains temporarily as a compatibility and remote-error
wrapper, but is reparented from `ValueError` to `VLLMClientError`. A symmetric
`OmniServerError(VLLMServerError)` represents server failures reconstructed
from the wire.

### 2. Preserve semantics across stages

Introduce one internal serializable record:

```python
@dataclass(frozen=True, slots=True)
class OmniErrorDetails:
    message: str
    category: Literal["client", "server"]
    status_code: int
    error_type: str
    parameter: str | None = None
    request_id: str | None = None
    stage_id: int | None = None
    fatal: bool = False
```

Required invariants:

- client errors have a 4xx status and server errors have a 5xx status;
- unknown exceptions normalize to server error 500;
- `fatal=True` is valid only for server errors;
- public messages and parameter names are sanitized;
- tracebacks, exception objects, media, tensors, credentials, and rejected
  values are never serialized; and
- consumers use `category`, not the numeric status range, to reconstruct
  ownership.

`ErrorMessage`, `OmniRequestOutput`, `DiffusionOutput`, and stage IPC carry this
record or its complete field set. Existing flat message/status/type fields
remain readable for one compatibility window.

### 3. Centralize rendering

Register one Omni-aware `VLLMError` handler that:

- preserves upstream status, type, and parameter mapping;
- dispatches engine errors through Omni's stage-aware termination behavior;
- renders Omni-specific 409, 501, 503, and 504 errors centrally; and
- runs inside the instrumented middleware boundary so HTTP metrics observe the
  actual response status.

Temporary handlers for raw `ValueError`, `TypeError`, `OverflowError`, and
`NotImplementedError` remain during migration. The permanent `Exception`
safety net always maps unknown failures to 500.

Transport rules:

- non-streaming HTTP status must equal `error.code`;
- SSE and WebSocket streams that already sent success headers emit one terminal
  event with the canonical nested `error` object;
- async jobs store the canonical error and do not guess a new status when
  retrieved; and
- offline APIs raise the semantic exception or a remote
  `OmniClientError`/`OmniServerError` carrying `.details`.

Routes must not return a Pydantic `ErrorResponse` without an explicit HTTP
status.

### 4. Migrate incrementally

1. **Baseline:** add table-driven mapping, middleware metrics, transport, and
   engine-lifecycle regression tests for current behavior.
2. **Bridge:** adopt the required vLLM baseline, reparent compatibility types,
   add `OmniErrorDetails`, and register the semantic handler.
3. **Request boundaries:** migrate project-owned validation one raise site at a
   time to `VLLMValidationError`, `VLLMNotFoundError`, or
   `VLLMUnprocessableEntityError`. Catch `VLLMClientError`, not `ValueError`.
4. **Stage and transports:** preserve stage context, standardize HTTP/SSE/
   WebSocket/job rendering, then remove raw fallbacks only when their audited
   inventories reach zero or an explicit allowlist.

This is not a mechanical replacement of every `ValueError`. Each site must be
reviewed to distinguish user-correctable input from internal failure.

## Compatibility and correctness

Correctly typed errors retain their current status and OpenAI-compatible
envelope. Intentional corrections may change:

- an internal raw `ValueError` from 400 to 500;
- invalid or inaccessible media from 400/500 to 422;
- missing resources to 404;
- conflicting async-job operations to 409; and
- unavailable pipelines or stages to 503.

Every status change requires an endpoint regression test and release note.

Reparenting `OmniClientError` means external Python code that catches only
`ValueError` will no longer catch it. Retain its import and constructor for at
least one release, document the inheritance change, and recommend catching
`VLLMClientError` or a specific subtype.

Minimum test coverage:

- every upstream and Omni exception-to-status/type mapping;
- unknown `ValueError` becoming 500 after its fallback is removed;
- wire round trips preserving category, parameter, stage, and fatality;
- HTTP status matching `error.code`;
- SSE, WebSocket, and async-job terminal errors;
- `http_requests_total` distinguishing 4xx and 5xx;
- nonfatal stage failure affecting one request;
- fatal stage failure retaining existing termination behavior; and
- public errors excluding tracebacks, paths, secrets, and multimodal payloads.

## Non-goals

- Copying or forking upstream generic exception classes.
- Converting every model/kernel exception in one change.
- Making control-flow cancellation a public server error.
- Redesigning endpoint ownership, retry policy, or fatal OOM lifecycle.
- Adding speculative exception subclasses without a production raise site.

## Open questions

1. Should the generic 409, 501, 503, and 504 types eventually move upstream
   after Omni validates their use cases?
2. Should `request_id` and `stage_id` become shared upstream response fields?
3. How long must mixed-version stage processes support the legacy wire fields?
4. For failed async jobs, should metadata retrieval return the stored error
   status or HTTP 200 with a canonical embedded error?

## Community help wanted

We welcome community help with:

- validating the proposed status and exception mapping;
- identifying representative raw-exception sites for the first migrations;
- reviewing mixed-version and multi-node wire compatibility;
- adding mapping, middleware-metrics, stage round-trip, and transport tests; and
- coordinating the required vLLM baseline update.

If you would like to help, please comment with the endpoint, stage, or test area
you want to own. After the feedback period, implementation can be split into
small, independently reviewable PRs following the migration phases above.

## Feedback period and CC

Feedback is requested for two weeks after the RFC issue is published.

CC: maintainers for entrypoints, engine/orchestrator, diffusion, metrics, and
the upstream vLLM error hierarchy.

## References

- [vLLM PR #44303: Fix 4xx errors recorded as 5xx in `http_requests_total`](https://github.com/vllm-project/vllm/pull/44303)
- [vLLM RFC #48227: Standardize vLLM Entrypoint Error Handling](https://github.com/vllm-project/vllm/issues/48227)
- [vLLM PR #49665: Standardize request error handling with `VLLMError`](https://github.com/vllm-project/vllm/pull/49665)
- [vLLM-Omni PR #3969: Correct HTTP status for voice errors](https://github.com/vllm-project/vllm-omni/pull/3969)
- [vLLM-Omni PR #4297: Guardrail client-error propagation](https://github.com/vllm-project/vllm-omni/pull/4297)
- [vLLM-Omni RFC #1346: Exit on OOM](https://github.com/vllm-project/vllm-omni/issues/1346)
- [vLLM-Omni RFC #5198: Broad exception cleanup](https://github.com/vllm-project/vllm-omni/issues/5198)
- [vLLM-Omni RFC #5227: Entrypoint and API-server refactor](https://github.com/vllm-project/vllm-omni/issues/5227)
