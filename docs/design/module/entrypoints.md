---
title: Entrypoints
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@yinpeiqi"
  - "@fake0fan"
  - "@alex-jw-brooks"
primary_code_paths:
  - vllm_omni/entrypoints/**
related_code_paths:
  - vllm_omni/reasoning/**
depends_on:
  - vllm_omni_config.md
  - input_output_modality_contracts.md
  - engine_orchestration.md
validation_paths:
  - tests/entrypoints/**
  - tests/reasoning/**
upstream_refs:
  - vllm.entrypoints
last_reviewed: 2026-07-16
---

# Entrypoints

Entrypoints translate offline, CLI, and serving requests into stable engine
operations and translate engine outputs into public responses.

## Candidate invariants

### ENTRY-INV-001: Entrypoints adapt but do not orchestrate

**Rule:** Entrypoints MUST NOT implement cross-stage routing or stage lifecycle
policy.

### ENTRY-INV-002: Public requests are normalized once

**Rule:** Public protocol values MUST be validated and converted to internal
request contracts before engine submission.

### ENTRY-INV-003: Streaming preserves request identity

**Rule:** Every streamed response MUST remain associated with the request and
output modality that produced it.

## Safe-change guide

Test request validation, protocol conversion, streaming, cancellation, and
error mapping for each affected entrypoint.
