---
title: Input, Output, and Modality Contracts
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@gcanlin"
primary_code_paths:
  - vllm_omni/inputs/**
  - vllm_omni/outputs/**
related_code_paths:
  - vllm_omni/tokenizers/**
depends_on: []
validation_paths:
  - tests/inputs/**
upstream_refs:
  - vllm.inputs
  - vllm.outputs
last_reviewed: 2026-07-16
---

# Input, output, and modality contracts

These contracts define the data that may cross entrypoint, orchestration,
stage, and model boundaries.

## Candidate invariants

### IO-INV-001: Boundary data has an explicit modality

**Rule:** Data crossing a module or stage boundary MUST identify its modality
and use the corresponding validated contract.

### IO-INV-002: Request identity is stable

**Rule:** Request identity MUST be preserved across conversions, stages,
streaming updates, cancellation, and errors.

### IO-INV-003: Internal objects do not leak into public protocols

**Rule:** Entrypoints MUST explicitly translate internal output objects into
public response types.

## Safe-change guide

Test construction, validation, serialization, streaming, optional fields, and
compatibility at every affected producer-consumer boundary.
