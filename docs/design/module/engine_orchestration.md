---
title: Engine Orchestration
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@yinpeiqi"
  - "@fake0fan"
  - "@Sy0307"
primary_code_paths:
  - vllm_omni/engine/**
related_code_paths:
  - vllm_omni/distributed/omni_coordinator/**
  - vllm_omni/distributed/ray_utils/**
depends_on:
  - vllm_omni_config.md
  - input_output_modality_contracts.md
  - omni_connector.md
validation_paths:
  - tests/engine/**
  - tests/distributed/omni_coordinator/**
upstream_refs:
  - vllm.v1.engine
last_reviewed: 2026-07-16
---

# Engine orchestration

Engine orchestration coordinates configured AR and diffusion stages while
keeping public clients independent from stage processes and transports.

## Candidate invariants

### ORCH-INV-001: The orchestrator owns cross-stage routing

**Rule:** Entrypoints and stage clients MUST NOT independently forward a
request to a downstream stage.

### ORCH-INV-002: Stage clients do not own routing policy

**Rule:** Stage clients MUST implement communication and lifecycle operations
without selecting the next logical stage.

### ORCH-INV-003: Terminal state is monotonic

**Rule:** Once a request reaches a terminal state, orchestration MUST NOT
forward new work for that request.

## Safe-change guide

Test routing, output ordering, cancellation, failure propagation, shutdown, and
representative multi-stage execution.
