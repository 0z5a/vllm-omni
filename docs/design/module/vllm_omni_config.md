---
title: vLLM-Omni Configuration
kind: module
status: draft
owners:
  - "@lishunyang12"
  - "@alex-jw-brooks"
primary_code_paths:
  - vllm_omni/config/**
  - vllm_omni/deploy/**
  - vllm_omni/model_executor/stage_configs/**
related_code_paths:
  - vllm_omni/platforms/*/stage_configs/**
depends_on: []
validation_paths:
  - tests/config/**
upstream_refs:
  - vllm.config
last_reviewed: 2026-07-16
---

# vLLM-Omni configuration

Configuration defines how user input becomes validated engine, stage,
deployment, and platform settings.

## Candidate invariants

### CONFIG-INV-001: Configuration is validated before startup

**Rule:** Invalid stage graphs, resource assignments, and incompatible options
MUST fail before managed runtime processes start.

### CONFIG-INV-002: Precedence is deterministic

**Rule:** Defaults, files, environment variables, and CLI overrides MUST have a
documented and deterministic precedence.

### CONFIG-INV-003: Runtime modules consume validated configuration

**Rule:** Runtime modules MUST NOT independently reinterpret raw user
configuration.

## Safe-change guide

Test defaults, parsing, override precedence, serialization, invalid
combinations, and representative multi-stage configurations.
