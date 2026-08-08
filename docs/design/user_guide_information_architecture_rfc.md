---
title: User Guide Information Architecture and Content Refactor
kind: rfc
status: proposed
owners:
  - vLLM-Omni documentation maintainers
last_reviewed: 2026-08-08
---

# User Guide Information Architecture and Content Refactor

**Status:** Proposed

**Decision requested:** Approve the target information architecture and the
staged migration plan in this RFC.

## Summary

The vLLM-Omni User Guide has broad coverage, but its navigation is currently
organized partly around repository directories and build-generated categories
rather than around user tasks. As a result, onboarding, serving, examples,
deployment configuration, optimization, integrations, and operations are mixed
at the same level. The same concept can also appear under different labels;
for example, the published site currently shows both `Examples →
Quantization` and `Features → Quantization`.

This RFC proposes a task-oriented User Guide with a small number of stable
sections, a single model-and-recipe entry point, and explicit separation
between conceptual guides, runnable recipes, reference material, and
troubleshooting. The first migration should preserve existing page URLs and
focus on navigation and cross-links.

## Motivation and current status

Before this RFC's proposed changes, the source navigation was defined in
[`docs/.nav.yml`](../.nav.yml) and its User Guide contained:

```text
Getting Started
Serving
Examples
General
Configuration
Models
Features
Fault Tolerance
```

This is a reasonable inventory, but the boundaries are inconsistent:

- `General` contains FAQ and metrics, while fault tolerance is a separate
  top-level section. These are all operational concerns.
- `Features` is a catch-all for Sleep Mode, Diffusion Features, Quantization,
  and ComfyUI, even though those pages represent different kinds of content.
- `Configuration` mixes deployment topology, memory sizing, parallel
  strategies, and a general landing page.
- The examples generator scans every top-level directory under `examples/` and
  rewrites the Examples section at build time
  ([`generate_examples.py`](../mkdocs/hooks/generate_examples.py)). The
  `examples/quantization/` directory therefore becomes a second Quantization
  entry in the published sidebar.
- `Supported Models` is a large compatibility table, but it only links to the
  recipes directory in general. It does not connect each model row to a
  runnable, hardware-specific recipe.
- The supported-model table has 76 rows, while the repository contains 50
  recipe files. Recipe availability is therefore useful information in its
  own right and must not be implied by model support.
- [`Session State Manager`](../features/session_state_manager.md) exists as a
  user-facing document, but its primary placement should be clearly marked as
  experimental rather than presented as a general feature.
- [`Pipeline Parallelism`](../user_guide/diffusion/parallelism/pipeline_parallel.md)
  exists as a user-facing document, but its primary User Guide placement is
  deferred until its support and maintenance status are settled.
- [`mot_config.md`](../user_guide/diffusion/mot_config.md) is an internal
  tuning-reference document and should not be promoted to the main User Guide
  unless its audience and maintenance model change.

The published [vLLM Recipes](https://recipes.vllm.ai/) site already provides
the right model-oriented experience: users choose a model and receive
hardware-specific commands and deployment guidance. The vLLM-Omni model index
should connect to that experience instead of leaving users to search for a
recipe separately.

## Goals

1. Organize the User Guide around the questions users ask: how do I install,
   run, serve, configure, optimize, integrate, and operate vLLM-Omni?
2. Give every supported model a clear recipe status and, when available, a
   direct recipe link.
3. Keep the global sidebar short and stable without creating one navigation
   entry per model.
4. Preserve existing page URLs during the first migration.
5. Make generated examples fit an explicit conceptual category rather than
   deriving category names directly from source-directory names.
6. Distinguish user-facing guidance from developer design documents and
   internal tuning references.

## Non-goals

- Changing the vLLM-Omni API, CLI, or runtime behavior.
- Rewriting every page in the first change.
- Making a recipe claim that has not been validated on a concrete hardware
  configuration.
- Removing existing examples or recipes because they are not yet represented
  in the model index.
- Redesigning the MkDocs theme or responsive CSS.

## Proposed User Guide structure

The target structure is:

```text
User Guide
├── Start Here
│   ├── Installation
│   │   ├── GPU
│   │   └── NPU
│   └── Quickstart
├── Models and Recipes
│   └── Supported Models
├── Serving
│   ├── OpenAI-Compatible API
│   │   ├── Chat Completions
│   │   ├── Image Generation
│   │   ├── Image Editing
│   │   ├── Text-to-Speech
│   │   ├── Audio Generation
│   │   └── Video Generation
│   └── Realtime and Interactive Streaming
│       ├── Realtime API
│       ├── Full-Duplex API (experimental)
│       ├── Streaming Video Input
│       └── Streaming Text-to-Speech
├── Examples
│   ├── Offline Inference
│   └── Online Serving
├── Features
│   ├── Runtime and Stage Execution
│   │   ├── Execution Modes and Streaming
│   │   └── Sleep Mode
│   ├── Diffusion Acceleration
│   │   ├── Overview and Compatibility
│   │   ├── Quantization
│   │   ├── CPU Offloading
│   │   ├── Cache Acceleration
│   │   ├── Parallelism
│   │   │   ├── Parallelism Overview
│   │   │   └── VAE Parallelism
│   │   ├── Attention Backends
│   │   ├── Regional Compilation
│   │   ├── Frame Interpolation
│   │   ├── Startup and Loading
│   │   │   └── Multi-Thread Weight Loading
│   │   └── LoRA
│   └── Experimental
│       └── Session State Manager
├── Deployment and Configuration
│   ├── Pipeline and Deploy Configurations
│   └── GPU Memory
├── Integrations
│   └── ComfyUI
└── Operations and Troubleshooting
    ├── FAQ
    ├── Production Metrics
    └── Fault Tolerance
```

This is a conceptual navigation tree, not a required filesystem move. Existing
URLs should remain valid; the first implementation can move pages in the
sidebar without moving files.

### Why this structure

- **Start Here** gives new users one obvious entry point and makes Installation
  visible instead of exposing only the hardware children.
- **Models and Recipes** puts model choice before model-specific execution.
- **Serving** is the API reference organized by input/output modality.
- **Realtime and Interactive Streaming** gives long-lived WebSocket and
  full-duplex sessions a first-class home instead of hiding them in model
  examples. The current implementation has several model-specific contracts,
  so this section should document the common protocol first and link to each
  model recipe for capability-specific behavior.
- **Examples** contains runnable workflows, not conceptual feature guides.
- **Deployment and Configuration** contains the basic deployment settings and
  resource sizing needed to run a model.
- **Integrations** separates external tools from core runtime features.
- **Operations and Troubleshooting** groups metrics, failure modes, and FAQ
  content around operating a deployment.

### Alignment with Developer Guide feature design

The User Guide uses the same primary feature-design group names already used by
the Developer Guide: `Runtime and Stage Execution` and `Diffusion Acceleration`.
This keeps contributor architecture and user operation guides discoverable by
the same concept without exposing implementation pages in the user sidebar.

`Communication` remains Developer Guide-only because the current connector
pages describe implementation contracts rather than user operations. The User
Guide may add a Communication subsection later if a stable deployment or
connector usage guide is created.

The User Guide explains configuration, supported usage, API operations, and
practical limitations. The existing Developer Guide feature-design pages remain
unchanged and continue to explain runtime architecture, implementation
boundaries, and contributor contracts. Quantization links to the existing
[`design/feature/quantization.md`](feature/quantization.md) design document;
this RFC does not add or move Developer Guide pages for Sleep Mode.

### Deferred from the primary User Guide

This RFC intentionally removes the following topics from the primary User Guide
navigation. The source pages and public URLs are retained; this is a scope and
discoverability decision, not a deletion proposal:

- Composable Parallel Strategies.
- Prefill-Decode (PD) Disaggregation.
- Diffusion Pipeline Parallelism until its user-facing support and placement
  are settled.
- Internal tuning references such as `mot_config.md`.

These topics can remain reachable from existing direct links, recipes, design
documents, or later dedicated documentation proposals. They should not be
placed under a generic `Features and Optimization` bucket in this RFC. The
supported user-facing guides are grouped by task instead.

## Model and recipe index

`Supported Models` should remain one searchable reference page rather than
becoming a long list of model pages in the sidebar. Each model row should
provide:

| Field | Purpose |
| --- | --- |
| Model family | Stable user-facing model name |
| Capabilities | Text, image, video, audio, action, or combinations |
| Example checkpoints | Hugging Face identifiers |
| Supported hardware | Backend compatibility currently claimed by vLLM-Omni |
| Recipe | Direct link to a validated deployment recipe, possibly several links by hardware/profile |
| Examples | Links to matching offline and/or online examples |

Recipe links should use this precedence:

1. The published [vLLM Recipes](https://recipes.vllm.ai/) page when the model
   has a published recipe.
2. The corresponding file under the repository's
   [`recipes/`](https://github.com/vllm-project/vllm-omni/tree/main/recipes)
   directory when it is not yet published.
3. `No validated recipe yet` when neither exists.

The index must not equate implementation support with recipe coverage. For
models with multiple profiles, such as different GPU sizes, NPU, or MUSA
deployments, the Recipe cell should expose each tested profile rather than
silently choosing one. For model families with several variants, one row may
link to a family landing page containing multiple recipes.

To prevent the table and recipe catalog from drifting, the recipe mapping
should eventually be represented as structured metadata keyed by canonical
model ID. A generated table is preferred to maintaining a growing set of
hand-written links. The existing [recipe catalog](https://github.com/vllm-project/vllm-omni/tree/main/recipes)
and `supported_models.md` remain the human-readable sources during migration.

## Serving surface inventory

The current repository already exposes more than the HTTP endpoints listed in
the original sidebar:

| Surface | Current evidence | Documentation direction |
| --- | --- | --- |
| OpenAI-compatible HTTP APIs | Chat, image, audio, speech, and video pages under `docs/serving/` | Keep as the stable API reference group |
| Realtime WebSocket | Qwen3-Omni `/v1/realtime`, plus model-specific clients and recipes | Add a canonical Realtime API guide and link model recipes |
| Full-duplex WebSocket | MiniCPM-o `/v1/realtime?duplex=1`, PersonaPlex `/api/chat` and `/v1/audio/duplex`, and the unified duplex control plane | Add an experimental Full-Duplex API guide; document protocol differences explicitly |
| Streaming video input | `serving/video_stream_api.md` and the Qwen3-Omni client | Keep as a serving guide under Realtime and Interactive Streaming |
| Streaming text-to-speech | `/v1/audio/speech/stream` WebSocket and PCM/SSE modes in `serving/speech_api.md` | Keep the contract in the speech API page and link it from the Realtime landing page |
| Model-specific interactive endpoints | Cosmos3/OpenPI and JoyVL interaction recipes | Keep endpoint details in the model recipe; link from the model index rather than adding one global sidebar item per model |

Realtime and full-duplex are related but should not be collapsed into one
generic promise. Realtime can mean incremental input or output over a
long-lived connection, while full-duplex means the server consumes and produces
streams concurrently, often with model-specific turn, barge-in, audio framing,
and session-state semantics.

## Content placement rules

Every new page should have one primary home determined by its user intent:

| Content type | Primary location | Example |
| --- | --- | --- |
| Installation or first successful run | Start Here | GPU installation, Quickstart |
| API or CLI contract | Serving or Reference | Chat Completions, Video API |
| Runnable model/hardware procedure | Examples or Models and Recipes | Qwen3-Omni recipe |
| Deployment topology or resource sizing | Deployment and Configuration | Deploy configuration, GPU memory |
| User-facing feature configuration or operation | User Guide → Features | Quantization, Sleep Mode |
| Feature architecture or contributor contract | Developer Guide → Feature Design | Existing quantization and diffusion acceleration design pages |
| Developer extension workflow | Developer Guide | Custom Pipeline Extension |
| Advanced runtime capability or optimization | Deferred from primary navigation | Composable Parallel Strategies, PD Disaggregation, internal tuning references |
| External product integration | Integrations | ComfyUI |
| Diagnosis, monitoring, or incident behavior | Operations and Troubleshooting | FAQ, metrics, failure modes |
| Internal architecture or extension contract | Developer Guide | Design documents |

Avoid using directory names as user-facing categories. In particular,
generated categories must have an explicit display-name mapping; a source
directory named `quantization` should not automatically create a sidebar item
with the same label as the user-facing Quantization guide.

## Navigation and generation changes

The first implementation should make `docs/.nav.yml` the conceptual source of
truth:

1. Keep the example-generation hook responsible for generating example pages.
2. Stop rewriting the tracked conceptual navigation from arbitrary top-level
   example directories, or replace that behavior with explicit category globs
   for `Offline Inference` and `Online Serving` only. Quantization scripts
   should not create a primary User Guide category.
3. Add explicit display names for generated categories where globs are not
   sufficient.
4. Keep the User Guide Features section grouped using the existing Developer
   Guide feature-design names, while limiting its pages to practical usage
   guides; expose implementation architecture under Developer Guide → Feature
   Design.
5. Keep internal tuning references out of the main User Guide.
6. Use redirects only when a later content migration changes a public URL.

The Serving section should add Realtime and Interactive Streaming as a separate
conceptual group. It should cover Realtime, Full-Duplex, streaming video input,
and streaming text-to-speech without duplicating every model-specific recipe in
the global sidebar.

The primary navigation should not expose generated categories that duplicate
or compete with a canonical guide. In particular, `Quantization Tools` should
not be introduced as a substitute for the Developer Guide feature design and
the model-specific recipe/configuration links.

## Migration plan

### Phase 1: Navigation-only cleanup

- Introduce the target section names in `docs/.nav.yml`.
- Keep existing Markdown paths and URLs.
- Remove the generated Quantization category from the primary Examples nav.
- Add a focused User Guide Features section using the existing Runtime and
  Stage Execution and Diffusion Acceleration group names. Put practical memory,
  execution, acceleration, loading, and adaptation guides under those groups;
  keep method-specific quantization pages and detailed acceleration pages
  grouped under their landing topics rather than flattening the global sidebar.
- Leave the existing Developer Guide structure and feature-design pages
  unchanged; align only the User Guide labels and grouping.
- Move FAQ, metrics, and fault tolerance under Operations and Troubleshooting.
- Put ComfyUI under Integrations.
- Keep Composable Parallel Strategies and PD Disaggregation out of the
  primary Configuration nav.
- Do not create a flat `Features and Optimization` section; use the grouped
  Feature subsections defined above.

### Phase 2: Model and recipe linking

- Add recipe and example columns to `supported_models.md`.
- Establish canonical model IDs and recipe-profile metadata.
- Link to published recipes first and repository recipes second.
- Mark supported models without validated recipes explicitly.

### Phase 3: Landing pages and cross-links

- Add short landing pages for Models and Recipes, Deployment and Configuration,
  Operations and Troubleshooting, and Realtime and Interactive Streaming where
  the existing pages are not sufficient.
- Add consistent `Next steps` links from Quickstart to a model recipe, serving
  API, and configuration guide.
- Add reciprocal links from recipes back to the supported-model row and the
  relevant API/feature guides.

### Phase 4: Validation and maintenance

- Build with `mkdocs build --strict`.
- Check that the build does not leave a modified tracked navigation file.
- Run a link checker over local and external recipe links.
- Inspect the User Guide at desktop and narrow viewport widths.
- Review recipe status whenever a model-support row changes.

## Acceptance criteria

- The User Guide has no more than eight focused top-level sections and no flat
  Features and Optimization bucket.
- Every top-level section answers a distinct user task.
- `Quantization Tools` does not appear as a primary User Guide section.
- Quantization and Sleep Mode each have a concise user-facing entry under User
  Guide → Features, while existing Developer Guide feature-design pages remain
  unchanged.
- Every supported-model row has either a direct recipe link, a recipe-family
  link, or an explicit no-recipe status.
- Realtime and Full-Duplex have a canonical serving entry, with model-specific
  protocol and capability differences linked to the relevant recipes.
- Deferred topics remain available through existing direct URLs or contextual
  links; no source page is deleted as part of this RFC.
- Existing public documentation URLs continue to resolve, or have redirects.
- A clean documentation build does not rewrite the tracked conceptual nav.
- The navigation remains usable on narrow screens without requiring users to
  scan the complete model catalog.

## Open decisions

The recommended decisions are:

1. Use the target structure above as the User Guide baseline.
2. Keep one Supported Models page and add direct recipe links to its rows.
3. Prefer published recipes, with repository recipes as fallback links.
4. Treat recipe availability as a separate status from backend support.
5. Perform a navigation-only migration before moving or rewriting content.
6. Use the existing Developer Guide feature-design names for the User Guide
   Features hierarchy, keep contributor architecture unchanged, and keep
   explicitly deferred topology/internal tuning topics out of the primary
   navigation.
7. Add Realtime and Full-Duplex as a serving subsection, with one shared
   protocol overview and model-specific recipe links.
