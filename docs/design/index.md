# Design Documents

This section contains design documents and architecture specifications for vLLM-Omni.

## Architecture Documents

- [Architecture Overview](architecture_overview.md)

## Feature Design Documents

- [Disaggregated Inference](feature/disaggregated_inference.md)
- [Ray-based Execution](feature/ray_based_execution.md)
- [Adding Step Execution Support for Diffusion Pipelines](feature/diffusion_step_execution.md)
- [Request-Level Batching for Diffusion](feature/diffusion_request_level_batching.md)
- [Continuous Batching for Step-Wise Diffusion](feature/diffusion_continuous_batching.md)

## Infrastructure Design Documents

- [Prometheus Metrics](metrics.md)

## Module Design Documents

- [Entrypoints](module/entrypoints.md)
- [vLLM-Omni Configuration](module/vllm_omni_config.md)
- [Input, Output, and Modality Contracts](module/input_output_modality_contracts.md)
- [Engine Orchestration](module/engine_orchestration.md)
- [OmniConnector](module/omni_connector.md)
- [Model Integration](module/model_integration.md)
- [Autoregressive Runtime](module/ar_runtime.md)
- Diffusion
  - [Overview](module/diffusion/index.md)
  - [Runtime](module/diffusion/diffusion_runtime.md)
  - [Model Integration](module/diffusion/diffusion_model_integration.md)
  - [Continuous Batching](module/diffusion/continuous_batching.md)
  - [Parallelism](module/diffusion/parallelism.md)
  - [Offloader](module/diffusion/offloader.md)
- [Execution Platforms](module/execution_platforms.md)
- [Cache Management](module/cache_management.md)
- [Quantization](module/quantization.md)
- [Observability](module/observability.md)
- [Profiling](module/profiling.md)
- [Benchmarking](module/benchmarking.md)
