# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Supported execution contract for the native whole-clip SeedVR2 pipeline."""

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig


def validate_seedvr2_config(config: OmniDiffusionConfig) -> None:
    if config.dtype != torch.float16:
        raise ValueError("SeedVR2 3B requires dtype=float16")
    if not config.enforce_eager:
        raise ValueError("SeedVR2 requires enforce_eager=True; compiled execution is not supported")
    if config.cache_backend != "none" or config.quantization_config is not None:
        raise ValueError("SeedVR2 requires unquantized weights and cache_backend=none for its single Euler step")
    if config.vae_use_slicing or config.vae_use_tiling:
        raise ValueError("SeedVR2 whole-clip VAE does not support slicing or tiling")
    if config.enable_cpu_offload or config.enable_layerwise_offload or config.enable_distributed_layerwise_offload:
        raise ValueError("SeedVR2 native whole-clip execution does not support CPU offload")
    parallel = config.parallel_config
    degrees = {
        "cfg_parallel_size": parallel.cfg_parallel_size,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "vae_patch_parallel_size": parallel.vae_patch_parallel_size,
        "text_encoder_tp_size": parallel.text_encoder_tp_size,
        "ulysses_degree": parallel.ulysses_degree,
        "ring_degree": parallel.ring_degree,
        "allgather_degree": parallel.allgather_degree,
    }
    for name, degree in degrees.items():
        if degree != 1:
            raise ValueError(f"SeedVR2 requires {name}=1; use window_parallel_size for its model-owned SP")
    if parallel.use_hsdp or parallel.enable_expert_parallel:
        raise ValueError("SeedVR2 does not support HSDP or expert parallelism")
    if config.lora_path is not None:
        raise ValueError("SeedVR2 does not support LoRA adapters")
