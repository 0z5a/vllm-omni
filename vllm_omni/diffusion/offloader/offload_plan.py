# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative OffloadPlan for distributed layerwise offload.

Models declare this as a class attribute ``_offload_plan`` on the
pipeline class. When present, the offloader uses it instead of
heuristic block discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class OffloadPlan:
    """Optional declarative metadata for distributed layerwise offload.

    Models declare this as a class attribute ``_offload_plan`` on the
    pipeline class.  When present, the offloader uses it instead of
    heuristic block discovery, making it easier to support new models
    without offload-specific logic in constructor or forward code.

    If not declared, the offloader falls back to:
    1. ``_layerwise_offload_blocks_attrs`` on each DiT module class.
    2. Heuristic search for ``layers`` / ``blocks`` / ``h`` attributes.

    Attributes:
        block_attrs: Maps DiT path → tuple of block-list attribute names.
            e.g. ``{"transformer": ("gen_layers",),
                    "transformer.language_model": ("layers",)}``
        offload_submodules: Maps child name → block-list attribute name,
            for large non-DiT submodules within a DiT that should be
            independently offloaded with their own hooks.
            e.g. ``{"context_encoder": "layers"}``
        resident_dit_paths: DiT paths whose leading blocks may be kept on the
            device when ``dlo_resident_layers`` is nonzero. Keeping this
            model-declared avoids applying a consumer-GPU tuning knob to
            auxiliary or dual DiTs unintentionally.
        encoder_block_attrs: Maps encoder paths to rank-local block-list paths.
            These blocks are streamed with ordinary layerwise hooks, never
            with the DiT AllGather group.
    """

    on_demand_component_paths: frozenset[str] = field(default_factory=frozenset)

    block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offload_submodules: dict[str, str] = field(default_factory=dict)
    resident_dit_paths: frozenset[str] = field(default_factory=frozenset)
    encoder_block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)


def get_offload_plan(pipeline: nn.Module) -> OffloadPlan | None:
    """Retrieve the OffloadPlan declared by the pipeline, if any."""
    return getattr(pipeline, "_offload_plan", None)


def supports_mmap_loading(pipeline: nn.Module) -> bool:
    """Whether the pipeline supports mmap weight loading.

    A pipeline supports mmap when either:

    * a module defines ``_remap_ckpt_key``, which maps checkpoint keys to
      model parameter names; or
    * the pipeline explicitly opts in with ``_supports_mmap_loading=True``
      and declares ``weights_sources``.  In that case each source prefix is
      used as the model namespace.

    The explicit opt-in is important: many pipelines declare
    ``weights_sources`` but still require arbitrary loader callbacks and must
    stay on the regular ``load_weights()`` path.

    This check is shared between ``diffusers_loader.py`` (to decide whether
    to skip ``load_weights``) and ``DistributedLayerwiseOffloadBackend.enable()``
    (to decide whether to call ``_load_weights_via_mmap``), ensuring both
    paths use the same gate condition.
    """
    declared_support = getattr(pipeline, "_supports_mmap_loading", None)
    if declared_support is False:
        return False

    has_remap = any(callable(getattr(type(m), "_remap_ckpt_key", None)) for m in pipeline.modules())
    if has_remap:
        return True

    return declared_support is True and bool(getattr(pipeline, "weights_sources", ()))


def has_online_quantization(model: nn.Module) -> bool:
    """Whether any module defers online-quantized weights on ``meta``."""
    return any(getattr(getattr(module, "quant_method", None), "uses_meta_device", False) for module in model.modules())


def should_select_dlo_mmap(
    *,
    mmap_supported: bool,
    dlo_use_allgather: bool,
    has_online_quant: bool,
    tensor_parallel_size: int,
    use_hsdp: bool,
) -> bool:
    """Whether loader and backend should enter DLO's mmap path.

    AllGather preserves its existing fail-closed topology validation inside
    the mmap loader. The extra TP/HSDP gate applies to rank-local mmap only.
    """
    rank_local_compatible = tensor_parallel_size == 1 and not use_hsdp
    return mmap_supported and not has_online_quant and (dlo_use_allgather or rank_local_compatible)
