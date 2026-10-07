# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned CPU block mirror (mirrors vLLM BlockPool)."""

import logging
from dataclasses import dataclass

import torch

from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig, TensorName
from vllm_omni.core.prefix_cache.transfer_plan import compile_write_plan

logger = logging.getLogger(__name__)


@dataclass
class TransferStats:
    """Why each pool write took the path it took. Counters only, no samples."""

    writes: int = 0
    span_writes: int = 0
    indexed_writes: int = 0
    unplannable_writes: int = 0
    write_rows: int = 0
    write_spans: int = 0
    gathers: int = 0
    gather_rows: int = 0


class PrefixBlockPool:
    """Durable CPU mirror of vLLM KV slots, one tensor per name.

    Storage is ``(num_blocks, block_size, feat)`` per key, viewed flat as
    ``(num_slots, feat)`` so vLLM slot ids index rows directly. Write
    access is single-writer (the controller committer thread); readers
    take row views.

    ``_caches`` is not an unbounded store and has no LRU. Dict keys are
    tensor names (``__hidden_states__`` plus mm whose first dim is tokens),
    opened once by ``ensure_key`` and never dropped — a model emits a
    handful of those names, not a per-request set. Each value is a
    fixed tensor sized to the same ``num_blocks`` as the upstream KV
    pool. Rows *are* those kv slots: when vLLM recycles a block, the
    next pool write overwrites that row. Slot reuse is the eviction.
    """

    def __init__(self, config: PrefixCacheConfig):
        self._config = config
        self._caches: dict[TensorName, torch.Tensor] = {}
        self.stats = TransferStats()

    def _alloc(self, dtype: torch.dtype, feat: int) -> torch.Tensor:
        return torch.zeros(
            (self._config.num_blocks, self._config.block_size, feat),
            dtype=dtype,
            device="cpu",
            # Pinning lets device→host overlap compute; unsupported on
            # CPU-only builds where that overlap is off anyway.
            pin_memory=torch.cuda.is_available(),
        )

    def alloc_key(self, key: str, dtype: torch.dtype, feat: int) -> torch.Tensor | None:
        """Allocate storage for a new ``key`` without publishing it (None if
        already open). Pinned allocation is slow; run this unlocked and
        hand the tensor to ``install_key`` under the manager's state lock."""
        if key in self._caches:
            return None
        return self._alloc(dtype, feat)

    def install_key(self, key: str, storage: torch.Tensor) -> None:
        if key in self._caches:
            return
        self._caches[key] = storage
        logger.info("prefix_cache: initialized mirror %s for key %s", list(storage.shape), key)

    def ensure_key(self, key: str, dtype: torch.dtype, feat: int) -> None:
        storage = self.alloc_key(key, dtype, feat)
        if storage is not None:
            self.install_key(key, storage)

    def has_key(self, key: str) -> bool:
        return key in self._caches

    def keys(self) -> set[TensorName]:
        return set(self._caches.keys())

    def _flat(self, key: str) -> torch.Tensor:
        cache = self._caches[key]
        return cache.view(-1, cache.shape[-1])

    def row_width(self, key: str) -> int:
        return int(self._caches[key].shape[-1])

    def row_dtype(self, key: str) -> torch.dtype:
        return self._caches[key].dtype

    def empty_rows(self, key: str, rows: int) -> torch.Tensor:
        """Uninitialised CPU rows shaped like this key's pool rows.

        Lets a caller reserve its own destination and have the gather write
        straight into it instead of gathering a copy and copying again.
        """
        return torch.empty((rows, self.row_width(key)), dtype=self.row_dtype(key))

    def rows_into(self, dst: torch.Tensor, key: str, slots: torch.Tensor) -> torch.Tensor:
        """Gather ``slots`` rows straight into ``dst`` (row-aligned, same length).

        One traversal: ``index_select`` writes into the destination slice.
        The two-step ``rows()`` + ``copy_`` costs a second pass over the same
        bytes for no benefit.
        """
        self.stats.gathers += 1
        self.stats.gather_rows += int(slots.numel())
        return torch.index_select(self._flat(key), 0, self._to_slots(slots), out=dst)

    def rows(self, key: str, slots: torch.Tensor) -> torch.Tensor:
        """Gather rows for non-contiguous slots (returns a copy)."""
        return self.rows_into(self.empty_rows(key, int(slots.numel())), key, slots)

    def write(self, key: str, slots: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Write rows into the pool; caller (committer thread) is the single writer."""
        slots = self._to_slots(slots)
        flat = self._flat(key)
        plan = compile_write_plan(slots)
        self.stats.writes += 1
        self.stats.write_rows += int(slots.numel())
        if plan is None:
            # Repeated or reordered destination: keep the audited
            # last-writer-wins order of the indexed op.
            self.stats.unplannable_writes += 1
        elif plan.coalesces():
            self.stats.span_writes += 1
            self.stats.write_spans += len(plan.spans)
            for span in plan.spans:
                flat[span.dst : span.dst + span.length] = src_cpu[span.src : span.src + span.length]
            return
        self.stats.indexed_writes += 1
        # index_copy_ dispatches to a faster single-dim CPU path than
        # advanced-indexing assignment (aten::index_put_).
        flat.index_copy_(0, slots, src_cpu)

    @staticmethod
    def _to_slots(slots: torch.Tensor) -> torch.Tensor:
        return slots if slots.dtype == torch.int64 else slots.to(torch.int64)
