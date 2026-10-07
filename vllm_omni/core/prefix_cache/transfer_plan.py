# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coalesced copy plans over a legal prefix-cache slot snapshot.

``slots_cpu`` is a flat row-id array in token order. Consecutive tokens keep
consecutive pool rows only where the block table itself is consecutive, so a
save or a hit read is a few long runs plus, at worst, one run per token.
Compiling those runs once makes the structure explicit and, more usefully,
lets the executor choose between one indexed op and a memcpy per run from
measurement instead of assumption.

The compiler is pure: a CPU int64 tensor in, an immutable plan out. It reads
no pool, takes no lock and never touches scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: A run has to be long enough to pay for its own dispatch. Measured on the
#: target host with ``(rows, 2048)`` bf16 rows, a contiguous ``copy_`` beats
#: one ``index_copy_`` over the same rows from roughly 64 rows per run upward;
#: below that the per-call overhead dominates and the indexed op wins. The
#: write path keeps the indexed op under this threshold.
SPAN_COPY_MIN_RUN_ROWS = 64


@dataclass(frozen=True, slots=True)
class CopySpan:
    """One contiguous run: ``length`` rows from ``src`` copied to ``dst``."""

    src: int
    dst: int
    length: int


@dataclass(frozen=True, slots=True)
class TransferPlan:
    """The runs of one slot snapshot, plus the statistics that route on them.

    ``rows`` counts every row the snapshot covers; ``run_rows`` counts only
    rows inside a run of two or more, so ``rows - run_rows`` is the row count
    a run-based executor would have to issue one call each.
    """

    spans: tuple[CopySpan, ...]
    rows: int
    run_rows: int

    @property
    def mean_run(self) -> float:
        """Average rows per span; 0 for an empty plan."""
        return self.rows / len(self.spans) if self.spans else 0.0

    def payload_bytes(self, row_bytes: int) -> int:
        return self.rows * row_bytes

    def coalesces(self, min_run_rows: int = SPAN_COPY_MIN_RUN_ROWS) -> bool:
        """True when a memcpy per run beats one indexed op over the same rows."""
        return 0 < len(self.spans) < self.rows and self.mean_run >= min_run_rows


def _compile(slots: torch.Tensor, *, write: bool) -> TransferPlan:
    """Cut ``slots`` at every break where the next row is not this row + 1.

    Both directions advance the source by one row per position, so a single
    equality on consecutive ids is the whole merge condition.
    """
    n = int(slots.numel())
    if n == 0:
        return TransferPlan((), 0, 0)
    breaks = (slots[1:] - slots[:-1] != 1).nonzero().flatten()
    starts = torch.cat([breaks.new_zeros(1), breaks + 1])
    lengths = torch.cat([breaks, breaks.new_full((1,), n - 1)]) - starts + 1
    head = slots.index_select(0, starts)
    src, dst = (starts, head) if write else (head, starts)
    spans = tuple(
        CopySpan(int(s), int(d), int(length))
        for s, d, length in zip(src.tolist(), dst.tolist(), lengths.tolist())
    )
    return TransferPlan(spans, n, int(lengths[lengths > 1].sum()))


def compile_write_plan(slots: torch.Tensor) -> TransferPlan | None:
    """Plan a save: source row ``i`` lands on pool row ``slots[i]``.

    ``None`` when the snapshot is not strictly increasing. A repeated or
    reordered destination is resolved by the indexed op's audited
    last-writer-wins order, and coalescing must not quietly redefine it.
    """
    if int(slots.numel()) > 1 and bool((slots[1:] <= slots[:-1]).any()):
        return None
    return _compile(slots, write=True)


def compile_read_plan(slots: torch.Tensor) -> TransferPlan:
    """Plan a hit read: destination row ``i`` comes from pool row ``slots[i]``.

    Reading the same row twice is harmless, so any snapshot is plannable.
    """
    return _compile(slots, write=False)
