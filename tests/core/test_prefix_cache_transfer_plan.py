# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Span compilation and execution for prefix-cache slot snapshots.

Two independent oracles, because a round trip alone hides a wrong save paired
with a wrong read:

* mapping oracle — ``expand(plan)`` must equal the row-by-row reference;
* copy oracle — executing the plan must land the same bytes the reference
  row-by-row scatter lands.
"""

import random

import pytest
import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.core.prefix_cache.transfer_plan import (
    SPAN_COPY_MIN_RUN_ROWS,
    CopySpan,
    TransferPlan,
    compile_read_plan,
    compile_write_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def expand(plan: TransferPlan) -> list[tuple[int, int]]:
    """Row-by-row reference for a plan: the (src, dst) pairs it stands for."""
    return [(span.src + i, span.dst + i) for span in plan.spans for i in range(span.length)]


def reference_write(slots: list[int]) -> list[tuple[int, int]]:
    return [(i, slot) for i, slot in enumerate(slots)]


def reference_read(slots: list[int]) -> list[tuple[int, int]]:
    return [(slot, i) for i, slot in enumerate(slots)]


def scatter_into(dst: torch.Tensor, src: torch.Tensor, rows: list[tuple[int, int]]) -> None:
    for src_row, dst_row in rows:
        dst[dst_row] = src[src_row]


def execute_write(flat: torch.Tensor, src: torch.Tensor, plan: TransferPlan) -> None:
    for span in plan.spans:
        flat[span.dst : span.dst + span.length] = src[span.src : span.src + span.length]


def execute_read(pool: torch.Tensor, plan: TransferPlan) -> torch.Tensor:
    out = torch.empty((plan.rows, pool.shape[-1]), dtype=pool.dtype)
    for span in plan.spans:
        out[span.dst : span.dst + span.length] = pool[span.src : span.src + span.length]
    return out


def write_plan(slots: list[int]) -> TransferPlan:
    plan = compile_write_plan(torch.tensor(slots, dtype=torch.long))
    assert plan is not None
    return plan


def test_contiguous_snapshot_is_one_span():
    plan = write_plan([8, 9, 10, 11, 12])
    assert plan.spans == (CopySpan(src=0, dst=8, length=5),)
    assert plan.rows == 5 and plan.run_rows == 5
    assert plan.mean_run == 5.0
    # one span, but too short to pay for its own dispatch
    assert plan.coalesces() is False
    assert write_plan(list(range(SPAN_COPY_MIN_RUN_ROWS))).coalesces()


def test_fragmented_snapshot_splits_without_reordering():
    plan = write_plan([8, 9, 20, 21, 22, 40])
    assert plan.spans == (
        CopySpan(src=0, dst=8, length=2),
        CopySpan(src=2, dst=20, length=3),
        CopySpan(src=5, dst=40, length=1),
    )
    assert plan.run_rows == 5
    assert expand(plan) == reference_write([8, 9, 20, 21, 22, 40])


def test_doc_example_is_unplannable_for_a_write_and_plannable_for_a_read():
    """source rows 0..5 on pool slots 8 9 2 3 4 20 (the §4.3 example).

    A write to those slots is not a monotonic snapshot, so the audited
    last-writer-wins order of the indexed op stays the definition; the plan
    refuses rather than redefining it. The same ids as a hit read carry no
    ordering meaning and compile to exactly those three runs.
    """
    slots = torch.tensor([8, 9, 2, 3, 4, 20], dtype=torch.long)
    assert compile_write_plan(slots) is None
    assert compile_read_plan(slots).spans == (
        CopySpan(src=8, dst=0, length=2),
        CopySpan(src=2, dst=2, length=3),
        CopySpan(src=20, dst=5, length=1),
    )


def test_source_discontinuity_cannot_merge_on_destination_alone():
    # read: destinations 0,1 are contiguous, but the pool rows 3,0 are not.
    plan = compile_read_plan(torch.tensor([3, 0], dtype=torch.long))
    assert plan.spans == (
        CopySpan(src=3, dst=0, length=1),
        CopySpan(src=0, dst=1, length=1),
    )
    assert expand(plan) == reference_read([3, 0])
    # write: a destination jump splits the run even though the source runs on.
    plan = write_plan([3, 4, 5, 9])
    assert plan.spans == (
        CopySpan(src=0, dst=3, length=3),
        CopySpan(src=3, dst=9, length=1),
    )
    assert expand(plan) == reference_write([3, 4, 5, 9])


def test_repeated_or_reordered_destination_is_not_plannable():
    for slots in ([0, 0], [7, 7, 8], [5, 4], [1, 2, 2, 3], [9, 9, 9]):
        assert compile_write_plan(torch.tensor(slots, dtype=torch.long)) is None


def test_empty_and_partial_tail():
    assert compile_write_plan(torch.tensor([], dtype=torch.long)) == TransferPlan((), 0, 0)
    assert compile_write_plan(torch.tensor([], dtype=torch.long)).coalesces() is False
    for rows in (1, 15, 16, 17, 31, 32, 33):
        slots = [4 * i + 1 for i in range(rows)]
        plan = write_plan(slots)
        assert plan.rows == rows
        assert expand(plan) == reference_write(slots)


def test_batch_boundary_does_not_merge_across_a_slot_jump():
    first = [100, 101, 102]
    second = [140, 141, 142]
    plan = write_plan(first + second)
    assert plan.spans == (
        CopySpan(src=0, dst=100, length=3),
        CopySpan(src=3, dst=140, length=3),
    )
    assert expand(plan) == reference_write(first + second)


def test_coalesces_uses_the_measured_run_length():
    short = write_plan([i * 2 for i in range(256)])
    assert short.mean_run == 1.0
    assert short.coalesces() is False

    long = write_plan([i for i in range(256)])
    assert long.coalesces()

    boundary = write_plan([i for i in range(SPAN_COPY_MIN_RUN_ROWS)])
    assert boundary.mean_run == float(SPAN_COPY_MIN_RUN_ROWS)
    assert boundary.coalesces()

    below = write_plan(list(range(SPAN_COPY_MIN_RUN_ROWS - 1)) + [1000])
    assert below.mean_run < SPAN_COPY_MIN_RUN_ROWS
    assert below.coalesces() is False


def test_read_plan_allows_repeats_and_keeps_position_order():
    plan = compile_read_plan(torch.tensor([9, 9, 2, 3, 4], dtype=torch.long))
    assert plan.spans == (
        CopySpan(src=9, dst=0, length=1),
        CopySpan(src=9, dst=1, length=1),
        CopySpan(src=2, dst=2, length=3),
    )
    assert expand(plan) == reference_read([9, 9, 2, 3, 4])


def test_read_plan_accepts_any_permutation():
    slots = [5, 0, 9, 2, 7, 1]
    plan = compile_read_plan(torch.tensor(slots, dtype=torch.long))
    assert expand(plan) == reference_read(slots)
    assert plan.rows == 6


def test_write_plan_matches_reference_bytes():
    torch.manual_seed(11)
    slots = [3, 4, 5, 6, 11, 12, 30, 31, 32, 33, 34]
    plan = write_plan(slots)
    src = torch.arange(len(slots) * 3, dtype=torch.float32).reshape(len(slots), 3)
    span_dst = torch.full((64, 3), -1.0)
    ref_dst = torch.full((64, 3), -1.0)
    execute_write(span_dst, src, plan)
    scatter_into(ref_dst, src, reference_write(slots))
    assert torch.equal(span_dst, ref_dst)


def test_read_plan_matches_reference_bytes():
    slots = [3, 4, 5, 11, 12, 30]
    plan = compile_read_plan(torch.tensor(slots, dtype=torch.long))
    pool = torch.arange(40 * 3, dtype=torch.float32).reshape(40, 3)
    ref = torch.empty((len(slots), 3), dtype=torch.float32)
    scatter_into(ref, pool, reference_read(slots))
    assert torch.equal(execute_read(pool, plan), ref)


def test_randomized_row_oracle():
    """Every plan expands to its reference mapping, and executes to its bytes."""
    sentinel = -777.0
    rng = random.Random(805180668056)
    for _ in range(500):
        rows = rng.randrange(0, 96)
        pool_rows = rng.randrange(rows, rows * 4 + 4)
        # strictly increasing draws make a plannable write snapshot
        slots = sorted(rng.sample(range(pool_rows), rows))
        plan = write_plan(slots)
        assert expand(plan) == reference_write(slots)
        assert sum(span.length for span in plan.spans) == rows

        src = torch.randn((rows, 8))
        span_dst = torch.full((pool_rows, 8), sentinel)
        ref_dst = torch.full((pool_rows, 8), sentinel)
        execute_write(span_dst, src, plan)
        scatter_into(ref_dst, src, reference_write(slots))
        assert torch.equal(span_dst, ref_dst)
        assert int((span_dst == sentinel).sum()) == (pool_rows - rows) * 8

        order = list(range(rows))
        rng.shuffle(order)
        read_slots = [slots[i] for i in order]
        read_plan = compile_read_plan(torch.tensor(read_slots, dtype=torch.long))
        assert expand(read_plan) == reference_read(read_slots)
        pool = torch.randn((pool_rows, 8))
        ref = torch.empty((rows, 8))
        scatter_into(ref, pool, reference_read(read_slots))
        assert torch.equal(execute_read(pool, read_plan), ref)


def test_plannable_write_snapshot_expands_to_every_row_once():
    for slots in ([0], [0, 1, 2], [4, 5, 9, 10, 11, 20], list(range(40))):
        plan = write_plan(slots)
        assert expand(plan) == reference_write(slots)
        assert len({dst for _, dst in expand(plan)}) == len(slots)


def make_pool(num_slots: int, width: int = 4) -> PrefixBlockPool:
    pool = PrefixBlockPool(PrefixCacheConfig(num_blocks=num_slots, block_size=1))
    pool.ensure_key("k", torch.float32, width)
    return pool


def expected_pool(num_slots: int, slots: list[int], src: torch.Tensor) -> torch.Tensor:
    want = torch.zeros((num_slots, src.shape[-1]))
    for i, slot in enumerate(slots):
        want[slot] = src[i]
    return want


def test_pool_write_uses_spans_for_a_long_run_and_matches_the_indexed_op():
    num_slots, rows = 4096, SPAN_COPY_MIN_RUN_ROWS * 2
    slots = [100 + i for i in range(rows)]
    src = torch.randn((rows, 4))
    pool = make_pool(num_slots)
    pool.write("k", torch.tensor(slots, dtype=torch.long), src)
    assert pool.stats.span_writes == 1
    assert pool.stats.indexed_writes == 0
    assert pool.stats.write_spans == 1
    assert torch.equal(pool.rows("k", torch.arange(num_slots)), expected_pool(num_slots, slots, src))


def test_pool_write_keeps_the_indexed_op_when_runs_are_short():
    num_slots, rows = 4096, 256
    slots = [8 * i for i in range(rows)]  # one row per run
    src = torch.randn((rows, 4))
    pool = make_pool(num_slots)
    pool.write("k", torch.tensor(slots, dtype=torch.long), src)
    assert pool.stats.indexed_writes == 1
    assert pool.stats.span_writes == 0
    assert torch.equal(pool.rows("k", torch.arange(num_slots)), expected_pool(num_slots, slots, src))


def test_pool_write_keeps_last_writer_wins_for_repeated_slots():
    slots = [3, 4, 5, 5, 5, 6]
    src = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    pool = make_pool(16)
    pool.write("k", torch.tensor(slots, dtype=torch.long), src)
    assert pool.stats.unplannable_writes == 1
    # the reason is recorded and the audited indexed op is what runs
    assert pool.stats.span_writes == 0 and pool.stats.indexed_writes == 1
    assert torch.equal(pool.rows("k", torch.arange(16)), expected_pool(16, slots, src))


def test_pool_rows_into_fills_the_destination_slice_in_one_gather():
    pool = make_pool(32)
    src = torch.randn((32, 4))
    pool.write("k", torch.arange(32), src)
    slots = torch.tensor([9, 4, 4, 31, 0], dtype=torch.long)
    target = torch.full((8, 4), -1.0)
    returned = pool.rows_into(target[2:7], "k", slots)
    assert returned.data_ptr() == target[2:7].data_ptr()
    assert torch.equal(target[2:7], src[slots])
    assert torch.equal(target[:2], torch.full((2, 4), -1.0))
    assert torch.equal(target[7:], torch.full((1, 4), -1.0))
    assert pool.stats.gathers == 1 and pool.stats.gather_rows == 5


def test_pool_rows_matches_rows_into():
    pool = make_pool(32)
    src = torch.randn((32, 4))
    pool.write("k", torch.arange(32), src)
    slots = torch.tensor([9, 4, 4, 31, 0], dtype=torch.long)
    assert torch.equal(pool.rows("k", slots), src[slots])
