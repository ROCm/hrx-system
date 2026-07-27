# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang intra-workgroup distribution helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loom.builder import ValueRef
from loom.importers.tilelang.context import (
    TileLangConversionContext,
    TileLangDistributedIndex,
)
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.nodes import node_text
from loom.importers.tilelang.ops.topology import THREAD_AXES, map_thread_axis
from loom.ir import I1, INDEX

DistributedBodyEmitter = Callable[
    [tuple[ValueRef, ...], TileLangConversionContext], None
]

MAX_DISTRIBUTED_UNROLL_LANES = 8


@dataclass(frozen=True, slots=True)
class Distributed1DPlan:
    """Materialized mapping from a 1D TileLang parallel space to workitems."""

    thread_index: ValueRef
    base: ValueRef
    lane_count: int
    extent_integer: int
    thread_count: int


def static_index_value(
    ref: ValueRef,
    context: TileLangConversionContext,
) -> int | None:
    """Returns the integer payload for a cached index constant."""

    for (value_type, value), constant in context.constants.items():
        if value_type == "index" and constant.id == ref.id:
            return int(value)
    return None


def materialize_distributed_1d_plan(
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    extent_integer: int,
    index_name: str,
) -> Distributed1DPlan | None:
    """Materialize the per-workitem chunk plan for a 1D parallel space."""

    thread_count = context.static_topology_extent("threadIdx.x")
    if thread_count is None:
        context.record_blocked(
            node_text(owner),
            "parallel loop requires static threadIdx.x extent",
        )
        return None
    if extent_integer <= 0 or thread_count <= 0:
        context.record_blocked(
            node_text(owner),
            "parallel loop extent and threadIdx.x extent must be positive",
        )
        return None

    thread_index = map_thread_axis(
        axis=THREAD_AXES["threadIdx.x"],
        sources=(),
        context=context,
        converter=converter,
        name="tx",
    )
    if thread_index is None:
        context.record_blocked(
            node_text(owner), "parallel loop thread index is not mapped"
        )
        return None

    lanes_per_thread = (extent_integer + thread_count - 1) // thread_count
    if lanes_per_thread == 1:
        return Distributed1DPlan(
            thread_index=thread_index,
            base=thread_index,
            lane_count=1,
            extent_integer=extent_integer,
            thread_count=thread_count,
        )
    lane_count = context.ensure_constant(
        str(lanes_per_thread),
        "index",
        f"c{lanes_per_thread}",
    )
    lane_base = context.builder.index.mul(
        lhs=thread_index,
        rhs=lane_count,
        results=[INDEX],
        name=context.fresh_name(f"{index_name}_base"),
    )
    return Distributed1DPlan(
        thread_index=thread_index,
        base=lane_base,
        lane_count=lanes_per_thread,
        extent_integer=extent_integer,
        thread_count=thread_count,
    )


def emit_distributed_1d(
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    extent: ValueRef | None,
    extent_integer: int,
    index_name: str,
    emit_body: DistributedBodyEmitter,
) -> bool:
    """Distribute a one-dimensional TileLang parallel index space over x lanes."""

    plan = materialize_distributed_1d_plan(
        owner,
        context,
        converter,
        extent_integer=extent_integer,
        index_name=index_name,
    )
    if plan is None:
        return False

    if plan.lane_count == 1:
        guard = None
        if extent_integer < plan.thread_count:
            extent_value = _materialize_extent(extent, extent_integer, context)
            guard = context.builder.index.cmp(
                predicate="slt",
                lhs=plan.thread_index,
                rhs=extent_value,
                results=[I1],
                name=context.fresh_name(f"{index_name}_active"),
            )
        context.map_distributed_index(
            plan.thread_index,
            TileLangDistributedIndex(
                base=plan.base,
                lane=0,
                lane_count=plan.lane_count,
            ),
        )
        _emit_body(owner, context, (plan.thread_index,), emit_body, guard=guard)
        return True

    if plan.lane_count <= MAX_DISTRIBUTED_UNROLL_LANES:
        _emit_unrolled_lanes(
            owner,
            context,
            plan,
            extent=extent,
            index_name=index_name,
            emit_body=emit_body,
        )
        return True

    lane_count = _materialize_extent(None, plan.lane_count, context)
    zero = context.ensure_constant("0", "index", "c0")
    one = context.ensure_constant("1", "index", "c1")
    body = context.builder.region(args=[(f"{index_name}_lane", INDEX)])
    lane = ValueRef(body.blocks[0].arg_ids[0], context.builder.ir)
    child = context.fork(preview_block=body.blocks[0])
    with context.builder.insertion_block(body.blocks[0]):
        index = child.builder.index.add(
            lhs=plan.base,
            rhs=lane,
            results=[INDEX],
            name=child.fresh_name(index_name),
        )
        child.map_distributed_index(
            index,
            TileLangDistributedIndex(
                base=plan.base,
                lane=-1,
                lane_count=plan.lane_count,
            ),
        )
        guard = None
        if plan.lane_count * plan.thread_count != extent_integer:
            extent_value = _materialize_extent(extent, extent_integer, child)
            guard = child.builder.index.cmp(
                predicate="slt",
                lhs=index,
                rhs=extent_value,
                results=[I1],
                name=child.fresh_name(f"{index_name}_active"),
            )
        _emit_body(owner, child, (index,), emit_body, guard=guard)
        child.builder.scf.yield_()
    context.merge_child_records(child)
    context.builder.scf.for_(
        lower_bound=zero,
        upper_bound=lane_count,
        step=one,
        iter_args=[],
        results=[],
        body=body,
    )
    return True


def _emit_unrolled_lanes(
    owner: object,
    context: TileLangConversionContext,
    plan: Distributed1DPlan,
    *,
    extent: ValueRef | None,
    index_name: str,
    emit_body: DistributedBodyEmitter,
) -> None:
    for lane in range(plan.lane_count):
        index = plan.base
        if lane != 0:
            lane_offset = context.ensure_constant(str(lane), "index", f"c{lane}")
            index = context.builder.index.add(
                lhs=plan.base,
                rhs=lane_offset,
                results=[INDEX],
                name=context.fresh_name(index_name),
            )
        context.map_distributed_index(
            index,
            TileLangDistributedIndex(
                base=plan.base,
                lane=lane,
                lane_count=plan.lane_count,
            ),
        )
        guard = None
        if plan.lane_count * plan.thread_count != plan.extent_integer:
            extent_value = _materialize_extent(extent, plan.extent_integer, context)
            guard = context.builder.index.cmp(
                predicate="slt",
                lhs=index,
                rhs=extent_value,
                results=[I1],
                name=context.fresh_name(f"{index_name}_active"),
            )
        _emit_body(owner, context, (index,), emit_body, guard=guard)


def _materialize_extent(
    extent: ValueRef | None,
    extent_integer: int,
    context: TileLangConversionContext,
) -> ValueRef:
    if extent is not None:
        return extent
    return context.ensure_constant(
        str(extent_integer),
        "index",
        f"c{extent_integer}",
    )


def _emit_body(
    _owner: object,
    context: TileLangConversionContext,
    indices: tuple[ValueRef, ...],
    emit_body: DistributedBodyEmitter,
    *,
    guard: ValueRef | None,
) -> None:
    if guard is None:
        child = context.fork(preview_block=context.builder.ir.insertion_block)
        emit_body(indices, child)
        context.merge_child_records(child)
        return

    then_region = context.builder.region()
    child = context.fork(preview_block=then_region.blocks[0])
    with context.builder.insertion_block(then_region.blocks[0]):
        emit_body(indices, child)
        child.builder.scf.yield_()
    context.merge_child_records(child)
    context.builder.scf.if_(
        condition=guard,
        results=[],
        then_region=then_region,
        else_region=None,
    )
