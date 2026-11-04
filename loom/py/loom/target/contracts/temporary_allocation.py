# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lifetime packing for target-Low rule temporaries."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loom.target.contracts.emits import (
    ContractEmit,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterMove,
    EmitRegisterSlice,
    is_per_lane_sequence,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.lower_rule_bindings import (
    _descriptor_operand_is_input,
    _descriptor_operand_is_output,
)
from loom.target.contracts.source import ValueRef


@dataclass(frozen=True, slots=True)
class TemporarySlotPlan:
    """Packed temporary slots assigned at each rule emit."""

    # Temporary definitions introduced by each emit, keyed by authored name.
    definition_slots: tuple[Mapping[str, int], ...]
    # Number of simultaneously addressable rule-local temporary slots.
    temporary_count: int


@dataclass(slots=True)
class _TemporaryInterval:
    """One authored temporary definition and its complete live interval."""

    definition_emit_index: int
    name: str
    start: int
    end: int
    slot: int = -1


def _temporary_name(value_ref: ValueRef | None) -> str | None:
    if value_ref is None or value_ref.kind is not SourceValueKind.TEMPORARY:
        return None
    return value_ref.field


def _temporary_uses(emit: ContractEmit) -> tuple[str, ...]:
    uses: list[str] = []
    if isinstance(emit, EmitDescriptorOp):
        operand_bindings = emit.operands if emit.operands is not None else {}
        for descriptor_operand in emit.descriptor.operands:
            if not _descriptor_operand_is_input(descriptor_operand.role):
                continue
            name = _temporary_name(operand_bindings.get(descriptor_operand.field_name))
            if name is not None:
                uses.append(name)
        result_type_bindings = (
            emit.result_types if emit.result_types is not None else {}
        )
        for descriptor_operand in emit.descriptor.operands:
            if not _descriptor_operand_is_output(descriptor_operand.role):
                continue
            binding = result_type_bindings.get(descriptor_operand.field_name)
            if isinstance(binding, ValueRef):
                name = _temporary_name(binding)
                if name is not None:
                    uses.append(name)
        return tuple(uses)

    if isinstance(emit, EmitRegisterConcat):
        sources = emit.sources
    elif isinstance(emit, (EmitRegisterCopy, EmitRegisterMove, EmitRegisterSlice)):
        sources = (emit.source,)
    else:
        raise TypeError(f"unsupported contract emit type: {type(emit).__name__}")
    for source in sources:
        name = _temporary_name(source)
        if name is not None:
            uses.append(name)
    if isinstance(emit.result_type, ValueRef):
        name = _temporary_name(emit.result_type)
        if name is not None:
            uses.append(name)
    return tuple(uses)


def _temporary_definitions(emit: ContractEmit) -> tuple[str, ...]:
    if isinstance(emit, EmitDescriptorOp):
        result_bindings = emit.results if emit.results is not None else {}
        definitions: list[str] = []
        for descriptor_operand in emit.descriptor.operands:
            if not _descriptor_operand_is_output(descriptor_operand.role):
                continue
            name = _temporary_name(result_bindings.get(descriptor_operand.field_name))
            if name is not None:
                definitions.append(name)
        return tuple(definitions)
    name = _temporary_name(emit.result)
    return () if name is None else (name,)


def allocate_temporary_slots(
    emits: Sequence[ContractEmit], *, sequence_carried: Sequence[str] = ()
) -> TemporarySlotPlan:
    """Packs nonoverlapping authored temporary lifetimes into shared slots."""

    sequence_start = next(
        (
            emit_index
            for emit_index, emit in enumerate(emits)
            if is_per_lane_sequence(emit)
        ),
        None,
    )
    sequence_end = 2 * len(emits) - 1
    intervals: list[_TemporaryInterval] = []
    active_definitions: dict[str, int] = {}
    definition_indices: list[dict[str, int]] = []
    for emit_index, emit in enumerate(emits):
        use_position = 2 * emit_index
        for name in _temporary_uses(emit):
            interval_index = active_definitions.get(name)
            if interval_index is None:
                raise ValueError(f"temporary '{name}' is used before its definition")
            interval = intervals[interval_index]
            use_end = use_position
            # A per-lane sequence executes its complete authored tail once per
            # lane. Values produced by setup before that tail remain inputs to
            # every iteration even when their final authored use appears in
            # the first sequence emit.
            if (
                sequence_start is not None
                and emit_index >= sequence_start
                and interval.definition_emit_index < sequence_start
            ):
                use_end = sequence_end
            interval.end = max(interval.end, use_end)

        emit_definition_indices: dict[str, int] = {}
        definition_position = use_position + 1
        for name in _temporary_definitions(emit):
            interval_index = len(intervals)
            intervals.append(
                _TemporaryInterval(
                    definition_emit_index=emit_index,
                    name=name,
                    start=definition_position,
                    end=definition_position,
                )
            )
            active_definitions[name] = interval_index
            emit_definition_indices[name] = interval_index
        definition_indices.append(emit_definition_indices)

        if isinstance(emit, EmitRegisterMove):
            active_definitions.pop(emit.source.field, None)

    active_intervals: list[tuple[int, int, int]] = []
    free_slots: list[int] = []
    carried_slots = {name: index for index, name in enumerate(sequence_carried)}
    next_slot = len(carried_slots)
    for interval_index, interval in sorted(
        enumerate(intervals), key=lambda item: (item[1].start, item[0])
    ):
        if interval.name in carried_slots:
            interval.slot = carried_slots[interval.name]
            continue
        while active_intervals and active_intervals[0][0] < interval.start:
            _, slot, _ = heapq.heappop(active_intervals)
            heapq.heappush(free_slots, slot)
        if free_slots:
            interval.slot = heapq.heappop(free_slots)
        else:
            interval.slot = next_slot
            next_slot += 1
        heapq.heappush(active_intervals, (interval.end, interval.slot, interval_index))

    definition_slots = tuple(
        {
            name: intervals[interval_index].slot
            for name, interval_index in emit_definitions.items()
        }
        for emit_definitions in definition_indices
    )
    return TemporarySlotPlan(
        definition_slots=definition_slots,
        temporary_count=next_slot,
    )
