# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU descriptor-derived wait-packet tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_planning_table_inputs import (  # noqa: E402
    AmdgpuPlanningTableInputs,
    load_amdgpu_planning_table_inputs,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _COUNTER_ALU,
    _COUNTER_ASYNC,
    _COUNTER_LDS,
    _COUNTER_SMEM,
    _COUNTER_TENSOR,
    _COUNTER_VMEM_LOAD,
    _COUNTER_VMEM_STORE,
    _COUNTER_X,
    _WAIT_COUNTER_ALU_ENCODING_ID,
    _WAIT_COUNTER_ASYNC_ENCODING_ID,
    _WAIT_COUNTER_LDS_ENCODING_ID,
    _WAIT_COUNTER_LGKM_ENCODING_ID,
    _WAIT_COUNTER_SMEM_ENCODING_ID,
    _WAIT_COUNTER_TENSOR_ENCODING_ID,
    _WAIT_COUNTER_VMEM_ENCODING_ID,
    _WAIT_COUNTER_VMEM_LOAD_ENCODING_ID,
    _WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    _WAIT_COUNTER_X_ENCODING_ID,
    amdgpu_descriptor_ref_keys,
)
from loom.target.arch.amdgpu.names import amdgpu_c_identifier_fragment  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AMDGPU_PROCESSOR_INFOS,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    amdgpu_descriptor_set_ordinal,
    amdgpu_target_descriptor_set_key,
    sorted_target_infos,
)
from loom.target.low_descriptors import (  # noqa: E402
    Descriptor,
    DescriptorSet,
    EffectFlag,
    EffectKind,
    HazardKind,
    Immediate,
    ImmediateKind,
    MemorySpace,
    target_relative_name,
)

_UINT16_MAX = 0xFFFF
_WAIT_PACKET_IMMEDIATE_CAPACITY = 4
_WAIT_COUNTER_MASK_COUNT = 1 << _COUNTER_X
_WAIT_COUNTER_ALU_SCHEDULING_BITS = AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR | AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR
_DEPENDENCY_MEMORY_SPACES = frozenset(
    (
        MemorySpace.GENERIC,
        MemorySpace.GLOBAL,
        MemorySpace.STACK,
        MemorySpace.WORKGROUP,
    )
)


@dataclass(frozen=True, slots=True)
class _WaitPacketImmediateRow:
    descriptor_key: str
    descriptor_immediate_index: int
    field_name: str
    counter_mask: int
    no_wait_value: int


@dataclass(frozen=True, slots=True)
class _WaitPacketDescriptorRow:
    descriptor_set_key: str
    descriptor_set_ordinal: int
    descriptor_key: str
    descriptor_ref: str
    counter_mask: int
    counter_count: int
    immediate_start: int
    immediate_count: int


@dataclass(frozen=True, slots=True)
class _WaitPacketDescriptorRange:
    descriptor_set_key: str
    descriptor_set_ordinal: int
    first_descriptor: int
    descriptor_count: int
    first_descriptor_lookup: int
    descriptor_lookup_count: int
    max_descriptor_immediate_count: int


@dataclass(frozen=True, slots=True)
class _WaitPacketSelectionRow:
    descriptor_set_key: str
    descriptor_set_ordinal: int
    counter_mask: int
    descriptor_index: int
    covered_counter_mask: int


@dataclass(frozen=True, slots=True)
class _WaitPacketTables:
    descriptor_rows: tuple[_WaitPacketDescriptorRow, ...]
    immediate_rows: tuple[_WaitPacketImmediateRow, ...]
    range_rows: tuple[_WaitPacketDescriptorRange, ...]
    descriptor_lookup_rows: tuple[int, ...]
    selection_rows: tuple[_WaitPacketSelectionRow, ...]


def _descriptor_ref_constant_name(key: str) -> str:
    ref_name = amdgpu_c_identifier_fragment(target_relative_name("amdgpu", key))
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{ref_name}"


def _counter_mask(counter_id: int) -> int:
    if counter_id < _COUNTER_VMEM_LOAD or counter_id > _COUNTER_X:
        raise ValueError(f"unknown AMDGPU wait counter id {counter_id}")
    return 1 << (counter_id - 1)


_WAIT_COUNTER_READ_MASK = _counter_mask(_COUNTER_VMEM_LOAD) | _counter_mask(_COUNTER_LDS) | _counter_mask(_COUNTER_SMEM) | _counter_mask(_COUNTER_TENSOR) | _counter_mask(_COUNTER_ASYNC)
_WAIT_COUNTER_WRITE_MASK = _counter_mask(_COUNTER_VMEM_STORE) | _counter_mask(_COUNTER_LDS) | _counter_mask(_COUNTER_SMEM) | _counter_mask(_COUNTER_TENSOR) | _counter_mask(_COUNTER_ASYNC)


_IMMEDIATE_COUNTER_MASKS = {
    _WAIT_COUNTER_VMEM_ENCODING_ID: _counter_mask(_COUNTER_VMEM_LOAD) | _counter_mask(_COUNTER_VMEM_STORE),
    _WAIT_COUNTER_LGKM_ENCODING_ID: _counter_mask(_COUNTER_LDS) | _counter_mask(_COUNTER_SMEM),
    _WAIT_COUNTER_VMEM_LOAD_ENCODING_ID: _counter_mask(_COUNTER_VMEM_LOAD),
    _WAIT_COUNTER_VMEM_STORE_ENCODING_ID: _counter_mask(_COUNTER_VMEM_STORE),
    _WAIT_COUNTER_LDS_ENCODING_ID: _counter_mask(_COUNTER_LDS),
    _WAIT_COUNTER_SMEM_ENCODING_ID: _counter_mask(_COUNTER_SMEM),
    _WAIT_COUNTER_ALU_ENCODING_ID: _counter_mask(_COUNTER_ALU),
    _WAIT_COUNTER_TENSOR_ENCODING_ID: _counter_mask(_COUNTER_TENSOR),
    _WAIT_COUNTER_ASYNC_ENCODING_ID: _counter_mask(_COUNTER_ASYNC),
    _WAIT_COUNTER_X_ENCODING_ID: _counter_mask(_COUNTER_X),
}

_COUNTER_MASK_EXPR_TERMS = (
    (_counter_mask(_COUNTER_VMEM_LOAD), "LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD"),
    (
        _counter_mask(_COUNTER_VMEM_STORE),
        "LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE",
    ),
    (_counter_mask(_COUNTER_LDS), "LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS"),
    (_counter_mask(_COUNTER_SMEM), "LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM"),
    (_counter_mask(_COUNTER_ALU), "LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU"),
    (_counter_mask(_COUNTER_TENSOR), "LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR"),
    (_counter_mask(_COUNTER_ASYNC), "LOOM_AMDGPU_WAIT_COUNTER_MASK_ASYNC"),
    (_counter_mask(_COUNTER_X), "LOOM_AMDGPU_WAIT_COUNTER_MASK_X"),
)


def _counter_mask_expr(counter_mask: int) -> str:
    if counter_mask == 0:
        return "0u"
    remaining_mask = counter_mask
    terms: list[str] = []
    for mask, expr in _COUNTER_MASK_EXPR_TERMS:
        if (counter_mask & mask) == 0:
            continue
        terms.append(expr)
        remaining_mask &= ~mask
    if remaining_mask != 0:
        raise ValueError(f"unknown AMDGPU wait counter mask bits 0x{remaining_mask:x}")
    return " | ".join(terms)


def _descriptor_counter_mask(descriptor: Descriptor) -> int:
    counter_mask = 0
    for effect in descriptor.effects:
        if effect.kind is not EffectKind.COUNTER:
            continue
        if effect.counter_id == 0:
            continue
        counter_mask |= _counter_mask(effect.counter_id)
    return counter_mask


def _immediate_counter_mask(immediate: Immediate) -> int:
    try:
        return _IMMEDIATE_COUNTER_MASKS[immediate.encoding_id]
    except KeyError as exc:
        raise ValueError(f"unsupported AMDGPU wait immediate encoding id {immediate.encoding_id}") from exc


def _validate_uint16(owner: str, field_name: str, value: int) -> None:
    if value < 0 or value > _UINT16_MAX:
        raise ValueError(f"{owner} has {field_name} {value}, expected uint16_t")


def _descriptor_wait_packet_rows(
    descriptor: Descriptor,
    descriptor_ref_key_set: set[str],
    immediate_start: int,
) -> tuple[_WaitPacketDescriptorRow | None, tuple[_WaitPacketImmediateRow, ...]]:
    counter_mask = _descriptor_counter_mask(descriptor)
    if counter_mask == 0:
        return None, ()
    owner = f"AMDGPU wait descriptor '{descriptor.key}'"
    if descriptor.key not in descriptor_ref_key_set:
        raise ValueError(f"{owner} requires a descriptor ref")
    if len(descriptor.immediates) > _WAIT_PACKET_IMMEDIATE_CAPACITY:
        raise ValueError(f"{owner} has {len(descriptor.immediates)} immediates, expected 0..{_WAIT_PACKET_IMMEDIATE_CAPACITY}")
    _validate_uint16(owner, "immediate start", immediate_start)

    immediate_rows: list[_WaitPacketImmediateRow] = []
    mapped_counter_mask = 0
    for descriptor_immediate_index, immediate in enumerate(descriptor.immediates):
        immediate_owner = f"{owner} immediate '{immediate.field_name}'"
        if immediate.kind is not ImmediateKind.UNSIGNED:
            raise ValueError(f"{immediate_owner} must be unsigned")
        immediate_counter_mask = _immediate_counter_mask(immediate) & counter_mask
        if immediate_counter_mask == 0:
            raise ValueError(f"{immediate_owner} does not map to any descriptor counter effect")
        _validate_uint16(immediate_owner, "no-wait value", immediate.unsigned_max)
        immediate_rows.append(
            _WaitPacketImmediateRow(
                descriptor_key=descriptor.key,
                descriptor_immediate_index=descriptor_immediate_index,
                field_name=immediate.field_name,
                counter_mask=immediate_counter_mask,
                no_wait_value=immediate.unsigned_max,
            )
        )
        mapped_counter_mask |= immediate_counter_mask
    if mapped_counter_mask != counter_mask:
        raise ValueError(f"{owner} maps counter mask 0x{mapped_counter_mask:x} but advertises 0x{counter_mask:x}")

    descriptor_row = _WaitPacketDescriptorRow(
        descriptor_set_key="",
        descriptor_set_ordinal=0,
        descriptor_key=descriptor.key,
        descriptor_ref=_descriptor_ref_constant_name(descriptor.key),
        counter_mask=counter_mask,
        counter_count=counter_mask.bit_count(),
        immediate_start=immediate_start,
        immediate_count=len(immediate_rows),
    )
    return descriptor_row, tuple(immediate_rows)


def _descriptor_set_wait_packet_rows(
    descriptor_set: DescriptorSet,
    descriptor_set_ordinal: int,
    descriptor_ref_key_set: set[str],
    first_descriptor: int,
    first_immediate: int,
) -> tuple[
    tuple[_WaitPacketDescriptorRow, ...],
    tuple[_WaitPacketImmediateRow, ...],
    tuple[int, ...],
    _WaitPacketDescriptorRange,
]:
    descriptor_rows: list[_WaitPacketDescriptorRow] = []
    immediate_rows: list[_WaitPacketImmediateRow] = []
    descriptor_lookup_rows = [0] * len(descriptor_set.descriptors)
    for descriptor_ordinal, descriptor in enumerate(descriptor_set.descriptors):
        descriptor_row, descriptor_immediates = _descriptor_wait_packet_rows(
            descriptor,
            descriptor_ref_key_set,
            first_immediate + len(immediate_rows),
        )
        if descriptor_row is None:
            continue
        local_descriptor_index = len(descriptor_rows)
        descriptor_lookup_rows[descriptor_ordinal] = local_descriptor_index + 1
        descriptor_rows.append(
            _WaitPacketDescriptorRow(
                descriptor_set_key=descriptor_set.key,
                descriptor_set_ordinal=descriptor_set_ordinal,
                descriptor_key=descriptor_row.descriptor_key,
                descriptor_ref=descriptor_row.descriptor_ref,
                counter_mask=descriptor_row.counter_mask,
                counter_count=descriptor_row.counter_count,
                immediate_start=descriptor_row.immediate_start,
                immediate_count=descriptor_row.immediate_count,
            )
        )
        immediate_rows.extend(descriptor_immediates)

    descriptor_count = len(descriptor_rows)
    _validate_uint16(descriptor_set.key, "first descriptor", first_descriptor)
    _validate_uint16(descriptor_set.key, "descriptor count", descriptor_count)
    _validate_uint16(descriptor_set.key, "first immediate", first_immediate)
    _validate_uint16(descriptor_set.key, "immediate count", len(immediate_rows))
    _validate_uint16(descriptor_set.key, "descriptor lookup count", len(descriptor_lookup_rows))
    max_descriptor_immediate_count = max(
        (row.immediate_count for row in descriptor_rows),
        default=0,
    )
    range_row = _WaitPacketDescriptorRange(
        descriptor_set_key=descriptor_set.key,
        descriptor_set_ordinal=descriptor_set_ordinal,
        first_descriptor=first_descriptor,
        descriptor_count=descriptor_count,
        first_descriptor_lookup=0,
        descriptor_lookup_count=len(descriptor_lookup_rows),
        max_descriptor_immediate_count=max_descriptor_immediate_count,
    )
    return tuple(descriptor_rows), tuple(immediate_rows), tuple(descriptor_lookup_rows), range_row


def _best_wait_packet_descriptor_selection(
    descriptor_rows: Sequence[_WaitPacketDescriptorRow],
    counter_mask: int,
) -> tuple[int, int]:
    best_descriptor_index = 0
    best_covered_counter_mask = 0
    best_covered_count = 0
    best_extra_count = _WAIT_COUNTER_MASK_COUNT
    best_immediate_count = _WAIT_PACKET_IMMEDIATE_CAPACITY + 1
    for descriptor_index, descriptor in enumerate(descriptor_rows):
        covered_counter_mask = descriptor.counter_mask & counter_mask
        if covered_counter_mask == 0:
            continue
        covered_count = covered_counter_mask.bit_count()
        extra_count = descriptor.counter_count - covered_count
        if (
            best_covered_counter_mask == 0
            or covered_count > best_covered_count
            or (covered_count == best_covered_count and extra_count < best_extra_count)
            or (covered_count == best_covered_count and extra_count == best_extra_count and descriptor.immediate_count < best_immediate_count)
        ):
            best_descriptor_index = descriptor_index
            best_covered_counter_mask = covered_counter_mask
            best_covered_count = covered_count
            best_extra_count = extra_count
            best_immediate_count = descriptor.immediate_count
    return best_descriptor_index, best_covered_counter_mask


def _descriptor_set_wait_packet_selection_rows(
    descriptor_set_key: str,
    descriptor_set_ordinal: int,
    descriptor_rows: Sequence[_WaitPacketDescriptorRow],
) -> tuple[_WaitPacketSelectionRow, ...]:
    selection_rows: list[_WaitPacketSelectionRow] = []
    for counter_mask in range(_WAIT_COUNTER_MASK_COUNT):
        descriptor_index, covered_counter_mask = _best_wait_packet_descriptor_selection(
            descriptor_rows,
            counter_mask,
        )
        selection_rows.append(
            _WaitPacketSelectionRow(
                descriptor_set_key=descriptor_set_key,
                descriptor_set_ordinal=descriptor_set_ordinal,
                counter_mask=counter_mask,
                descriptor_index=descriptor_index,
                covered_counter_mask=covered_counter_mask,
            )
        )
    return tuple(selection_rows)


def _descriptor_set_processor_scheduling_bits(descriptor_set_key: str) -> int:
    scheduling_bits = 0
    processors_by_name = {processor_info.processor: processor_info for processor_info in AMDGPU_PROCESSOR_INFOS}
    for target_info in sorted_target_infos():
        processor_info = processors_by_name[target_info.processor]
        if amdgpu_target_descriptor_set_key(target_info, processor_info) == descriptor_set_key:
            scheduling_bits |= processor_info.features.scheduling
    return scheduling_bits


def _descriptor_set_required_counter_mask(
    descriptor_set: DescriptorSet,
    processor_scheduling_bits: int,
) -> int:
    schedule_classes = {schedule_class.name: schedule_class for schedule_class in descriptor_set.schedule_classes}
    required_counter_mask = 0
    for descriptor in descriptor_set.descriptors:
        try:
            schedule_class = schedule_classes[descriptor.schedule_class]
        except KeyError as exc:
            raise ValueError(f"AMDGPU descriptor '{descriptor.key}' references missing schedule class '{descriptor.schedule_class}'") from exc
        hazard_counter_mask = 0
        for hazard in schedule_class.hazards:
            if hazard.kind is not HazardKind.WAIT_COUNTER:
                continue
            if hazard.counter_id is None:
                raise ValueError(f"AMDGPU descriptor '{descriptor.key}' schedule class '{schedule_class.name}' has a wait-counter hazard without a counter id")
            hazard_counter_mask |= _counter_mask(hazard.counter_id)
        for effect in descriptor.effects:
            if EffectFlag.DEPENDENCY not in effect.flags or effect.memory_space not in _DEPENDENCY_MEMORY_SPACES or effect.kind not in (EffectKind.READ, EffectKind.WRITE):
                continue
            if effect.counter_id != 0:
                effect_counter_mask = _counter_mask(effect.counter_id)
                if effect_counter_mask & ~hazard_counter_mask:
                    raise ValueError(f"AMDGPU descriptor '{descriptor.key}' dependency effect counter {effect.counter_id} is not present in schedule class '{schedule_class.name}' hazards")
            else:
                effect_counter_mask = hazard_counter_mask & (_WAIT_COUNTER_READ_MASK if effect.kind is EffectKind.READ else _WAIT_COUNTER_WRITE_MASK)
                if effect_counter_mask == 0:
                    effect_kind_name = "read" if effect.kind is EffectKind.READ else "write"
                    raise ValueError(f"AMDGPU descriptor '{descriptor.key}' dependency {effect_kind_name} effect has no counter in schedule class '{schedule_class.name}' hazards")
            required_counter_mask |= effect_counter_mask
    if processor_scheduling_bits & _WAIT_COUNTER_ALU_SCHEDULING_BITS:
        required_counter_mask |= _counter_mask(_COUNTER_ALU)
    return required_counter_mask


def _validate_descriptor_set_wait_packet_coverage(
    descriptor_set: DescriptorSet,
    descriptor_rows: Sequence[_WaitPacketDescriptorRow],
    selection_rows: Sequence[_WaitPacketSelectionRow],
    *,
    processor_scheduling_bits: int | None = None,
) -> None:
    if processor_scheduling_bits is None:
        processor_scheduling_bits = _descriptor_set_processor_scheduling_bits(descriptor_set.key)
    required_counter_mask = _descriptor_set_required_counter_mask(descriptor_set, processor_scheduling_bits)
    available_counter_mask = 0
    for descriptor_row in descriptor_rows:
        available_counter_mask |= descriptor_row.counter_mask
    missing_counter_mask = required_counter_mask & ~available_counter_mask
    if missing_counter_mask != 0:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' requires wait counter mask 0x{required_counter_mask:x} but packet descriptors only cover 0x{available_counter_mask:x}")

    if len(selection_rows) != _WAIT_COUNTER_MASK_COUNT:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' has {len(selection_rows)} wait selections, expected {_WAIT_COUNTER_MASK_COUNT}")
    for requested_counter_mask in range(_WAIT_COUNTER_MASK_COUNT):
        if requested_counter_mask & ~required_counter_mask:
            continue
        remaining_counter_mask = requested_counter_mask
        while remaining_counter_mask != 0:
            selection = selection_rows[remaining_counter_mask]
            if selection.covered_counter_mask == 0:
                raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' cannot materialize wait counter mask 0x{requested_counter_mask:x}")
            remaining_counter_mask &= ~selection.covered_counter_mask


def _validate_wait_packet_tables(tables: _WaitPacketTables) -> None:
    descriptor_count = len(tables.descriptor_rows)
    immediate_count = len(tables.immediate_rows)
    descriptor_lookup_count = len(tables.descriptor_lookup_rows)
    for row in tables.descriptor_rows:
        owner = f"AMDGPU wait descriptor '{row.descriptor_key}'"
        if row.immediate_start > immediate_count or row.immediate_count > immediate_count - row.immediate_start:
            raise ValueError(f"{owner} immediate range is out of bounds")
    for row in tables.range_rows:
        owner = f"AMDGPU wait descriptor-set range '{row.descriptor_set_key}'"
        if row.first_descriptor > descriptor_count or row.descriptor_count > descriptor_count - row.first_descriptor:
            raise ValueError(f"{owner} descriptor range is out of bounds")
        if row.first_descriptor_lookup > descriptor_lookup_count or row.descriptor_lookup_count > descriptor_lookup_count - row.first_descriptor_lookup:
            raise ValueError(f"{owner} descriptor lookup range is out of bounds")
        for descriptor_ordinal, descriptor_index_plus_one in enumerate(tables.descriptor_lookup_rows[row.first_descriptor_lookup : row.first_descriptor_lookup + row.descriptor_lookup_count]):
            lookup_owner = f"{owner} descriptor ordinal {descriptor_ordinal}"
            if descriptor_index_plus_one == 0:
                continue
            descriptor_index = descriptor_index_plus_one - 1
            if descriptor_index >= row.descriptor_count:
                raise ValueError(f"{lookup_owner} descriptor index is out of bounds")
    range_rows_by_ordinal = {row.descriptor_set_ordinal: row for row in tables.range_rows}
    expected_selection_count = len(tables.range_rows) * _WAIT_COUNTER_MASK_COUNT
    if len(tables.selection_rows) != expected_selection_count:
        raise ValueError(f"AMDGPU wait selection table has {len(tables.selection_rows)} rows, expected {expected_selection_count}")
    seen_selection_keys: set[tuple[int, int]] = set()
    for row in tables.selection_rows:
        owner = f"AMDGPU wait selection '{row.descriptor_set_key}' mask 0x{row.counter_mask:x}"
        selection_key = (row.descriptor_set_ordinal, row.counter_mask)
        if selection_key in seen_selection_keys:
            raise ValueError(f"{owner} duplicates a descriptor-set/mask row")
        seen_selection_keys.add(selection_key)
        if row.counter_mask < 0 or row.counter_mask >= _WAIT_COUNTER_MASK_COUNT:
            raise ValueError(f"{owner} has out-of-range counter mask")
        if row.covered_counter_mask & ~row.counter_mask:
            raise ValueError(f"{owner} covers counters outside its input mask")
        range_row = range_rows_by_ordinal.get(row.descriptor_set_ordinal)
        if range_row is None:
            raise ValueError(f"{owner} references missing descriptor-set range")
        if row.covered_counter_mask == 0:
            continue
        if row.descriptor_index >= range_row.descriptor_count:
            raise ValueError(f"{owner} descriptor index is out of bounds")
        descriptor_row = tables.descriptor_rows[range_row.first_descriptor + row.descriptor_index]
        if row.covered_counter_mask != (descriptor_row.counter_mask & row.counter_mask):
            raise ValueError(f"{owner} covered mask does not match descriptor row")


def _materialize_wait_packet_tables(
    inputs: AmdgpuPlanningTableInputs,
) -> _WaitPacketTables:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    descriptor_rows: list[_WaitPacketDescriptorRow] = []
    immediate_rows: list[_WaitPacketImmediateRow] = []
    range_rows: list[_WaitPacketDescriptorRange] = []
    descriptor_lookup_rows: list[int] = []
    selection_rows: list[_WaitPacketSelectionRow] = []
    for info in inputs.descriptor_set_infos:
        descriptor_set = inputs.descriptor_sets_by_key[info.key]
        descriptor_set_ordinal = amdgpu_descriptor_set_ordinal(info.key)
        if descriptor_set_ordinal >= AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"AMDGPU descriptor set '{info.key}' has invalid ordinal {descriptor_set_ordinal}")
        set_descriptor_rows, set_immediate_rows, set_descriptor_lookup_rows, range_row = _descriptor_set_wait_packet_rows(
            descriptor_set,
            descriptor_set_ordinal,
            descriptor_ref_key_set,
            len(descriptor_rows),
            len(immediate_rows),
        )
        descriptor_rows.extend(set_descriptor_rows)
        immediate_rows.extend(set_immediate_rows)
        range_rows.append(
            _WaitPacketDescriptorRange(
                descriptor_set_key=range_row.descriptor_set_key,
                descriptor_set_ordinal=range_row.descriptor_set_ordinal,
                first_descriptor=range_row.first_descriptor,
                descriptor_count=range_row.descriptor_count,
                first_descriptor_lookup=len(descriptor_lookup_rows),
                descriptor_lookup_count=range_row.descriptor_lookup_count,
                max_descriptor_immediate_count=range_row.max_descriptor_immediate_count,
            )
        )
        descriptor_lookup_rows.extend(set_descriptor_lookup_rows)
        set_selection_rows = _descriptor_set_wait_packet_selection_rows(
            descriptor_set.key,
            descriptor_set_ordinal,
            set_descriptor_rows,
        )
        _validate_descriptor_set_wait_packet_coverage(descriptor_set, set_descriptor_rows, set_selection_rows)
        selection_rows.extend(set_selection_rows)
    tables = _WaitPacketTables(
        descriptor_rows=tuple(descriptor_rows),
        immediate_rows=tuple(immediate_rows),
        range_rows=tuple(range_rows),
        descriptor_lookup_rows=tuple(descriptor_lookup_rows),
        selection_rows=tuple(selection_rows),
    )
    _validate_wait_packet_tables(tables)
    return tables


def _generated_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.descriptors.amdgpu_wait_packet_tables"),
        ),
        "",
    ]


def _descriptor_row_initializer(row: _WaitPacketDescriptorRow) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR(",
            f"    {row.descriptor_ref}, {_counter_mask_expr(row.counter_mask)},",
            f"    {row.counter_count}, {row.immediate_start},",
            f"    {row.immediate_count})",
        ]
    )


def _immediate_row_initializer(row: _WaitPacketImmediateRow) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_WAIT_PACKET_IMMEDIATE(",
            f"    {row.descriptor_immediate_index}, {_c_string_arg(row.field_name)},",
            f"    {_counter_mask_expr(row.counter_mask)}, {row.no_wait_value})",
        ]
    )


def _range_row_initializer(row: _WaitPacketDescriptorRange) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_RANGE(",
            f"    {row.descriptor_set_ordinal}, {row.first_descriptor},",
            f"    {row.descriptor_count}, {row.first_descriptor_lookup},",
            f"    {row.descriptor_lookup_count}, {row.max_descriptor_immediate_count})",
        ]
    )


def _descriptor_lookup_row_initializer(descriptor_index_plus_one: int) -> str:
    return f"LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_LOOKUP({descriptor_index_plus_one})"


def _selection_row_initializer(row: _WaitPacketSelectionRow) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_WAIT_PACKET_SELECTION(",
            f"    {row.descriptor_set_ordinal}, {_counter_mask_expr(row.counter_mask)},",
            f"    {row.descriptor_index}, {_counter_mask_expr(row.covered_counter_mask)})",
        ]
    )


def _emit_descriptor_rows(tables: _WaitPacketTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_descriptor_row_initializer(row) for row in tables.descriptor_rows),
            ]
        )
        + "\n"
    )


def _emit_immediate_rows(tables: _WaitPacketTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_immediate_row_initializer(row) for row in tables.immediate_rows),
            ]
        )
        + "\n"
    )


def _emit_range_rows(tables: _WaitPacketTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_range_row_initializer(row) for row in tables.range_rows),
            ]
        )
        + "\n"
    )


def _emit_descriptor_lookup_rows(tables: _WaitPacketTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_descriptor_lookup_row_initializer(row) for row in tables.descriptor_lookup_rows),
            ]
        )
        + "\n"
    )


def _emit_selection_rows(tables: _WaitPacketTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_selection_row_initializer(row) for row in tables.selection_rows),
            ]
        )
        + "\n"
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def generate_wait_packet_table_outputs(
    inputs: AmdgpuPlanningTableInputs,
    *,
    descriptor_rows_path: Path | None = None,
    immediate_rows_path: Path | None = None,
    descriptor_ranges_path: Path | None = None,
    descriptor_lookups_path: Path | None = None,
    selection_rows_path: Path | None = None,
) -> None:
    """Generates the requested wait-packet table fragments."""

    tables = _materialize_wait_packet_tables(inputs)
    if descriptor_rows_path is not None:
        _write_output(descriptor_rows_path, _emit_descriptor_rows(tables))
    if immediate_rows_path is not None:
        _write_output(immediate_rows_path, _emit_immediate_rows(tables))
    if descriptor_ranges_path is not None:
        _write_output(descriptor_ranges_path, _emit_range_rows(tables))
    if descriptor_lookups_path is not None:
        _write_output(
            descriptor_lookups_path,
            _emit_descriptor_lookup_rows(tables),
        )
    if selection_rows_path is not None:
        _write_output(selection_rows_path, _emit_selection_rows(tables))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU descriptor-derived wait-packet table fragments.")
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument(
        "--descriptor-rows",
        type=Path,
        help="Generated wait-packet descriptor row fragment path.",
    )
    parser.add_argument(
        "--immediate-rows",
        type=Path,
        help="Generated wait-packet immediate row fragment path.",
    )
    parser.add_argument(
        "--descriptor-ranges",
        type=Path,
        help="Generated wait-packet descriptor-set range fragment path.",
    )
    parser.add_argument(
        "--descriptor-lookups",
        type=Path,
        help="Generated wait-packet descriptor lookup fragment path.",
    )
    parser.add_argument(
        "--selection-rows",
        type=Path,
        help="Generated wait-packet counter-mask selection row fragment path.",
    )
    args = parser.parse_args(argv)
    if args.descriptor_rows is None and args.immediate_rows is None and args.descriptor_ranges is None and args.descriptor_lookups is None and args.selection_rows is None:
        parser.error("at least one output path is required")

    generate_wait_packet_table_outputs(
        load_amdgpu_planning_table_inputs(args.isa_xml, {}),
        descriptor_rows_path=args.descriptor_rows,
        immediate_rows_path=args.immediate_rows,
        descriptor_ranges_path=args.descriptor_ranges,
        descriptor_lookups_path=args.descriptor_lookups,
        selection_rows_path=args.selection_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
