# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 SIMD source-memory contract rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.scalar import (
    x86_factored_index_emit,
    x86_factored_memory_immediates,
    x86_source_memory_byte_offset_materializer,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

# Source element formats sharing one physical full-register transfer. Grouping
# equal-width scalar types in one type pattern keeps the generated rule table
# proportional to physical memory forms instead of source type spellings.
_STORAGE_FORMATS = (
    (("i32", "f32"), 4),
    (("i8",), 1),
)


class _VectorMemoryAddressing(Enum):
    STATIC = "static"
    DIRECT = "direct"
    FACTORED_2 = "factored-2"
    FACTORED_4 = "factored-4"
    FACTORED_8 = "factored-8"
    PRESERVE_SOURCE_INDEX = "preserve-source-index"
    MATERIALIZE_BYTE_OFFSET = "materialize-byte-offset"

    @property
    def is_dynamic(self) -> bool:
        return self is not _VectorMemoryAddressing.STATIC

    @property
    def dynamic_byte_stride_factor(self) -> int:
        return {
            _VectorMemoryAddressing.FACTORED_2: 2,
            _VectorMemoryAddressing.FACTORED_4: 4,
            _VectorMemoryAddressing.FACTORED_8: 8,
        }.get(self, 1)


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    element_byte_count: int,
    lane_count: int,
    addressing: _VectorMemoryAddressing,
    diagnostic: GuardDiagnostic,
) -> SourceMemoryConstraint:
    accepts_any_dynamic_terms = addressing in {
        _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET,
        _VectorMemoryAddressing.PRESERVE_SOURCE_INDEX,
    }
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=_DISP32_MIN,
        static_byte_offset_maximum=_DISP32_MAX,
        dynamic_term_count=(
            None
            if addressing.is_dynamic and accepts_any_dynamic_terms
            else 1
            if addressing.is_dynamic
            else 0
        ),
        dynamic_term_count_minimum=(
            1 if addressing.is_dynamic and accepts_any_dynamic_terms else 0
        ),
        dynamic_view_base_term_count=(
            None if addressing is _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET else 0
        ),
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if addressing.is_dynamic and not accepts_any_dynamic_terms
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=(
            0
            if addressing.is_dynamic and accepts_any_dynamic_terms
            else element_byte_count * addressing.dynamic_byte_stride_factor
        )
        if addressing.is_dynamic
        else 0,
        preserve_source_index=(
            addressing is _VectorMemoryAddressing.PRESERVE_SOURCE_INDEX
        ),
        diagnostic=diagnostic,
    )


def _memory_immediates(
    addressing: _VectorMemoryAddressing,
    *,
    element_byte_count: int,
) -> dict[str, SourceMemoryProject | int]:
    immediates: dict[str, SourceMemoryProject | int] = {
        "disp32": SourceMemoryProject.static_byte_offset()
    }
    if addressing.is_dynamic:
        immediates["scale"] = (
            1
            if addressing is _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET
            else element_byte_count
            if addressing is _VectorMemoryAddressing.PRESERVE_SOURCE_INDEX
            else SourceMemoryProject.dynamic_byte_stride()
        )
    return immediates


def _vector_memory_rule(
    operation: SourceMemoryOperation,
    value_type: TypePattern,
    *,
    element_byte_count: int,
    lane_count: int,
    addressing: _VectorMemoryAddressing,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {"base": ValueRef.operand("view")}
    results: dict[str, ValueRef] = {}
    if operation is SourceMemoryOperation.LOAD:
        source_op = vector.vector_load
        type_field = "result"
        results["dst"] = ValueRef.result("result")
    elif operation is SourceMemoryOperation.STORE:
        source_op = vector.vector_store
        type_field = "value"
        operands["value"] = ValueRef.operand("value")
    else:
        raise ValueError(f"unsupported x86 vector memory operation {operation.value}")
    if addressing.is_dynamic:
        if addressing is _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif addressing.dynamic_byte_stride_factor != 1:
            operands["index"] = ValueRef.temporary("factored_index")
        elif addressing is _VectorMemoryAddressing.PRESERVE_SOURCE_INDEX:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    source_memory = _source_memory_constraint(
        operation,
        element_byte_count=element_byte_count,
        lane_count=lane_count,
        addressing=addressing,
        diagnostic=diagnostic,
    )
    memory_emit = EmitDescriptorOp(
        descriptor=descriptor,
        operands=operands,
        results=results,
        immediates=(
            x86_factored_memory_immediates(
                element_byte_count=element_byte_count,
            )
            if addressing.dynamic_byte_stride_factor != 1
            else _memory_immediates(
                addressing,
                element_byte_count=element_byte_count,
            )
        ),
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_byte_offset_materializer=(
            x86_source_memory_byte_offset_materializer(descriptor_lookup)
            if addressing is _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET
            else None
        ),
    )
    if addressing.dynamic_byte_stride_factor != 1:
        emit = (
            x86_factored_index_emit(
                descriptor_lookup=descriptor_lookup,
                element_byte_count=element_byte_count,
                dynamic_byte_stride_factor=addressing.dynamic_byte_stride_factor,
                source_memory=source_memory,
            ),
            memory_emit,
        )
    else:
        emit = (memory_emit,)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *(
                ()
                if addressing is _VectorMemoryAddressing.MATERIALIZE_BYTE_OFFSET
                else (
                    Guard.operand_segment_count(
                        "indices", 1 if addressing.is_dynamic else 0
                    ),
                )
            ),
            Guard.value_type(type_field, value_type),
        ),
        emit=emit,
    )


def _memory_descriptor_key(
    descriptor_key_prefix: str,
    operation: str,
    *,
    addressing: _VectorMemoryAddressing,
    register_suffix: str,
) -> str:
    indexed = ".indexed" if addressing.is_dynamic else ""
    return f"{descriptor_key_prefix}.vmovdqu32.{operation}{indexed}.{register_suffix}"


def x86_vector_memory_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    vector_bit_widths: Sequence[int],
    diagnostic: GuardDiagnostic,
) -> tuple[DescriptorRule, ...]:
    """Builds exact source-memory rules for x86 full-register transfers."""

    rules: list[DescriptorRule] = []
    for vector_bit_width in vector_bit_widths:
        register_suffix = {128: "xmm", 256: "ymm", 512: "zmm"}.get(vector_bit_width)
        if register_suffix is None:
            raise ValueError(f"unsupported x86 memory vector width {vector_bit_width}")
        for element_types, element_byte_count in _STORAGE_FORMATS:
            vector_byte_width = vector_bit_width // 8
            if vector_byte_width % element_byte_count != 0:
                raise ValueError(
                    f"x86 vector width {vector_bit_width} cannot store "
                    f"{element_byte_count}-byte elements"
                )
            lane_count = vector_byte_width // element_byte_count
            value_type = Vector(element_types, lanes=lane_count)
            for operation in (
                SourceMemoryOperation.LOAD,
                SourceMemoryOperation.STORE,
            ):
                for addressing in _VectorMemoryAddressing:
                    descriptor_key = _memory_descriptor_key(
                        descriptor_key_prefix,
                        operation.value,
                        addressing=addressing,
                        register_suffix=register_suffix,
                    )
                    rules.append(
                        _vector_memory_rule(
                            operation,
                            value_type,
                            element_byte_count=element_byte_count,
                            lane_count=lane_count,
                            addressing=addressing,
                            descriptor_key=descriptor_key,
                            descriptor_lookup=descriptor_lookup,
                            diagnostic=diagnostic,
                        )
                    )
    return tuple(rules)
