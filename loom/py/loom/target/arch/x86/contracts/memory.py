# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 scalar and SIMD source-memory contract rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

from loom.dialect.buffer import defs as buffer
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryIntegerConversion,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_I64 = Scalar("i64")
_BYTE_STORAGE_TYPES = ("i8", "f8E4M3", "f8E5M2")
_WORD_STORAGE_TYPES = ("i16", "f16", "bf16")
_I64_MIN = -(2**63) + 1
_I64_MAX = (2**63) - 1

_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

# Source element formats sharing one physical full-register transfer. Grouping
# equal-width scalar types in one type pattern keeps the generated rule table
# proportional to physical memory forms instead of source type spellings.
_STORAGE_FORMATS = (
    (("i32", "f32"), 4),
    (_BYTE_STORAGE_TYPES, 1),
    (_WORD_STORAGE_TYPES, 2),
)


class _MemoryAddressing(Enum):
    STATIC = "static"
    DIRECT = "direct"
    FACTORED_2 = "factored-2"
    FACTORED_4 = "factored-4"
    FACTORED_8 = "factored-8"
    PRESERVE_SOURCE_INDEX = "preserve-source-index"
    MATERIALIZE_BYTE_OFFSET = "materialize-byte-offset"

    @property
    def is_dynamic(self) -> bool:
        return self is not _MemoryAddressing.STATIC

    @property
    def dynamic_byte_stride_factor(self) -> int:
        return {
            _MemoryAddressing.FACTORED_2: 2,
            _MemoryAddressing.FACTORED_4: 4,
            _MemoryAddressing.FACTORED_8: 8,
        }.get(self, 1)


def _byte_offset_materializer(
    descriptor_lookup: _DescriptorLookup,
) -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        constant=descriptor_lookup("x86.scalar.movimm.gpr64"),
        add=descriptor_lookup("x86.scalar.add.gpr64"),
        multiply=descriptor_lookup("x86.scalar.imul.gpr64"),
        shift_left=None,
        constant_immediate="imm64",
        integer_conversions=(
            SourceMemoryIntegerConversion(
                "i1", descriptor_lookup("x86.scalar.movzx.gpr64.gpr32")
            ),
            SourceMemoryIntegerConversion(
                "i32", descriptor_lookup("x86.scalar.movsxd.gpr64.gpr32")
            ),
        ),
    )


def _factored_index_emit(
    *,
    descriptor_lookup: _DescriptorLookup,
    element_byte_count: int,
    dynamic_byte_stride_factor: int,
    source_memory: SourceMemoryConstraint,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        form=DescriptorEmitForm.OP,
        descriptor=descriptor_lookup("x86.scalar.lea.scale.gpr64"),
        operands={"index": ValueRef.source_memory_dynamic_term()},
        results={"dst": ValueRef.temporary("factored_index")},
        result_types={"dst": _I64},
        immediates={
            "disp32": SourceMemoryProject.static_byte_offset_quotient(
                element_byte_count
            ),
            "scale": dynamic_byte_stride_factor,
        },
        source_memory=source_memory,
        source_memory_byte_offset_materializer=_byte_offset_materializer(
            descriptor_lookup
        ),
    )


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    element_byte_count: int,
    lane_count: int,
    addressing: _MemoryAddressing,
    diagnostic: GuardDiagnostic,
) -> SourceMemoryConstraint:
    accepts_any_dynamic_terms = addressing in {
        _MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
        _MemoryAddressing.PRESERVE_SOURCE_INDEX,
    }
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=_DISP32_MIN,
        static_byte_offset_maximum=_DISP32_MAX,
        dynamic_term_count=(
            None if accepts_any_dynamic_terms else 1 if addressing.is_dynamic else 0
        ),
        dynamic_term_count_minimum=1 if accepts_any_dynamic_terms else 0,
        dynamic_view_base_term_count=(
            None if addressing is _MemoryAddressing.MATERIALIZE_BYTE_OFFSET else 0
        ),
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if addressing.is_dynamic and not accepts_any_dynamic_terms
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=(
            element_byte_count * addressing.dynamic_byte_stride_factor
            if addressing.is_dynamic and not accepts_any_dynamic_terms
            else 0
        ),
        preserve_source_index=addressing is _MemoryAddressing.PRESERVE_SOURCE_INDEX,
        diagnostic=diagnostic,
    )


def _memory_immediates(
    addressing: _MemoryAddressing,
    *,
    element_byte_count: int,
) -> dict[str, SourceMemoryProject | int]:
    if addressing.dynamic_byte_stride_factor != 1:
        return {
            "disp32": SourceMemoryProject.static_byte_offset_remainder(
                element_byte_count
            ),
            "scale": element_byte_count,
        }
    immediates: dict[str, SourceMemoryProject | int] = {
        "disp32": SourceMemoryProject.static_byte_offset()
    }
    if addressing.is_dynamic:
        immediates["scale"] = (
            1
            if addressing is _MemoryAddressing.MATERIALIZE_BYTE_OFFSET
            else element_byte_count
            if addressing is _MemoryAddressing.PRESERVE_SOURCE_INDEX
            else SourceMemoryProject.dynamic_byte_stride()
        )
    return immediates


def _memory_rule(
    source_op: Op,
    operation: SourceMemoryOperation,
    value_type: TypePattern,
    *,
    element_byte_count: int,
    lane_count: int,
    addressing: _MemoryAddressing,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    operands = {"base": ValueRef.source_memory_root()}
    results: dict[str, ValueRef] = {}
    if operation is SourceMemoryOperation.LOAD:
        type_field = "result"
        results["dst"] = ValueRef.result("result")
    elif operation is SourceMemoryOperation.STORE:
        type_field = "value"
        operands["value"] = ValueRef.operand("value")
    else:
        raise ValueError(f"unsupported x86 memory operation {operation.value}")
    if addressing.is_dynamic:
        if addressing is _MemoryAddressing.MATERIALIZE_BYTE_OFFSET:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif addressing.dynamic_byte_stride_factor != 1:
            operands["index"] = ValueRef.temporary("factored_index")
        elif addressing is _MemoryAddressing.PRESERVE_SOURCE_INDEX:
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
        immediates=_memory_immediates(
            addressing, element_byte_count=element_byte_count
        ),
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_byte_offset_materializer=(
            _byte_offset_materializer(descriptor_lookup)
            if addressing
            in {
                _MemoryAddressing.DIRECT,
                _MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
            }
            else None
        ),
    )
    if addressing.dynamic_byte_stride_factor != 1:
        emit = (
            _factored_index_emit(
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
                if addressing is _MemoryAddressing.MATERIALIZE_BYTE_OFFSET
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
    operation: SourceMemoryOperation,
    *,
    addressing: _MemoryAddressing,
    register_suffix: str,
) -> str:
    indexed = ".indexed" if addressing.is_dynamic else ""
    return f"{descriptor_key_prefix}.{operation.value}{indexed}.{register_suffix}"


def _full_width_memory_rules(
    source_op: Op,
    operation: SourceMemoryOperation,
    value_type: TypePattern,
    *,
    element_byte_count: int,
    lane_count: int,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    diagnostic: GuardDiagnostic,
) -> tuple[DescriptorRule, ...]:
    """Materializes displacements that cannot fit an instruction's disp32."""

    descriptor = descriptor_lookup(descriptor_key)
    if operation is SourceMemoryOperation.LOAD:
        type_field = "result"
        results = {"dst": ValueRef.result("result")}
        value_operands = {}
    elif operation is SourceMemoryOperation.STORE:
        type_field = "value"
        results = {}
        value_operands = {"value": ValueRef.operand("value")}
    else:
        raise ValueError(f"unsupported x86 memory operation {operation.value}")
    rules: list[DescriptorRule] = []
    for dynamic in (False, True):
        source_memory = SourceMemoryConstraint(
            operation=operation,
            root_kind=SourceMemoryRootKind.ANY,
            memory_spaces=("unknown", "generic", "global"),
            element_byte_count=element_byte_count,
            vector_lane_count=lane_count,
            vector_lane_byte_stride=element_byte_count,
            static_byte_offset_minimum=_I64_MIN,
            static_byte_offset_maximum=_I64_MAX,
            dynamic_term_count=None if dynamic else 0,
            dynamic_term_count_minimum=1 if dynamic else 0,
            dynamic_view_base_term_count=None,
            allow_dynamic_stride_values=dynamic,
            diagnostic=diagnostic,
        )
        static_offset = ValueRef.temporary("static_byte_offset")
        byte_offset = static_offset
        emits = [
            EmitDescriptorOp(
                descriptor=descriptor_lookup("x86.scalar.movimm.gpr64"),
                results={"dst": static_offset},
                result_types={"dst": _I64},
                immediates={"imm64": SourceMemoryProject.static_byte_offset()},
                source_memory=source_memory,
                form=DescriptorEmitForm.CONST,
            )
        ]
        if dynamic:
            byte_offset = ValueRef.temporary("byte_offset")
            emits.append(
                EmitDescriptorOp(
                    form=DescriptorEmitForm.OP,
                    descriptor=descriptor_lookup("x86.scalar.add.gpr64"),
                    operands={
                        "lhs": static_offset,
                        "rhs": ValueRef.source_memory_dynamic_byte_offset(),
                    },
                    results={"dst": byte_offset},
                    result_types={"dst": _I64},
                    source_memory=source_memory,
                    source_memory_byte_offset_materializer=(
                        _byte_offset_materializer(descriptor_lookup)
                    ),
                )
            )
        emits.append(
            EmitDescriptorOp(
                form=DescriptorEmitForm.OP,
                descriptor=descriptor,
                operands={
                    "base": ValueRef.source_memory_root(),
                    "index": byte_offset,
                    **value_operands,
                },
                results=results,
                immediates={"disp32": 0, "scale": 1},
                source_memory=source_memory,
            )
        )
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=descriptor,
                guards=(Guard.value_type(type_field, value_type),),
                emit=tuple(emits),
            )
        )
    return tuple(rules)


def _view_carrier_constraint(
    *,
    element_byte_count: int,
    dynamic: bool,
    full_width_static_offset: bool,
    diagnostic: GuardDiagnostic,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=SourceMemoryOperation.VIEW_CARRIER,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=1,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=(
            _I64_MIN if full_width_static_offset else _DISP32_MIN
        ),
        static_byte_offset_maximum=(
            _I64_MAX if full_width_static_offset else _DISP32_MAX
        ),
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        dynamic_view_base_term_count=None,
        allow_dynamic_stride_values=dynamic,
        diagnostic=diagnostic,
    )


def _view_carrier_rule(
    source_op: Op,
    *,
    element_byte_count: int,
    dynamic: bool,
    full_width_static_offset: bool,
    descriptor_lookup: _DescriptorLookup,
    diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    source_memory = _view_carrier_constraint(
        element_byte_count=element_byte_count,
        dynamic=dynamic,
        full_width_static_offset=full_width_static_offset,
        diagnostic=diagnostic,
    )
    root = ValueRef.source_memory_root()
    result = ValueRef.result("result")
    emits: list[EmitDescriptorOp] = []
    if not full_width_static_offset:
        descriptor = descriptor_lookup(
            "x86.scalar.lea.add_scale.gpr64" if dynamic else "x86.scalar.lea.disp.gpr64"
        )
        operands = {"base": root}
        immediates: dict[str, SourceMemoryProject | int] = {
            "disp32": SourceMemoryProject.static_byte_offset()
        }
        if dynamic:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
            immediates["scale"] = 1
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": result},
                immediates=immediates,
                source_memory=source_memory,
                source_memory_byte_offset_materializer=(
                    _byte_offset_materializer(descriptor_lookup) if dynamic else None
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        return DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            emit=tuple(emits),
        )

    static_offset = ValueRef.temporary("static_byte_offset")
    byte_offset = static_offset
    emits.append(
        EmitDescriptorOp(
            descriptor=descriptor_lookup("x86.scalar.movimm.gpr64"),
            results={"dst": static_offset},
            result_types={"dst": _I64},
            immediates={"imm64": SourceMemoryProject.static_byte_offset()},
            source_memory=source_memory,
            form=DescriptorEmitForm.CONST,
        )
    )
    if dynamic:
        byte_offset = ValueRef.temporary("byte_offset")
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_lookup("x86.scalar.add.gpr64"),
                operands={
                    "lhs": static_offset,
                    "rhs": ValueRef.source_memory_dynamic_byte_offset(),
                },
                results={"dst": byte_offset},
                result_types={"dst": _I64},
                source_memory=source_memory,
                source_memory_byte_offset_materializer=(
                    _byte_offset_materializer(descriptor_lookup)
                ),
                form=DescriptorEmitForm.OP,
            )
        )
    descriptor = descriptor_lookup("x86.scalar.lea.add.gpr64")
    emits.append(
        EmitDescriptorOp(
            descriptor=descriptor,
            operands={"lhs": root, "rhs": byte_offset},
            results={"dst": result},
            source_memory=source_memory,
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        emit=tuple(emits),
    )


def x86_view_carrier_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    diagnostic: GuardDiagnostic,
) -> tuple[DescriptorRule, ...]:
    """Materializes demanded typed views as one complete-address GPR."""

    return tuple(
        _view_carrier_rule(
            source_op,
            element_byte_count=element_byte_count,
            dynamic=dynamic,
            full_width_static_offset=full_width_static_offset,
            descriptor_lookup=descriptor_lookup,
            diagnostic=diagnostic,
        )
        for source_op in (buffer.buffer_view, view.view_subview)
        for element_byte_count in (1, 2, 4, 8)
        for full_width_static_offset in (False, True)
        for dynamic in (False, True)
    )


def x86_scalar_memory_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    diagnostic: GuardDiagnostic,
) -> tuple[DescriptorRule, ...]:
    """Builds source-memory rules for x86 scalar register transfers."""

    rules: list[DescriptorRule] = []
    # Both static transfers precede the dynamic load and store groups. Within
    # each dynamic group, native addressing precedes byte-offset materialization.
    addressing_groups = (
        (_MemoryAddressing.STATIC,),
        tuple(addressing for addressing in _MemoryAddressing if addressing.is_dynamic),
    )
    for value_type, element_byte_count, register_suffix, load_mnemonic in (
        (Scalar("i32"), 4, "gpr32", "mov"),
        (_I64, 8, "gpr64", "mov"),
        (Scalar(_BYTE_STORAGE_TYPES), 1, "u8.gpr32", "movzx"),
        (Scalar(_WORD_STORAGE_TYPES), 2, "u16.gpr32", "movzx"),
    ):
        operations = (
            (view.view_load, SourceMemoryOperation.LOAD, load_mnemonic),
            (view.view_store, SourceMemoryOperation.STORE, "mov"),
        )
        for addressings in addressing_groups:
            for source_op, operation, mnemonic in operations:
                rules.extend(
                    _memory_rule(
                        source_op,
                        operation,
                        value_type,
                        element_byte_count=element_byte_count,
                        lane_count=1,
                        addressing=addressing,
                        descriptor_key=_memory_descriptor_key(
                            f"x86.scalar.{mnemonic}",
                            operation,
                            addressing=addressing,
                            register_suffix=register_suffix,
                        ),
                        descriptor_lookup=descriptor_lookup,
                        diagnostic=diagnostic,
                    )
                    for addressing in addressings
                )
        for source_op, operation, mnemonic in operations:
            rules.extend(
                _full_width_memory_rules(
                    source_op,
                    operation,
                    value_type,
                    element_byte_count=element_byte_count,
                    lane_count=1,
                    descriptor_key=_memory_descriptor_key(
                        f"x86.scalar.{mnemonic}",
                        operation,
                        addressing=_MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
                        register_suffix=register_suffix,
                    ),
                    descriptor_lookup=descriptor_lookup,
                    diagnostic=diagnostic,
                )
            )
    return tuple(rules)


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
            for source_op, operation in (
                (vector.vector_load, SourceMemoryOperation.LOAD),
                (vector.vector_store, SourceMemoryOperation.STORE),
            ):
                for addressing in _MemoryAddressing:
                    descriptor_key = _memory_descriptor_key(
                        f"{descriptor_key_prefix}.vmovdqu32",
                        operation,
                        addressing=addressing,
                        register_suffix=register_suffix,
                    )
                    rules.append(
                        _memory_rule(
                            source_op,
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
                rules.extend(
                    _full_width_memory_rules(
                        source_op,
                        operation,
                        value_type,
                        element_byte_count=element_byte_count,
                        lane_count=lane_count,
                        descriptor_key=_memory_descriptor_key(
                            f"{descriptor_key_prefix}.vmovdqu32",
                            operation,
                            addressing=_MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
                            register_suffix=register_suffix,
                        ),
                        descriptor_lookup=descriptor_lookup,
                        diagnostic=diagnostic,
                    )
                )
    return tuple(rules)
