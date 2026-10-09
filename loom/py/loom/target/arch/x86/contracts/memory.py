# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 scalar and SIMD source-memory contract rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    SourceNode,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_I16 = Scalar("i16")
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
    (("i64", "f64"), 8),
)


@dataclass(frozen=True, slots=True)
class _MemoryValueTransport:
    """Moves a scalar memory value between its memory and source carriers."""

    memory_type: TypePattern
    load: Descriptor
    store: Descriptor


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
        memory_spaces=("unknown", "generic", "global", "private")
        + (("constant",) if operation is SourceMemoryOperation.LOAD else ()),
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
    transport: _MemoryValueTransport | None = None,
    source_nodes: Sequence[SourceNode] = (),
    load_result: ValueRef | None = None,
    instruction_operands: Mapping[str, ValueRef] | None = None,
    pre_memory_emits: Sequence[EmitDescriptorOp] = (),
    priority: int = 0,
    report_key: str = "",
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    instruction_operands = instruction_operands or {}
    operands = {"base": ValueRef.source_memory_root(), **instruction_operands}
    results: dict[str, ValueRef] = {}
    result_types: dict[str, TypePattern] = {}
    transport_emit: EmitDescriptorOp | None = None
    transport_descriptor: Descriptor | None = None
    if operation is SourceMemoryOperation.LOAD:
        type_field = "result"
        if transport is None:
            results["dst"] = (
                ValueRef.result("result") if load_result is None else load_result
            )
        else:
            if load_result is not None:
                raise ValueError("transported x86 loads cannot override their result")
            transport_descriptor = transport.load
            memory_value = ValueRef.temporary("memory_value")
            results["dst"] = memory_value
            result_types["dst"] = transport.memory_type
            transport_emit = EmitDescriptorOp(
                descriptor=transport.load,
                operands={"input": memory_value},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            )
    elif operation is SourceMemoryOperation.STORE:
        type_field = "value"
        if transport is None:
            operands["value"] = ValueRef.operand("value")
        else:
            transport_descriptor = transport.store
            memory_value = ValueRef.temporary("memory_value")
            operands["value"] = memory_value
            transport_emit = EmitDescriptorOp(
                descriptor=transport.store,
                operands={"input": ValueRef.operand("value")},
                results={"dst": memory_value},
                result_types={"dst": transport.memory_type},
                form=DescriptorEmitForm.OP,
            )
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
        result_types=result_types,
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
    emits: list[EmitDescriptorOp] = []
    if addressing.dynamic_byte_stride_factor != 1:
        emits.append(
            _factored_index_emit(
                descriptor_lookup=descriptor_lookup,
                element_byte_count=element_byte_count,
                dynamic_byte_stride_factor=addressing.dynamic_byte_stride_factor,
                source_memory=source_memory,
            )
        )
    if operation is SourceMemoryOperation.STORE and transport_emit is not None:
        emits.append(transport_emit)
    emits.extend(pre_memory_emits)
    emits.append(memory_emit)
    if operation is SourceMemoryOperation.LOAD and transport_emit is not None:
        emits.append(transport_emit)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        source_nodes=source_nodes,
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
            *(
                (Guard.descriptor_available(transport_descriptor),)
                if transport_descriptor is not None
                else ()
            ),
            *(
                Guard.descriptor_available(dependency)
                for dependency in dict.fromkeys(
                    emit.descriptor for emit in pre_memory_emits
                )
                if dependency != descriptor
            ),
        ),
        emit=tuple(emits),
        priority=priority,
        report_key=report_key,
    )


def _memory_descriptor_key(
    descriptor_key_prefix: str,
    operation: SourceMemoryOperation,
    *,
    addressing: _MemoryAddressing,
    memory_form: str | None = None,
    modifiers: Sequence[str] = (),
    register_suffix: str,
) -> str:
    indexed = ".indexed" if addressing.is_dynamic else ""
    modifier_suffix = "".join(f".{modifier}" for modifier in modifiers)
    memory_form = operation.value if memory_form is None else memory_form
    return (
        f"{descriptor_key_prefix}.{memory_form}{indexed}{modifier_suffix}."
        f"{register_suffix}"
    )


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
    transport: _MemoryValueTransport | None = None,
    source_nodes: Sequence[SourceNode] = (),
    load_result: ValueRef | None = None,
    instruction_operands: Mapping[str, ValueRef] | None = None,
    pre_memory_emits: Sequence[EmitDescriptorOp] = (),
    priority: int = 0,
    report_key: str = "",
) -> tuple[DescriptorRule, ...]:
    """Materializes displacements that cannot fit an instruction's disp32."""

    descriptor = descriptor_lookup(descriptor_key)
    instruction_operands = instruction_operands or {}
    result_types: dict[str, TypePattern] = {}
    transport_descriptor: Descriptor | None = None
    transport_emit: EmitDescriptorOp | None = None
    if operation is SourceMemoryOperation.LOAD:
        type_field = "result"
        if transport is None:
            results = {
                "dst": ValueRef.result("result") if load_result is None else load_result
            }
        else:
            if load_result is not None:
                raise ValueError("transported x86 loads cannot override their result")
            transport_descriptor = transport.load
            memory_value = ValueRef.temporary("memory_value")
            results = {"dst": memory_value}
            result_types = {"dst": transport.memory_type}
            transport_emit = EmitDescriptorOp(
                descriptor=transport.load,
                operands={"input": memory_value},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            )
        value_operands = {}
    elif operation is SourceMemoryOperation.STORE:
        type_field = "value"
        results = {}
        if transport is None:
            value_operands = {"value": ValueRef.operand("value")}
        else:
            transport_descriptor = transport.store
            memory_value = ValueRef.temporary("memory_value")
            value_operands = {"value": memory_value}
            transport_emit = EmitDescriptorOp(
                descriptor=transport.store,
                operands={"input": ValueRef.operand("value")},
                results={"dst": memory_value},
                result_types={"dst": transport.memory_type},
                form=DescriptorEmitForm.OP,
            )
    else:
        raise ValueError(f"unsupported x86 memory operation {operation.value}")
    rules: list[DescriptorRule] = []
    for dynamic in (False, True):
        source_memory = SourceMemoryConstraint(
            operation=operation,
            root_kind=SourceMemoryRootKind.ANY,
            memory_spaces=("unknown", "generic", "global", "private")
            + (("constant",) if operation is SourceMemoryOperation.LOAD else ()),
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
        if operation is SourceMemoryOperation.STORE and transport_emit is not None:
            emits.append(transport_emit)
        emits.extend(pre_memory_emits)
        emits.append(
            EmitDescriptorOp(
                form=DescriptorEmitForm.OP,
                descriptor=descriptor,
                operands={
                    "base": ValueRef.source_memory_root(),
                    "index": byte_offset,
                    **instruction_operands,
                    **value_operands,
                },
                results=results,
                result_types=result_types,
                immediates={"disp32": 0, "scale": 1},
                source_memory=source_memory,
            )
        )
        if operation is SourceMemoryOperation.LOAD and transport_emit is not None:
            emits.append(transport_emit)
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=descriptor,
                source_nodes=source_nodes,
                guards=(
                    Guard.value_type(type_field, value_type),
                    *(
                        (Guard.descriptor_available(transport_descriptor),)
                        if transport_descriptor is not None
                        else ()
                    ),
                    *(
                        Guard.descriptor_available(dependency)
                        for dependency in dict.fromkeys(
                            emit.descriptor for emit in pre_memory_emits
                        )
                        if dependency != descriptor
                    ),
                ),
                emit=tuple(emits),
                priority=priority,
                report_key=report_key,
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
        memory_spaces=("unknown", "generic", "global", "private", "constant"),
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


def _scalar_memory_family_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    value_type: TypePattern,
    element_byte_count: int,
    register_suffix: str,
    load_mnemonic: str,
    diagnostic: GuardDiagnostic,
    transport: _MemoryValueTransport | None = None,
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    # Both static transfers precede the dynamic load and store groups. Within
    # each dynamic group, native addressing precedes byte-offset materialization.
    addressing_groups = (
        (_MemoryAddressing.STATIC,),
        tuple(addressing for addressing in _MemoryAddressing if addressing.is_dynamic),
    )
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
                    transport=transport,
                    priority=priority,
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
                transport=transport,
                priority=priority,
            )
        )
    return tuple(rules)


def x86_scalar_memory_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    diagnostic: GuardDiagnostic,
) -> tuple[DescriptorRule, ...]:
    """Builds source-memory rules for x86 scalar register transfers."""

    return tuple(
        rule
        for value_type, element_byte_count, register_suffix, load_mnemonic in (
            (Scalar("i32"), 4, "gpr32", "mov"),
            (_I64, 8, "gpr64", "mov"),
            (Scalar(_BYTE_STORAGE_TYPES), 1, "u8.gpr32", "movzx"),
            (Scalar(_WORD_STORAGE_TYPES), 2, "u16.gpr32", "movzx"),
        )
        for rule in _scalar_memory_family_rules(
            descriptor_lookup,
            value_type=value_type,
            element_byte_count=element_byte_count,
            register_suffix=register_suffix,
            load_mnemonic=load_mnemonic,
            diagnostic=diagnostic,
        )
    )


def x86_scalar_xmm_word_memory_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    value_type: TypePattern,
    diagnostic: GuardDiagnostic,
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    """Transfers a 16-bit scalar through memory and an XMM source carrier."""

    return _scalar_memory_family_rules(
        descriptor_lookup,
        value_type=value_type,
        element_byte_count=2,
        register_suffix="u16.gpr32",
        load_mnemonic="movzx",
        diagnostic=diagnostic,
        transport=_MemoryValueTransport(
            memory_type=_I16,
            load=descriptor_lookup("x86.avx2.vmovd.xmm.gpr32"),
            store=descriptor_lookup("x86.avx2.vmovd.gpr32.xmm"),
        ),
        priority=priority,
    )


def _vector_memory_type_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    register_suffix: str,
    value_type: TypePattern,
    element_byte_count: int,
    lane_count: int,
    diagnostic: GuardDiagnostic,
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    """Builds every addressing form for one exact vector transfer type."""
    rules: list[DescriptorRule] = []
    for source_op, operation in (
        (vector.vector_load, SourceMemoryOperation.LOAD),
        (vector.vector_store, SourceMemoryOperation.STORE),
    ):
        for addressing in _MemoryAddressing:
            descriptor_key = _memory_descriptor_key(
                descriptor_key_prefix,
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
                    priority=priority,
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
                    descriptor_key_prefix,
                    operation,
                    addressing=_MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
                    register_suffix=register_suffix,
                ),
                descriptor_lookup=descriptor_lookup,
                diagnostic=diagnostic,
                priority=priority,
            )
        )
    return tuple(rules)


def x86_low_xmm_vector_memory_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    value_type: TypePattern,
    element_byte_count: int,
    lane_count: int,
    diagnostic: GuardDiagnostic,
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    """Transfers one 64-bit logical vector through the low half of XMM."""

    if element_byte_count * lane_count != 8:
        raise ValueError("low-XMM source memory requires an eight-byte payload")
    return _vector_memory_type_rules(
        descriptor_lookup,
        descriptor_key_prefix="x86.avx2.vmovsd",
        register_suffix="xmm",
        value_type=value_type,
        element_byte_count=element_byte_count,
        lane_count=lane_count,
        diagnostic=diagnostic,
        priority=priority,
    )


def x86_fused_load_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    source_op: Op,
    source_type: TypePattern,
    source_nodes: Sequence[SourceNode],
    result: ValueRef,
    element_byte_count: int,
    lane_count: int,
    descriptor_key_prefix: str,
    descriptor_memory_form: str = "load",
    descriptor_key_modifiers: Sequence[str] = (),
    register_suffix: str,
    instruction_operands: Mapping[str, ValueRef] | None = None,
    pre_memory_emits: Sequence[EmitDescriptorOp] = (),
    diagnostic: GuardDiagnostic,
    report_key: str,
    priority: int = 1,
) -> tuple[DescriptorRule, ...]:
    """Builds every x86 addressing form for a fused source load."""

    rules = [
        _memory_rule(
            source_op,
            SourceMemoryOperation.LOAD,
            source_type,
            element_byte_count=element_byte_count,
            lane_count=lane_count,
            addressing=addressing,
            descriptor_key=_memory_descriptor_key(
                descriptor_key_prefix,
                SourceMemoryOperation.LOAD,
                addressing=addressing,
                memory_form=descriptor_memory_form,
                modifiers=descriptor_key_modifiers,
                register_suffix=register_suffix,
            ),
            descriptor_lookup=descriptor_lookup,
            diagnostic=diagnostic,
            source_nodes=source_nodes,
            load_result=result,
            instruction_operands=instruction_operands,
            pre_memory_emits=pre_memory_emits,
            priority=priority,
            report_key=report_key,
        )
        for addressing in _MemoryAddressing
    ]
    rules.extend(
        _full_width_memory_rules(
            source_op,
            SourceMemoryOperation.LOAD,
            source_type,
            element_byte_count=element_byte_count,
            lane_count=lane_count,
            descriptor_key=_memory_descriptor_key(
                descriptor_key_prefix,
                SourceMemoryOperation.LOAD,
                addressing=_MemoryAddressing.MATERIALIZE_BYTE_OFFSET,
                memory_form=descriptor_memory_form,
                modifiers=descriptor_key_modifiers,
                register_suffix=register_suffix,
            ),
            descriptor_lookup=descriptor_lookup,
            diagnostic=diagnostic,
            source_nodes=source_nodes,
            load_result=result,
            instruction_operands=instruction_operands,
            pre_memory_emits=pre_memory_emits,
            priority=priority,
            report_key=report_key,
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
            rules.extend(
                _vector_memory_type_rules(
                    descriptor_lookup,
                    descriptor_key_prefix=f"{descriptor_key_prefix}.vmovdqu32",
                    register_suffix=register_suffix,
                    value_type=value_type,
                    element_byte_count=element_byte_count,
                    lane_count=lane_count,
                    diagnostic=diagnostic,
                )
            )
    return tuple(rules)
