# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom source-operation selection for the Core VM ISA."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

from model.isa import InstructionFieldRole
from model.isa.selectors import (
    SELECTOR_TABLES_BY_NAME,
    SELECTOR_VALUES,
    parse_integer_conversion_name,
)
from model.isa.validation import PACKED_SELECTORS
from model.schema import EntityReference

from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.cfg import defs as cfg_defs
from loom.dialect.func import defs as func_defs
from loom.dialect.index import defs as index_defs
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import defs as scf_defs
from loom.dsl import ATTR_TYPE_ENUM, Op
from loom.ir import BUFFER_TYPE, I1, ScalarType, Type
from loom.scalar_type import ScalarTypeKind
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
)
from loom.target.low_descriptors import DescriptorCarrier, ImmediateKind, OperandRole
from loom.verify import type_satisfies_constraint

_DESCRIPTORS_BY_KEY = {
    descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
}
_SELECTOR_VALUES_BY_KEY = {
    (value.table_id, value.name): value.value for value in SELECTOR_VALUES
}
_INTEGER_TYPE_BY_BIT_COUNT = {
    1: ScalarTypeKind.I1,
    8: ScalarTypeKind.I8,
    16: ScalarTypeKind.I16,
    32: ScalarTypeKind.I32,
    64: ScalarTypeKind.I64,
}


@dataclasses.dataclass(frozen=True, slots=True)
class VmSourceLowering:
    """One generated concrete source signature selecting a VM descriptor."""

    source_op: Op
    operand_types: tuple[Type, ...]
    result_types: tuple[Type, ...]
    descriptor_key: str
    selector_immediate_ordinal: int | None = None
    selector_source_attr_ordinal: int | None = None
    selector_value: int = 0

    def __post_init__(self) -> None:
        if self.selector_immediate_ordinal is None:
            if self.selector_source_attr_ordinal is not None:
                raise ValueError(
                    "VM source lowering has a source selector without a target"
                )
            if self.selector_value != 0:
                raise ValueError("VM source lowering has a value without a selector")
            return
        if not 0 <= self.selector_immediate_ordinal <= 0xFF:
            raise ValueError("VM source selector immediate ordinal exceeds u8")
        if (
            self.selector_source_attr_ordinal is not None
            and not 0 <= self.selector_source_attr_ordinal <= 0xFF
        ):
            raise ValueError("VM source selector attribute ordinal exceeds u8")
        if self.selector_source_attr_ordinal is not None and self.selector_value != 0:
            raise ValueError(
                "VM source lowering cannot combine a source selector with a value"
            )
        if not 0 <= self.selector_value <= 0xFF:
            raise ValueError("VM source selector value exceeds u8")


VM_SOURCE_CONVERSION_MAX_STEP_COUNT = 3


@dataclasses.dataclass(frozen=True, slots=True)
class VmSourceConversionLowering:
    """One source conversion implemented by exact VM conversion steps."""

    source_op: Op
    source_type: ScalarTypeKind
    result_type: ScalarTypeKind
    steps: tuple[VmSourceLowering, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.steps) <= VM_SOURCE_CONVERSION_MAX_STEP_COUNT:
            raise ValueError(
                f"{self.source_op.name}: VM conversion requires one to "
                f"{VM_SOURCE_CONVERSION_MAX_STEP_COUNT} steps"
            )
        if any(
            len(step.operand_types) != 1 or len(step.result_types) != 1
            for step in self.steps
        ):
            raise ValueError(
                f"{self.source_op.name}: VM conversion steps must be unary"
            )
        if self.steps[0].operand_types != (ScalarType(self.source_type),):
            raise ValueError(f"{self.source_op.name}: VM conversion source type drifts")
        if self.steps[-1].result_types != (ScalarType(self.result_type),):
            raise ValueError(f"{self.source_op.name}: VM conversion result type drifts")
        for lhs, rhs in pairwise(self.steps):
            if lhs.result_types != rhs.operand_types:
                raise ValueError(
                    f"{self.source_op.name}: VM conversion steps are discontinuous"
                )
        if any(
            step.selector_immediate_ordinal != 0
            or step.selector_source_attr_ordinal is not None
            for step in self.steps
        ):
            raise ValueError(
                f"{self.source_op.name}: VM conversion steps require one fixed selector"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class VmSourceOpcode:
    """Physical opcode and optional selector chosen by source semantics."""

    descriptor_key: str
    selector_table: str | None = None
    selector_name: str | None = None
    selector_source_attr: str | None = None

    def __post_init__(self) -> None:
        selector_source_count = sum(
            value is not None
            for value in (self.selector_name, self.selector_source_attr)
        )
        if self.selector_table is None:
            if selector_source_count:
                raise ValueError("VM source selector requires a selector table")
        elif selector_source_count != 1:
            raise ValueError(
                "VM source selector requires one fixed name or source attribute"
            )


VM_ATOMIC_COMPONENT_NAMES = (
    "kind",
    "carrier",
    "ordering",
    "success_ordering",
    "failure_ordering",
    "scope",
)


@dataclasses.dataclass(frozen=True, slots=True)
class VmAtomicSelectorComponent:
    """Location of one logical atomic selector within a packed immediate."""

    selector_ordinal: int
    bit_offset: int
    bit_length: int


@dataclasses.dataclass(frozen=True, slots=True)
class VmAtomicSourceLowering:
    """Generated bridge from one source atomic family to its VM record."""

    operation_kind: str
    descriptor_key: str
    selector_count: int
    components: tuple[VmAtomicSelectorComponent | None, ...]


def _integer_conversion_cases(
    operation: str,
) -> tuple[tuple[ScalarTypeKind, ScalarTypeKind, str], ...]:
    """Projects one integer conversion family from the VM selector table."""

    selector_table_id = SELECTOR_TABLES_BY_NAME["integer.convert"].entity_id
    cases = tuple(
        (
            _INTEGER_TYPE_BY_BIT_COUNT[source_bit_count],
            _INTEGER_TYPE_BY_BIT_COUNT[result_bit_count],
            selector.name,
        )
        for selector in SELECTOR_VALUES
        if selector.table_id == selector_table_id
        for selector_operation, source_bit_count, result_bit_count in (
            parse_integer_conversion_name(selector.name),
        )
        if selector_operation == operation
    )
    if not cases:
        raise ValueError(f"VM integer conversion operation {operation!r} is unknown")
    return cases


def _integer_conversion_opcode(
    operation: str,
    source_type: ScalarTypeKind,
    result_type: ScalarTypeKind,
) -> VmSourceOpcode:
    """Selects the VM integer conversion for one fixed-width type pair."""

    selector_names = tuple(
        selector_name
        for case_source_type, case_result_type, selector_name in (
            _integer_conversion_cases(operation)
        )
        if case_source_type is source_type and case_result_type is result_type
    )
    if len(selector_names) != 1:
        raise ValueError(
            f"VM integer conversion {operation} has no unique selector for "
            f"{source_type.name} to {result_type.name}"
        )
    return VmSourceOpcode(
        "vm.conversion.integer",
        "integer.convert",
        selector_names[0],
    )


def _index_cast_opcodes() -> dict[
    tuple[ScalarTypeKind, ScalarTypeKind], str | VmSourceOpcode
]:
    """Projects every valid 64-bit VM address-boundary conversion."""

    fixed_integer_types = tuple(_INTEGER_TYPE_BY_BIT_COUNT.values())
    address_types = (ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET)
    opcodes: dict[tuple[ScalarTypeKind, ScalarTypeKind], str | VmSourceOpcode] = {}

    for source_type in fixed_integer_types:
        for result_type in address_types:
            if source_type is ScalarTypeKind.I64:
                opcode: str | VmSourceOpcode = "vm.value.copy"
            else:
                operation = (
                    "zero_extend"
                    if source_type is ScalarTypeKind.I1
                    or result_type is ScalarTypeKind.OFFSET
                    else "sign_extend"
                )
                opcode = _integer_conversion_opcode(
                    operation,
                    source_type,
                    ScalarTypeKind.I64,
                )
            opcodes[(source_type, result_type)] = opcode

    for source_type in address_types:
        for result_type in fixed_integer_types:
            opcodes[(source_type, result_type)] = (
                "vm.value.copy"
                if result_type is ScalarTypeKind.I64
                else _integer_conversion_opcode(
                    "truncate",
                    ScalarTypeKind.I64,
                    result_type,
                )
            )

    opcodes[(ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET)] = "vm.value.copy"
    opcodes[(ScalarTypeKind.OFFSET, ScalarTypeKind.INDEX)] = "vm.value.copy"
    return opcodes


def _require_fixed_source_shape(
    source_op: Op,
    *,
    operand_count: int,
    result_count: int,
) -> None:
    if (
        len(source_op.operands) != operand_count
        or len(source_op.results) != result_count
    ):
        raise ValueError(
            f"{source_op.name}: expected {operand_count} operands and "
            f"{result_count} results"
        )
    if any(operand.optional or operand.variadic for operand in source_op.operands):
        raise ValueError(f"{source_op.name}: source operands must have fixed arity")
    if any(result.variadic for result in source_op.results):
        raise ValueError(f"{source_op.name}: source results must have fixed arity")


def _require_descriptor_shape(
    source_op: Op,
    source_opcode: VmSourceOpcode,
    *,
    operand_count: int,
    result_count: int,
) -> tuple[int | None, int | None, int]:
    descriptor_key = source_opcode.descriptor_key
    descriptor = _DESCRIPTORS_BY_KEY.get(source_opcode.descriptor_key)
    if descriptor is None:
        raise ValueError(
            f"VM source lowering names unknown descriptor {descriptor_key!r}"
        )
    descriptor_operand_count = sum(
        operand.role is OperandRole.OPERAND for operand in descriptor.operands
    )
    descriptor_result_count = sum(
        operand.role is OperandRole.RESULT for operand in descriptor.operands
    )
    if (
        descriptor_operand_count != operand_count
        or descriptor_result_count != result_count
    ):
        raise ValueError(
            f"{descriptor_key}: descriptor has {descriptor_operand_count} operands and "
            f"{descriptor_result_count} results; expected {operand_count} and "
            f"{result_count}"
        )
    if descriptor.carrier is not DescriptorCarrier.OP:
        raise ValueError(
            f"{descriptor_key}: source operation projection requires an "
            "instruction descriptor"
        )
    if source_opcode.selector_table is None:
        if descriptor.immediates:
            raise ValueError(
                f"{descriptor_key}: source operation projection does not supply "
                "required immediates"
            )
        return None, None, 0

    selector_table = SELECTOR_TABLES_BY_NAME[source_opcode.selector_table]
    selector_immediate_ordinals = tuple(
        index
        for index, immediate in enumerate(descriptor.immediates)
        if immediate.kind is ImmediateKind.ENUM
        and immediate.enum_domain == selector_table.entity_id
    )
    if len(selector_immediate_ordinals) != 1 or len(descriptor.immediates) != 1:
        raise ValueError(
            f"{descriptor_key}: source selector must be the descriptor's only immediate"
        )
    if source_opcode.selector_name is not None:
        selector_key = (selector_table.entity_id, source_opcode.selector_name)
        if selector_key not in _SELECTOR_VALUES_BY_KEY:
            raise ValueError(
                f"{descriptor_key}: unknown selector "
                f"{source_opcode.selector_name!r} in "
                f"{source_opcode.selector_table!r}"
            )
        return (
            selector_immediate_ordinals[0],
            None,
            _SELECTOR_VALUES_BY_KEY[selector_key],
        )

    source_attr_name = source_opcode.selector_source_attr
    if source_attr_name is None:
        raise ValueError(f"{descriptor_key}: source selector attribute is missing")
    source_attr_ordinal = next(
        (
            ordinal
            for ordinal, attr in enumerate(source_op.attrs)
            if attr.name == source_attr_name
        ),
        None,
    )
    if source_attr_ordinal is None:
        raise ValueError(
            f"{source_op.name}: source selector attribute "
            f"{source_attr_name!r} is not declared"
        )
    source_attr = source_op.attrs[source_attr_ordinal]
    if (
        source_attr.attr_type != ATTR_TYPE_ENUM
        or source_attr.enum_def is None
        or source_attr.optional
        or source_attr.open_enum
    ):
        raise ValueError(
            f"{source_op.name}: source selector attribute "
            f"{source_attr_name!r} must be a required closed enum"
        )
    for source_case in source_attr.enum_def.cases:
        selector_key = (selector_table.entity_id, source_case.keyword)
        target_value = _SELECTOR_VALUES_BY_KEY.get(selector_key)
        if target_value is None:
            raise ValueError(
                f"{descriptor_key}: source selector case "
                f"{source_case.keyword!r} is absent from "
                f"{source_opcode.selector_table!r}"
            )
        if target_value != source_case.value:
            raise ValueError(
                f"{descriptor_key}: source selector case "
                f"{source_case.keyword!r} has value {source_case.value}, but "
                f"{source_opcode.selector_table!r} uses {target_value}"
            )
    return selector_immediate_ordinals[0], source_attr_ordinal, 0


def _require_atomic_selector_bridge() -> None:
    """Proves that source atomic enums are exact VM selector ordinals."""

    kind_names = (
        "exchange.integer",
        "exchange.float",
        "add.integer",
        "add.float",
        "subtract.integer",
        "and.integer",
        "or.integer",
        "xor.integer",
        "minimum.signed",
        "maximum.signed",
        "minimum.unsigned",
        "maximum.unsigned",
        "minimum.float",
        "maximum.float",
        "minnum.float",
        "maxnum.float",
    )
    source_and_target_names = (
        (AtomicKind, "buffer.atomic.kind", kind_names),
        (
            AtomicOrdering,
            "buffer.atomic.ordering",
            tuple(case.keyword for case in AtomicOrdering.cases),
        ),
        (
            AtomicScope,
            "buffer.atomic.scope",
            tuple(case.keyword for case in AtomicScope.cases),
        ),
    )
    for source_enum, table_name, target_names in source_and_target_names:
        if len(source_enum.cases) != len(target_names):
            raise ValueError(f"{table_name}: source and VM selector counts differ")
        table_id = SELECTOR_TABLES_BY_NAME[table_name].entity_id
        for source_case, target_name in zip(
            source_enum.cases, target_names, strict=True
        ):
            target_value = _SELECTOR_VALUES_BY_KEY.get((table_id, target_name))
            if target_value != source_case.value:
                raise ValueError(
                    f"{table_name}: source {source_case.keyword!r} uses "
                    f"{source_case.value}, but VM {target_name!r} uses "
                    f"{target_value}"
                )

    carrier_table_id = SELECTOR_TABLES_BY_NAME["buffer.atomic.carrier"].entity_id
    carrier_values = tuple(
        _SELECTOR_VALUES_BY_KEY.get((carrier_table_id, name)) for name in ("i32", "i64")
    )
    if carrier_values != (0, 1):
        raise ValueError("VM atomic carriers must encode i32/i64 as 0/1")


def _atomic_source_lowering(
    operation_kind: str,
    mnemonic: str,
    expected_component_names: frozenset[str],
) -> VmAtomicSourceLowering:
    descriptor_key = f"vm.{mnemonic}"
    descriptor = _DESCRIPTORS_BY_KEY[descriptor_key]
    projection = next(
        (
            projection
            for projection in VM_INSTRUCTION_PROJECTIONS
            if projection.mnemonic == mnemonic
        ),
        None,
    )
    if projection is None:
        raise ValueError(f"{descriptor_key}: atomic instruction is not projected")

    component_table_names = {
        "kind": "buffer.atomic.kind",
        "carrier": "buffer.atomic.carrier",
        "ordering": "buffer.atomic.ordering",
        "success_ordering": "buffer.atomic.ordering",
        "failure_ordering": "buffer.atomic.ordering",
        "scope": "buffer.atomic.scope",
    }
    components: dict[str, VmAtomicSelectorComponent] = {}
    selector_ordinal = 0
    for field in projection.instruction.fields:
        if field.role is not InstructionFieldRole.IMMEDIATE:
            continue
        if field.array_length != 1:
            raise ValueError(f"{descriptor_key}: atomic selector must be scalar")
        rules = tuple(
            rule
            for rule in field.validation
            if rule.rule_id == PACKED_SELECTORS.entity_id
        )
        if len(rules) != 1 or len(rules[0].arguments) != 2:
            raise ValueError(
                f"{descriptor_key}: every immediate must be one packed selector"
            )
        _, packed_components = rules[0].arguments
        if not isinstance(packed_components, tuple) or not packed_components:
            raise ValueError(f"{descriptor_key}: malformed packed selector")
        for packed_component in packed_components:
            if not isinstance(packed_component, tuple) or len(packed_component) != 5:
                raise ValueError(f"{descriptor_key}: malformed selector component")
            name, bit_offset, bit_length, table_reference, _ = packed_component
            if (
                name not in component_table_names
                or name in components
                or not isinstance(bit_offset, int)
                or not isinstance(bit_length, int)
                or not isinstance(table_reference, EntityReference)
            ):
                raise ValueError(f"{descriptor_key}: malformed selector component")
            expected_table_id = SELECTOR_TABLES_BY_NAME[
                component_table_names[name]
            ].entity_id
            if table_reference.entity_id != expected_table_id:
                raise ValueError(
                    f"{descriptor_key}: selector component {name!r} uses the "
                    "wrong enum domain"
                )
            if bit_length <= 0 or bit_offset < 0 or bit_offset + bit_length > 8:
                raise ValueError(
                    f"{descriptor_key}: selector component {name!r} exceeds u8"
                )
            components[name] = VmAtomicSelectorComponent(
                selector_ordinal, bit_offset, bit_length
            )
        selector_ordinal += 1

    if selector_ordinal != len(descriptor.immediates):
        raise ValueError(f"{descriptor_key}: atomic selector projection drifted")
    if set(components) != expected_component_names:
        raise ValueError(
            f"{descriptor_key}: atomic selector components are "
            f"{sorted(components)}, expected {sorted(expected_component_names)}"
        )
    return VmAtomicSourceLowering(
        operation_kind=operation_kind,
        descriptor_key=descriptor_key,
        selector_count=selector_ordinal,
        components=tuple(components.get(name) for name in VM_ATOMIC_COMPONENT_NAMES),
    )


def _atomic_source_lowerings() -> tuple[VmAtomicSourceLowering, ...]:
    """Projects source memory-atomic families from the authoritative ISA."""

    _require_atomic_selector_bridge()
    apply_components = frozenset(("kind", "carrier", "ordering", "scope"))
    return (
        _atomic_source_lowering(
            "ATOMIC_REDUCE", "buffer.atomic.reduce", apply_components
        ),
        _atomic_source_lowering("ATOMIC_RMW", "buffer.atomic.rmw", apply_components),
        _atomic_source_lowering(
            "ATOMIC_CMPXCHG",
            "buffer.atomic.cmpxchg",
            frozenset(("carrier", "success_ordering", "failure_ordering", "scope")),
        ),
    )


VM_ATOMIC_SOURCE_LOWERINGS = _atomic_source_lowerings()


def _require_concrete_source_types(
    source_op: Op,
    operand_types: tuple[Type, ...],
    result_types: tuple[Type, ...],
) -> None:
    fields = (*source_op.operands, *source_op.results)
    source_types = (*operand_types, *result_types)
    for field, source_type in zip(fields, source_types, strict=True):
        if not type_satisfies_constraint(source_type, field.type_constraint):
            raise ValueError(
                f"{source_op.name}: {source_type!r} does not satisfy the "
                f"{field.name} type constraint"
            )


def _source_lowering(
    source_op: Op,
    operand_types: tuple[Type, ...],
    result_types: tuple[Type, ...],
    source_opcode: str | VmSourceOpcode,
) -> VmSourceLowering:
    if isinstance(source_opcode, str):
        source_opcode = VmSourceOpcode(source_opcode)
    _require_fixed_source_shape(
        source_op,
        operand_count=len(operand_types),
        result_count=len(result_types),
    )
    _require_concrete_source_types(source_op, operand_types, result_types)
    (
        selector_immediate_ordinal,
        selector_source_attr_ordinal,
        selector_value,
    ) = _require_descriptor_shape(
        source_op,
        source_opcode,
        operand_count=len(operand_types),
        result_count=len(result_types),
    )
    return VmSourceLowering(
        source_op,
        operand_types,
        result_types,
        source_opcode.descriptor_key,
        selector_immediate_ordinal,
        selector_source_attr_ordinal,
        selector_value,
    )


def _same_type_binary(
    source_op: Op,
    descriptor_by_type: dict[ScalarTypeKind, str | VmSourceOpcode],
) -> tuple[VmSourceLowering, ...]:
    """Projects a same-type binary Loom op onto concrete VM instructions."""

    _require_fixed_source_shape(source_op, operand_count=2, result_count=1)
    field_names = tuple(
        field.name for field in (*source_op.operands, *source_op.results)
    )
    if not any(
        constraint.name == "SameType" and constraint.args == field_names
        for constraint in source_op.constraints
    ):
        raise ValueError(
            f"{source_op.name}: same-type binary projection requires a "
            f"SameType{field_names} constraint"
        )
    if not descriptor_by_type:
        raise ValueError(f"{source_op.name}: same-type projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type), ScalarType(scalar_type)),
            (ScalarType(scalar_type),),
            descriptor_key,
        )
        for scalar_type, descriptor_key in descriptor_by_type.items()
    )


def _same_type_unary(
    source_op: Op,
    descriptor_by_type: dict[ScalarTypeKind, str | VmSourceOpcode],
) -> tuple[VmSourceLowering, ...]:
    """Projects a same-type unary Loom op onto concrete VM instructions."""

    _require_fixed_source_shape(source_op, operand_count=1, result_count=1)
    field_names = tuple(
        field.name for field in (*source_op.operands, *source_op.results)
    )
    if not any(
        constraint.name == "SameType" and constraint.args == field_names
        for constraint in source_op.constraints
    ):
        raise ValueError(
            f"{source_op.name}: same-type unary projection requires a "
            f"SameType{field_names} constraint"
        )
    if not descriptor_by_type:
        raise ValueError(f"{source_op.name}: same-type projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type),),
            (ScalarType(scalar_type),),
            descriptor_key,
        )
        for scalar_type, descriptor_key in descriptor_by_type.items()
    )


def _same_type_ternary(
    source_op: Op,
    descriptor_by_type: dict[ScalarTypeKind, str | VmSourceOpcode],
) -> tuple[VmSourceLowering, ...]:
    """Projects a same-type ternary Loom op onto concrete VM instructions."""

    _require_fixed_source_shape(source_op, operand_count=3, result_count=1)
    field_names = tuple(
        field.name for field in (*source_op.operands, *source_op.results)
    )
    if not any(
        constraint.name == "SameType" and constraint.args == field_names
        for constraint in source_op.constraints
    ):
        raise ValueError(
            f"{source_op.name}: same-type ternary projection requires a "
            f"SameType{field_names} constraint"
        )
    if not descriptor_by_type:
        raise ValueError(f"{source_op.name}: same-type projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type),) * 3,
            (ScalarType(scalar_type),),
            descriptor_key,
        )
        for scalar_type, descriptor_key in descriptor_by_type.items()
    )


def _scalar_select(source_op: Op) -> tuple[VmSourceLowering, ...]:
    """Projects whole-cell scalar selection onto the VM value bank."""

    _require_fixed_source_shape(source_op, operand_count=3, result_count=1)
    payload_names = tuple(
        field.name for field in (*source_op.operands[1:], *source_op.results)
    )
    if not any(
        constraint.name == "SameType" and constraint.args == payload_names
        for constraint in source_op.constraints
    ):
        raise ValueError(
            f"{source_op.name}: scalar select projection requires a "
            f"SameType{payload_names} constraint"
        )
    return tuple(
        _source_lowering(
            source_op,
            (I1, ScalarType(scalar_type), ScalarType(scalar_type)),
            (ScalarType(scalar_type),),
            "vm.value.select",
        )
        for scalar_type in ScalarTypeKind
    )


def _float_width_opcodes(
    mnemonic_stem: str,
    *,
    selector_table: str | None = None,
    selector_name: str | None = None,
    selector_source_attr: str | None = None,
) -> dict[ScalarTypeKind, VmSourceOpcode]:
    """Builds the f32/f64 pair for one width-selected VM instruction."""

    return {
        scalar_type: VmSourceOpcode(
            f"vm.{mnemonic_stem}.f{width}",
            selector_table,
            selector_name,
            selector_source_attr,
        )
        for scalar_type, width in (
            (ScalarTypeKind.F32, 32),
            (ScalarTypeKind.F64, 64),
        )
    }


def _float_math_unary(
    source_op: Op, selector_name: str
) -> tuple[VmSourceLowering, ...]:
    """Projects one Loom unary math op onto both native VM float widths."""

    return _same_type_unary(
        source_op,
        _float_width_opcodes(
            "float.math.unary",
            selector_table="float.math.unary",
            selector_name=selector_name,
        ),
    )


def _float_math_binary(
    source_op: Op, selector_name: str
) -> tuple[VmSourceLowering, ...]:
    """Projects one Loom binary math op onto both native VM float widths."""

    return _same_type_binary(
        source_op,
        _float_width_opcodes(
            "float.math.binary",
            selector_table="float.math.binary",
            selector_name=selector_name,
        ),
    )


def _float_comparison(source_op: Op) -> tuple[VmSourceLowering, ...]:
    """Projects a Loom float predicate onto both native VM float widths."""

    _require_fixed_source_shape(source_op, operand_count=2, result_count=1)
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type),) * 2,
            (I1,),
            VmSourceOpcode(
                f"vm.float.compare.f{width}",
                "float.compare",
                selector_source_attr="predicate",
            ),
        )
        for scalar_type, width in (
            (ScalarTypeKind.F32, 32),
            (ScalarTypeKind.F64, 64),
        )
    )


def _float_classification(
    source_op: Op, selector_name: str
) -> tuple[VmSourceLowering, ...]:
    """Projects a Loom float classification onto both native VM widths."""

    _require_fixed_source_shape(source_op, operand_count=1, result_count=1)
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type),),
            (I1,),
            VmSourceOpcode(
                f"vm.float.classify.f{width}",
                "float.classify",
                selector_name,
            ),
        )
        for scalar_type, width in (
            (ScalarTypeKind.F32, 32),
            (ScalarTypeKind.F64, 64),
        )
    )


def _selected_cast(
    source_op: Op,
    descriptor_key: str,
    selector_table: str,
    cases: tuple[tuple[ScalarTypeKind, ScalarTypeKind, str], ...],
) -> tuple[VmSourceLowering, ...]:
    """Projects selector-defined source/result type pairs onto one VM record."""

    type_pairs = tuple(
        (source_type, result_type) for source_type, result_type, _ in cases
    )
    if len(type_pairs) != len(set(type_pairs)):
        raise ValueError(f"{source_op.name}: selected cast repeats a type pair")
    return _cast(
        source_op,
        {
            (source_type, result_type): VmSourceOpcode(
                descriptor_key,
                selector_table,
                selector_name,
            )
            for source_type, result_type, selector_name in cases
        },
    )


def _integer_comparison(
    source_op: Op,
    descriptor_by_type: dict[ScalarTypeKind, str],
) -> tuple[VmSourceLowering, ...]:
    """Projects an integer comparison and its predicate onto the VM ISA."""

    _require_fixed_source_shape(source_op, operand_count=2, result_count=1)
    operand_names = tuple(operand.name for operand in source_op.operands)
    if not any(
        constraint.name == "SameType" and constraint.args == operand_names
        for constraint in source_op.constraints
    ):
        raise ValueError(
            f"{source_op.name}: integer comparison projection requires a "
            f"SameType{operand_names} constraint"
        )
    if not descriptor_by_type:
        raise ValueError(f"{source_op.name}: comparison projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(scalar_type), ScalarType(scalar_type)),
            (I1,),
            VmSourceOpcode(
                descriptor_key,
                "integer.compare",
                selector_source_attr="predicate",
            ),
        )
        for scalar_type, descriptor_key in descriptor_by_type.items()
    )


def _cast(
    source_op: Op,
    descriptor_by_type_pair: dict[
        tuple[ScalarTypeKind, ScalarTypeKind], str | VmSourceOpcode
    ],
) -> tuple[VmSourceLowering, ...]:
    """Projects a unary Loom cast onto concrete VM instructions."""

    _require_fixed_source_shape(source_op, operand_count=1, result_count=1)
    if not descriptor_by_type_pair:
        raise ValueError(f"{source_op.name}: cast projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (ScalarType(source_type),),
            (ScalarType(result_type),),
            source_opcode,
        )
        for (
            source_type,
            result_type,
        ), source_opcode in descriptor_by_type_pair.items()
    )


_FIXED_INTEGER_TYPES = (
    ScalarTypeKind.I1,
    ScalarTypeKind.I8,
    ScalarTypeKind.I16,
    ScalarTypeKind.I32,
    ScalarTypeKind.I64,
)

_FLOAT_TYPES = (
    ScalarTypeKind.F8E4M3,
    ScalarTypeKind.F8E5M2,
    ScalarTypeKind.F16,
    ScalarTypeKind.BF16,
    ScalarTypeKind.F32,
    ScalarTypeKind.F64,
)


def _direct_conversion_lowerings() -> tuple[VmSourceLowering, ...]:
    """Projects every conversion implemented by one Core VM instruction."""

    def float_to_integer_cases(
        signedness: str,
    ) -> tuple[tuple[ScalarTypeKind, ScalarTypeKind, str], ...]:
        return tuple(
            (
                source_type,
                result_type,
                f"f{ScalarType(source_type).bitwidth}.to."
                f"{signedness}{ScalarType(result_type).bitwidth}",
            )
            for source_type in (ScalarTypeKind.F32, ScalarTypeKind.F64)
            for result_type in _FIXED_INTEGER_TYPES
        )

    return (
        *_selected_cast(
            scalar_conversion.scalar_extf,
            "vm.conversion.float.extend",
            "float.extend",
            (
                (ScalarTypeKind.F8E4M3, ScalarTypeKind.F32, "f8e4m3.to.f32"),
                (ScalarTypeKind.F8E5M2, ScalarTypeKind.F32, "f8e5m2.to.f32"),
                (ScalarTypeKind.F16, ScalarTypeKind.F32, "f16.to.f32"),
                (ScalarTypeKind.BF16, ScalarTypeKind.F32, "bf16.to.f32"),
            ),
        ),
        *_selected_cast(
            scalar_conversion.scalar_extf,
            "vm.conversion.float.width",
            "float.width",
            ((ScalarTypeKind.F32, ScalarTypeKind.F64, "f32.to.f64"),),
        ),
        *_selected_cast(
            scalar_conversion.scalar_fptrunc,
            "vm.conversion.float.truncate",
            "float.truncate",
            (
                (ScalarTypeKind.F32, ScalarTypeKind.F8E4M3, "f32.to.f8e4m3"),
                (ScalarTypeKind.F32, ScalarTypeKind.F8E5M2, "f32.to.f8e5m2"),
                (ScalarTypeKind.F32, ScalarTypeKind.F16, "f32.to.f16"),
                (ScalarTypeKind.F32, ScalarTypeKind.BF16, "f32.to.bf16"),
                (ScalarTypeKind.F64, ScalarTypeKind.F8E4M3, "f64.to.f8e4m3"),
                (ScalarTypeKind.F64, ScalarTypeKind.F8E5M2, "f64.to.f8e5m2"),
                (ScalarTypeKind.F64, ScalarTypeKind.F16, "f64.to.f16"),
                (ScalarTypeKind.F64, ScalarTypeKind.BF16, "f64.to.bf16"),
            ),
        ),
        *_selected_cast(
            scalar_conversion.scalar_fptrunc,
            "vm.conversion.float.width",
            "float.width",
            ((ScalarTypeKind.F64, ScalarTypeKind.F32, "f64.to.f32"),),
        ),
        *_selected_cast(
            scalar_conversion.scalar_sitofp,
            "vm.conversion.integer.to.float",
            "integer.to.float",
            (
                (ScalarTypeKind.I32, ScalarTypeKind.F32, "s32.to.f32"),
                (ScalarTypeKind.I32, ScalarTypeKind.F64, "s32.to.f64"),
                (ScalarTypeKind.I64, ScalarTypeKind.F32, "s64.to.f32"),
                (ScalarTypeKind.I64, ScalarTypeKind.F64, "s64.to.f64"),
                (ScalarTypeKind.I32, ScalarTypeKind.BF16, "s32.to.bf16"),
                (ScalarTypeKind.I64, ScalarTypeKind.BF16, "s64.to.bf16"),
            ),
        ),
        *_selected_cast(
            scalar_conversion.scalar_uitofp,
            "vm.conversion.integer.to.float",
            "integer.to.float",
            (
                (ScalarTypeKind.I32, ScalarTypeKind.F32, "u32.to.f32"),
                (ScalarTypeKind.I32, ScalarTypeKind.F64, "u32.to.f64"),
                (ScalarTypeKind.I64, ScalarTypeKind.F32, "u64.to.f32"),
                (ScalarTypeKind.I64, ScalarTypeKind.F64, "u64.to.f64"),
                (ScalarTypeKind.I32, ScalarTypeKind.BF16, "u32.to.bf16"),
                (ScalarTypeKind.I64, ScalarTypeKind.BF16, "u64.to.bf16"),
            ),
        ),
        *_selected_cast(
            scalar_conversion.scalar_fptosi,
            "vm.conversion.float.to.integer",
            "float.to.integer",
            float_to_integer_cases("s"),
        ),
        *_selected_cast(
            scalar_conversion.scalar_fptoui,
            "vm.conversion.float.to.integer",
            "float.to.integer",
            float_to_integer_cases("u"),
        ),
        *_selected_cast(
            scalar_conversion.scalar_extsi,
            "vm.conversion.integer",
            "integer.convert",
            _integer_conversion_cases("sign_extend"),
        ),
        *_selected_cast(
            scalar_conversion.scalar_extui,
            "vm.conversion.integer",
            "integer.convert",
            _integer_conversion_cases("zero_extend"),
        ),
        *_selected_cast(
            scalar_conversion.scalar_trunci,
            "vm.conversion.integer",
            "integer.convert",
            _integer_conversion_cases("truncate"),
        ),
    )


VM_DIRECT_CONVERSION_LOWERINGS = _direct_conversion_lowerings()

_DIRECT_CONVERSION_BY_SIGNATURE = {
    (
        row.source_op,
        row.operand_types[0].kind,
        row.result_types[0].kind,
    ): row
    for row in VM_DIRECT_CONVERSION_LOWERINGS
}


def _composed_conversion_lowering(
    source_op: Op,
    source_type: ScalarTypeKind,
    result_type: ScalarTypeKind,
    step_signatures: tuple[tuple[Op, ScalarTypeKind, ScalarTypeKind], ...],
) -> VmSourceConversionLowering:
    """Builds and validates one lowering from two or three direct steps."""

    _require_fixed_source_shape(source_op, operand_count=1, result_count=1)
    _require_concrete_source_types(
        source_op,
        (ScalarType(source_type),),
        (ScalarType(result_type),),
    )
    try:
        steps = tuple(
            _DIRECT_CONVERSION_BY_SIGNATURE[signature] for signature in step_signatures
        )
    except KeyError as exc:
        raise ValueError(
            f"{source_op.name}: composed VM conversion references a non-direct step"
        ) from exc
    return VmSourceConversionLowering(source_op, source_type, result_type, steps)


def _composed_conversion_lowerings() -> tuple[VmSourceConversionLowering, ...]:
    """Builds every semantics-preserving multi-instruction conversion route."""

    lowerings: list[VmSourceConversionLowering] = []

    for source_op, extension_op in (
        (scalar_conversion.scalar_sitofp, scalar_conversion.scalar_extsi),
        (scalar_conversion.scalar_uitofp, scalar_conversion.scalar_extui),
    ):
        for source_type in _FIXED_INTEGER_TYPES:
            for result_type in _FLOAT_TYPES:
                signature = (source_op, source_type, result_type)
                if signature in _DIRECT_CONVERSION_BY_SIGNATURE:
                    continue
                working_type = source_type
                steps: list[tuple[Op, ScalarTypeKind, ScalarTypeKind]] = []
                if ScalarType(source_type).bitwidth < 32:
                    steps.append((extension_op, source_type, ScalarTypeKind.I32))
                    working_type = ScalarTypeKind.I32
                if result_type in (
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.F8E5M2,
                    ScalarTypeKind.F16,
                ):
                    # Every integer that can affect a finite f8/f16 result or
                    # its overflow boundary is exactly representable in f32.
                    # BF16 spans a much wider exponent range and therefore has
                    # direct integer conversion selectors instead.
                    steps.extend(
                        (
                            (source_op, working_type, ScalarTypeKind.F32),
                            (
                                scalar_conversion.scalar_fptrunc,
                                ScalarTypeKind.F32,
                                result_type,
                            ),
                        )
                    )
                else:
                    steps.append((source_op, working_type, result_type))
                lowerings.append(
                    _composed_conversion_lowering(
                        source_op,
                        source_type,
                        result_type,
                        tuple(steps),
                    )
                )

    for source_op in (
        scalar_conversion.scalar_fptosi,
        scalar_conversion.scalar_fptoui,
    ):
        for source_type in _FLOAT_TYPES:
            for result_type in _FIXED_INTEGER_TYPES:
                signature = (source_op, source_type, result_type)
                if signature in _DIRECT_CONVERSION_BY_SIGNATURE:
                    continue
                lowerings.append(
                    _composed_conversion_lowering(
                        source_op,
                        source_type,
                        result_type,
                        (
                            (
                                scalar_conversion.scalar_extf,
                                source_type,
                                ScalarTypeKind.F32,
                            ),
                            (source_op, ScalarTypeKind.F32, result_type),
                        ),
                    )
                )

    for source_type in _FLOAT_TYPES:
        for result_type in _FLOAT_TYPES:
            if ScalarType(result_type).bitwidth <= ScalarType(source_type).bitwidth:
                continue
            signature = (scalar_conversion.scalar_extf, source_type, result_type)
            if signature in _DIRECT_CONVERSION_BY_SIGNATURE:
                continue
            second_op = (
                scalar_conversion.scalar_extf
                if result_type is ScalarTypeKind.F64
                else scalar_conversion.scalar_fptrunc
            )
            lowerings.append(
                _composed_conversion_lowering(
                    scalar_conversion.scalar_extf,
                    source_type,
                    result_type,
                    (
                        (
                            scalar_conversion.scalar_extf,
                            source_type,
                            ScalarTypeKind.F32,
                        ),
                        (second_op, ScalarTypeKind.F32, result_type),
                    ),
                )
            )

    for source_type in _FLOAT_TYPES:
        for result_type in _FLOAT_TYPES:
            if ScalarType(result_type).bitwidth >= ScalarType(source_type).bitwidth:
                continue
            signature = (
                scalar_conversion.scalar_fptrunc,
                source_type,
                result_type,
            )
            if signature in _DIRECT_CONVERSION_BY_SIGNATURE:
                continue
            lowerings.append(
                _composed_conversion_lowering(
                    scalar_conversion.scalar_fptrunc,
                    source_type,
                    result_type,
                    (
                        (
                            scalar_conversion.scalar_extf,
                            source_type,
                            ScalarTypeKind.F32,
                        ),
                        (
                            scalar_conversion.scalar_fptrunc,
                            ScalarTypeKind.F32,
                            result_type,
                        ),
                    ),
                )
            )

    signatures = tuple(
        (lowering.source_op.name, lowering.source_type, lowering.result_type)
        for lowering in lowerings
    )
    if len(signatures) != len(set(signatures)):
        raise ValueError("composed VM conversions repeat a source signature")
    return tuple(lowerings)


def _conversion_lowerings() -> tuple[VmSourceConversionLowering, ...]:
    """Combines direct and composed routes for every scalar conversion."""

    direct_lowerings = tuple(
        VmSourceConversionLowering(
            row.source_op,
            row.operand_types[0].kind,
            row.result_types[0].kind,
            (row,),
        )
        for row in VM_DIRECT_CONVERSION_LOWERINGS
    )
    composed_lowerings = _composed_conversion_lowerings()
    source_ops = tuple(dict.fromkeys(row.source_op for row in direct_lowerings))
    lowerings = tuple(
        lowering
        for source_op in source_ops
        for lowering in sorted(
            (
                row
                for row in (*direct_lowerings, *composed_lowerings)
                if row.source_op is source_op
            ),
            key=lambda row: (row.source_type, row.result_type),
        )
    )
    signatures = tuple(
        (row.source_op.name, row.source_type, row.result_type) for row in lowerings
    )
    if len(signatures) != len(set(signatures)):
        raise ValueError("VM conversion lowerings repeat a source signature")
    return lowerings


VM_SOURCE_CONVERSION_LOWERINGS = _conversion_lowerings()


VM_SOURCE_LOWERINGS = (
    _source_lowering(
        cfg_defs.cfg_assert,
        (I1, BUFFER_TYPE),
        (),
        "vm.control.assert",
    ),
    _source_lowering(
        func_defs.func_fail,
        (BUFFER_TYPE,),
        (),
        VmSourceOpcode(
            "vm.control.fail",
            "control.status",
            selector_source_attr="status",
        ),
    ),
    *_scalar_select(scf_defs.scf_select),
    *_same_type_binary(
        index_defs.index_add,
        {
            ScalarTypeKind.INDEX: "vm.integer.add.i64",
            ScalarTypeKind.OFFSET: "vm.integer.add.i64",
        },
    ),
    *_same_type_binary(
        index_defs.index_sub,
        {
            ScalarTypeKind.INDEX: "vm.integer.sub.i64",
            ScalarTypeKind.OFFSET: "vm.integer.sub.i64",
        },
    ),
    *_same_type_binary(
        index_defs.index_mul,
        {ScalarTypeKind.INDEX: "vm.integer.mul.i64"},
    ),
    _source_lowering(
        index_defs.index_scale,
        (
            ScalarType(ScalarTypeKind.INDEX),
            ScalarType(ScalarTypeKind.OFFSET),
        ),
        (ScalarType(ScalarTypeKind.OFFSET),),
        "vm.integer.mul.i64",
    ),
    *_same_type_binary(
        index_defs.index_div,
        {ScalarTypeKind.INDEX: "vm.integer.div.u64"},
    ),
    *_same_type_binary(
        index_defs.index_rem,
        {ScalarTypeKind.INDEX: "vm.integer.rem.u64"},
    ),
    *_same_type_binary(
        index_defs.index_min,
        {ScalarTypeKind.INDEX: "vm.integer.min.s64"},
    ),
    *_same_type_binary(
        index_defs.index_max,
        {ScalarTypeKind.INDEX: "vm.integer.max.s64"},
    ),
    *_same_type_binary(
        index_defs.index_andi,
        {ScalarTypeKind.INDEX: "vm.integer.and.i64"},
    ),
    *_same_type_binary(
        index_defs.index_ori,
        {ScalarTypeKind.INDEX: "vm.integer.or.i64"},
    ),
    *_same_type_binary(
        index_defs.index_xori,
        {ScalarTypeKind.INDEX: "vm.integer.xor.i64"},
    ),
    *_same_type_binary(
        index_defs.index_shli,
        {ScalarTypeKind.INDEX: "vm.integer.shift.left.i64"},
    ),
    *_same_type_binary(
        index_defs.index_shrsi,
        {ScalarTypeKind.INDEX: "vm.integer.shift.right.s64"},
    ),
    *_same_type_binary(
        index_defs.index_shrui,
        {ScalarTypeKind.INDEX: "vm.integer.shift.right.u64"},
    ),
    *_same_type_binary(
        index_defs.index_rotli,
        {ScalarTypeKind.INDEX: "vm.integer.rotate.left.i64"},
    ),
    *_same_type_binary(
        index_defs.index_rotri,
        {ScalarTypeKind.INDEX: "vm.integer.rotate.right.i64"},
    ),
    *_same_type_unary(
        index_defs.index_ctlzi,
        {ScalarTypeKind.INDEX: "vm.integer.count.leading.zeros.i64"},
    ),
    *_same_type_unary(
        index_defs.index_cttzi,
        {ScalarTypeKind.INDEX: "vm.integer.count.trailing.zeros.i64"},
    ),
    *_same_type_unary(
        index_defs.index_ctpopi,
        {ScalarTypeKind.INDEX: "vm.integer.popcount.i64"},
    ),
    *_integer_comparison(
        index_defs.index_cmp,
        {
            ScalarTypeKind.INDEX: "vm.integer.compare.i64",
            ScalarTypeKind.OFFSET: "vm.integer.compare.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_addi,
        {
            ScalarTypeKind.I32: "vm.integer.add.i32",
            ScalarTypeKind.I64: "vm.integer.add.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_subi,
        {
            ScalarTypeKind.I32: "vm.integer.sub.i32",
            ScalarTypeKind.I64: "vm.integer.sub.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_muli,
        {
            ScalarTypeKind.I32: "vm.integer.mul.i32",
            ScalarTypeKind.I64: "vm.integer.mul.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_divsi,
        {
            ScalarTypeKind.I32: "vm.integer.div.s32",
            ScalarTypeKind.I64: "vm.integer.div.s64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_divui,
        {
            ScalarTypeKind.I32: "vm.integer.div.u32",
            ScalarTypeKind.I64: "vm.integer.div.u64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_remsi,
        {
            ScalarTypeKind.I32: "vm.integer.rem.s32",
            ScalarTypeKind.I64: "vm.integer.rem.s64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_remui,
        {
            ScalarTypeKind.I32: "vm.integer.rem.u32",
            ScalarTypeKind.I64: "vm.integer.rem.u64",
        },
    ),
    *_same_type_unary(
        scalar_arithmetic.scalar_negi,
        {
            ScalarTypeKind.I32: "vm.integer.neg.i32",
            ScalarTypeKind.I64: "vm.integer.neg.i64",
        },
    ),
    *_same_type_unary(
        scalar_arithmetic.scalar_absi,
        {
            ScalarTypeKind.I32: "vm.integer.abs.s32",
            ScalarTypeKind.I64: "vm.integer.abs.s64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_minsi,
        {
            ScalarTypeKind.I32: "vm.integer.min.s32",
            ScalarTypeKind.I64: "vm.integer.min.s64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_maxsi,
        {
            ScalarTypeKind.I32: "vm.integer.max.s32",
            ScalarTypeKind.I64: "vm.integer.max.s64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_minui,
        {
            ScalarTypeKind.I32: "vm.integer.min.u32",
            ScalarTypeKind.I64: "vm.integer.min.u64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_maxui,
        {
            ScalarTypeKind.I32: "vm.integer.max.u32",
            ScalarTypeKind.I64: "vm.integer.max.u64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_andi,
        {
            ScalarTypeKind.I1: "vm.integer.and.i32",
            ScalarTypeKind.I32: "vm.integer.and.i32",
            ScalarTypeKind.I64: "vm.integer.and.i64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_ori,
        {
            ScalarTypeKind.I1: "vm.integer.or.i32",
            ScalarTypeKind.I32: "vm.integer.or.i32",
            ScalarTypeKind.I64: "vm.integer.or.i64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_xori,
        {
            ScalarTypeKind.I1: "vm.integer.xor.i32",
            ScalarTypeKind.I32: "vm.integer.xor.i32",
            ScalarTypeKind.I64: "vm.integer.xor.i64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_shli,
        {
            ScalarTypeKind.I32: "vm.integer.shift.left.i32",
            ScalarTypeKind.I64: "vm.integer.shift.left.i64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_shrsi,
        {
            ScalarTypeKind.I32: "vm.integer.shift.right.s32",
            ScalarTypeKind.I64: "vm.integer.shift.right.s64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_shrui,
        {
            ScalarTypeKind.I32: "vm.integer.shift.right.u32",
            ScalarTypeKind.I64: "vm.integer.shift.right.u64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_rotli,
        {
            ScalarTypeKind.I32: "vm.integer.rotate.left.i32",
            ScalarTypeKind.I64: "vm.integer.rotate.left.i64",
        },
    ),
    *_same_type_binary(
        scalar_bitwise.scalar_rotri,
        {
            ScalarTypeKind.I32: "vm.integer.rotate.right.i32",
            ScalarTypeKind.I64: "vm.integer.rotate.right.i64",
        },
    ),
    *_same_type_unary(
        scalar_bitwise.scalar_ctlzi,
        {
            ScalarTypeKind.I32: "vm.integer.count.leading.zeros.i32",
            ScalarTypeKind.I64: "vm.integer.count.leading.zeros.i64",
        },
    ),
    *_same_type_unary(
        scalar_bitwise.scalar_cttzi,
        {
            ScalarTypeKind.I32: "vm.integer.count.trailing.zeros.i32",
            ScalarTypeKind.I64: "vm.integer.count.trailing.zeros.i64",
        },
    ),
    *_same_type_unary(
        scalar_bitwise.scalar_ctpopi,
        {
            ScalarTypeKind.I32: "vm.integer.popcount.i32",
            ScalarTypeKind.I64: "vm.integer.popcount.i64",
        },
    ),
    *_integer_comparison(
        scalar_comparison.scalar_cmpi,
        {
            ScalarTypeKind.I32: "vm.integer.compare.i32",
            ScalarTypeKind.I64: "vm.integer.compare.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_addf,
        _float_width_opcodes("float.add"),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_subf,
        _float_width_opcodes("float.sub"),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_mulf,
        _float_width_opcodes("float.mul"),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_divf,
        _float_width_opcodes("float.div"),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_remf,
        _float_width_opcodes("float.rem"),
    ),
    *_same_type_unary(
        scalar_arithmetic.scalar_negf,
        _float_width_opcodes("float.neg"),
    ),
    *_same_type_unary(
        scalar_arithmetic.scalar_absf,
        _float_width_opcodes("float.abs"),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_minimumf,
        _float_width_opcodes(
            "float.minmax",
            selector_table="float.minmax",
            selector_name="minimum",
        ),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_maximumf,
        _float_width_opcodes(
            "float.minmax",
            selector_table="float.minmax",
            selector_name="maximum",
        ),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_minnumf,
        _float_width_opcodes(
            "float.minmax",
            selector_table="float.minmax",
            selector_name="minnum",
        ),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_maxnumf,
        _float_width_opcodes(
            "float.minmax",
            selector_table="float.minmax",
            selector_name="maxnum",
        ),
    ),
    *_same_type_ternary(
        scalar_arithmetic.scalar_clampf,
        _float_width_opcodes(
            "float.clamp",
            selector_table="float.clamp",
            selector_source_attr="mode",
        ),
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_copysignf,
        _float_width_opcodes("float.copysign"),
    ),
    *_float_comparison(scalar_comparison.scalar_cmpf),
    *_float_classification(scalar_comparison.scalar_isnanf, "isnan"),
    *_float_classification(scalar_comparison.scalar_isinff, "isinf"),
    *_float_classification(scalar_comparison.scalar_isfinitef, "isfinite"),
    *_float_math_unary(scalar_comparison.scalar_signf, "sign"),
    *_float_math_unary(scalar_math.scalar_expf, "exp"),
    *_float_math_unary(scalar_math.scalar_exp2f, "exp2"),
    *_float_math_unary(scalar_math.scalar_expm1f, "expm1"),
    *_float_math_unary(scalar_math.scalar_logf, "log"),
    *_float_math_unary(scalar_math.scalar_log2f, "log2"),
    *_float_math_unary(scalar_math.scalar_log10f, "log10"),
    *_float_math_unary(scalar_math.scalar_log1pf, "log1p"),
    *_float_math_binary(scalar_math.scalar_powf, "pow"),
    *_float_math_unary(scalar_math.scalar_sqrtf, "sqrt"),
    *_float_math_unary(scalar_math.scalar_rsqrtf, "rsqrt"),
    *_float_math_unary(scalar_math.scalar_cbrtf, "cbrt"),
    *_float_math_unary(scalar_math.scalar_sinf, "sin"),
    *_float_math_unary(scalar_math.scalar_cosf, "cos"),
    *_float_math_unary(scalar_math.scalar_sinturnsf, "sinturns"),
    *_float_math_unary(scalar_math.scalar_costurnsf, "costurns"),
    *_float_math_unary(scalar_math.scalar_tanf, "tan"),
    *_float_math_unary(scalar_math.scalar_asinf, "asin"),
    *_float_math_unary(scalar_math.scalar_acosf, "acos"),
    *_float_math_unary(scalar_math.scalar_atanf, "atan"),
    *_float_math_binary(scalar_math.scalar_atan2f, "atan2"),
    *_float_math_unary(scalar_math.scalar_sinhf, "sinh"),
    *_float_math_unary(scalar_math.scalar_coshf, "cosh"),
    *_float_math_unary(scalar_math.scalar_tanhf, "tanh"),
    *_float_math_unary(scalar_math.scalar_asinhf, "asinh"),
    *_float_math_unary(scalar_math.scalar_acoshf, "acosh"),
    *_float_math_unary(scalar_math.scalar_atanhf, "atanh"),
    *_float_math_unary(scalar_math.scalar_erff, "erf"),
    *_float_math_unary(scalar_math.scalar_erfcf, "erfc"),
    *_float_math_unary(scalar_math.scalar_logisticf, "logistic"),
    *_float_math_unary(scalar_math.scalar_siluf, "silu"),
    *_float_math_unary(scalar_math.scalar_softplusf, "softplus"),
    *_float_math_unary(scalar_math.scalar_ceilf, "ceil"),
    *_float_math_unary(scalar_math.scalar_floorf, "floor"),
    *_float_math_unary(scalar_math.scalar_roundf, "round"),
    *_float_math_unary(scalar_math.scalar_roundevenf, "roundeven"),
    *_float_math_unary(scalar_math.scalar_truncf, "trunc"),
    *_same_type_ternary(
        scalar_math.scalar_fmaf,
        _float_width_opcodes(
            "float.math.ternary",
            selector_table="float.math.ternary",
            selector_name="fma",
        ),
    ),
    *_cast(
        scalar_conversion.scalar_bitcast,
        {
            (source_type, result_type): "vm.value.copy"
            for same_width_types in (
                (
                    ScalarTypeKind.I8,
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.F8E5M2,
                ),
                (ScalarTypeKind.I16, ScalarTypeKind.F16, ScalarTypeKind.BF16),
                (ScalarTypeKind.I32, ScalarTypeKind.F32),
                (ScalarTypeKind.I64, ScalarTypeKind.F64),
            )
            for source_type in same_width_types
            for result_type in same_width_types
            if source_type != result_type
        },
    ),
    *_cast(
        index_defs.index_cast,
        _index_cast_opcodes(),
    ),
)
