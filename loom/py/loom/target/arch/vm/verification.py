# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Projects Core VM packet constraints into a compact verifier program."""

from __future__ import annotations

import dataclasses
import enum

from model.isa import Instruction, InstructionField
from model.isa.selectors import (
    SELECTOR_TABLES_BY_NAME,
    SELECTOR_VALUES,
    memory_format_lane_count,
)
from model.isa.validation import (
    ALLOWED_RANGE,
    FIELDS_DISTINCT,
    FUNCTION_ADDRESS,
    INTEGER_BITSTREAM_SHAPE,
    PACKED_SELECTOR_ALLOWED_PAIRS,
    PACKED_SELECTOR_TARGET_SUPPORTED,
    PACKED_SELECTORS,
    VALUE_REGISTER_RANGE,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
)
from model.schema import EntityReference, FieldReference, RuleUse

from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
)
from loom.target.low_descriptors import Descriptor


class VmPacketConstraintKind(enum.Enum):
    """One operation implemented by the target-low packet verifier."""

    IMMEDIATE_NONZERO = "immediate_nonzero"
    MEMORY_FORMAT_UNIT_COUNT = "memory_format_unit_count"
    INTEGER_BITSTREAM_PACK = "integer_bitstream_pack"
    INTEGER_BITSTREAM_UNPACK = "integer_bitstream_unpack"
    PACKED_IMMEDIATE_MASK = "packed_immediate_mask"


@dataclasses.dataclass(frozen=True, slots=True)
class VmPacketConstraint:
    """One compact verifier operation within a descriptor constraint range."""

    kind: VmPacketConstraintKind
    arguments: tuple[int, ...]
    parameter: int = 0

    def __post_init__(self) -> None:
        if len(self.arguments) > 7:
            raise ValueError("VM packet constraint exceeds seven u8 arguments")
        if any(not 0 <= argument <= 0xFF for argument in self.arguments):
            raise ValueError("VM packet constraint argument exceeds u8")
        if not 0 <= self.parameter <= 0xFFFFFFFF:
            raise ValueError("VM packet constraint parameter exceeds u32")


@dataclasses.dataclass(frozen=True, slots=True)
class VmDeferredConstraint:
    """One ISA constraint enforced after target-low packet verification."""

    descriptor_ordinal: int
    rule_id: str


def _field(instruction: Instruction, field_name: str) -> InstructionField:
    fields = tuple(field for field in instruction.fields if field.name == field_name)
    if len(fields) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: expected one field named {field_name!r}"
        )
    return fields[0]


def _rule_use(field: InstructionField, rule_id: str) -> RuleUse | None:
    rules = tuple(rule for rule in field.validation if rule.rule_id == rule_id)
    if len(rules) > 1:
        raise ValueError(f"{field.name}: validation rule {rule_id} is repeated")
    return rules[0] if rules else None


def _field_reference(value: object, description: str) -> str:
    if not isinstance(value, FieldReference):
        raise ValueError(f"{description} is not a field reference")
    return value.field_name


def _descriptor_operand_ordinal(
    instruction: Instruction,
    descriptor: Descriptor,
    field_name: str,
) -> int:
    field = _field(instruction, field_name)
    ordinals = tuple(
        ordinal
        for ordinal, operand in enumerate(descriptor.operands)
        if operand.encoding_field_id == field.offset
    )
    if len(ordinals) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: field {field_name!r} does not select one "
            "descriptor operand"
        )
    return ordinals[0]


def _descriptor_immediate_ordinal(
    instruction: Instruction,
    descriptor: Descriptor,
    field_name: str,
) -> int:
    field = _field(instruction, field_name)
    if field.array_length != 1:
        raise ValueError(
            f"{instruction.mnemonic}: constrained immediate {field_name!r} is an array"
        )
    ordinals = tuple(
        ordinal
        for ordinal, immediate in enumerate(descriptor.immediates)
        if immediate.encoding_field_id == field.offset
    )
    if len(ordinals) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: field {field_name!r} does not select one "
            "descriptor immediate"
        )
    return ordinals[0]


_SELECTOR_VALUES_BY_TABLE_ID = {
    table.entity_id: frozenset(
        value.value for value in SELECTOR_VALUES if value.table_id == table.entity_id
    )
    for table in SELECTOR_TABLES_BY_NAME.values()
}


def _packed_component_values(
    packed_value: int,
    components: tuple[object, ...],
) -> dict[str, int] | None:
    values: dict[str, int] = {}
    for component in components:
        if not isinstance(component, tuple) or len(component) != 5:
            raise ValueError("malformed packed-selector component")
        name, bit_offset, bit_length, table_reference, allowed_values = component
        if not isinstance(name, str):
            raise ValueError("packed-selector component name is not text")
        if not isinstance(bit_offset, int) or not isinstance(bit_length, int):
            raise ValueError(f"packed-selector component {name!r} has invalid bits")
        if not isinstance(table_reference, EntityReference):
            raise ValueError(
                f"packed-selector component {name!r} has no selector table"
            )
        if not isinstance(allowed_values, tuple):
            raise ValueError(
                f"packed-selector component {name!r} has invalid allowed values"
            )
        value = (packed_value >> bit_offset) & ((1 << bit_length) - 1)
        table_values = _SELECTOR_VALUES_BY_TABLE_ID[table_reference.entity_id]
        if value not in table_values:
            return None
        if allowed_values and value not in allowed_values:
            return None
        values[name] = value
    return values


def _packed_value_satisfies_pairs(
    instruction: Instruction,
    field_name: str,
    component_values: dict[str, int],
) -> bool:
    for constraint in instruction.constraints:
        if constraint.rule_id != PACKED_SELECTOR_ALLOWED_PAIRS.entity_id:
            continue
        if len(constraint.arguments) != 4:
            raise ValueError(
                f"{instruction.mnemonic}: malformed packed-selector pair constraint"
            )
        constraint_field = _field_reference(
            constraint.arguments[0],
            f"{instruction.mnemonic} packed-selector pair field",
        )
        if constraint_field != field_name:
            continue
        first_component = constraint.arguments[1]
        second_component = constraint.arguments[2]
        allowed_pairs = constraint.arguments[3]
        if not isinstance(first_component, str) or not isinstance(
            second_component, str
        ):
            raise ValueError(
                f"{instruction.mnemonic}: packed-selector pair names are invalid"
            )
        if not isinstance(allowed_pairs, tuple):
            raise ValueError(
                f"{instruction.mnemonic}: packed-selector pairs are invalid"
            )
        actual_pair = (
            component_values[first_component],
            component_values[second_component],
        )
        if actual_pair not in allowed_pairs:
            return False
    return True


def _packed_selector_mask(
    instruction: Instruction,
    field: InstructionField,
) -> tuple[int, int, int, int]:
    rule = _rule_use(field, PACKED_SELECTORS.entity_id)
    if rule is None or len(rule.arguments) != 2:
        raise ValueError(
            f"{instruction.mnemonic}.{field.name}: malformed packed selector"
        )
    zero_mask, components = rule.arguments
    if not isinstance(zero_mask, int) or not isinstance(components, tuple):
        raise ValueError(
            f"{instruction.mnemonic}.{field.name}: malformed packed selector"
        )
    words = [0, 0, 0, 0]
    for packed_value in range(256):
        if packed_value & zero_mask:
            continue
        component_values = _packed_component_values(packed_value, components)
        if component_values is None or not _packed_value_satisfies_pairs(
            instruction, field.name, component_values
        ):
            continue
        words[packed_value // 64] |= 1 << (packed_value % 64)
    if not any(words):
        raise ValueError(
            f"{instruction.mnemonic}.{field.name}: packed selector accepts no values"
        )
    return (words[0], words[1], words[2], words[3])


def _memory_format_unit_counts() -> tuple[int, ...]:
    table_id = SELECTOR_TABLES_BY_NAME["memory.format"].entity_id
    values = tuple(value for value in SELECTOR_VALUES if value.table_id == table_id)
    if not values:
        raise ValueError("memory-format selector table is empty")
    maximum_value = max(value.value for value in values)
    counts = [0] * (maximum_value + 1)
    for value in values:
        counts[value.value] = memory_format_lane_count(value.value)
    if any(count == 0 for count in counts):
        raise ValueError("memory-format selector values are not dense")
    return tuple(counts)


VM_MEMORY_FORMAT_UNIT_COUNTS = _memory_format_unit_counts()


def _bitstream_range_bases(instruction: Instruction) -> dict[str, str]:
    bases_by_count: dict[str, str] = {}
    for constraint in instruction.constraints:
        if constraint.rule_id != VALUE_REGISTER_RANGE.entity_id:
            continue
        if len(constraint.arguments) != 2:
            raise ValueError(f"{instruction.mnemonic}: malformed value-register range")
        base_field = _field_reference(
            constraint.arguments[0], f"{instruction.mnemonic} range base"
        )
        count_field = _field_reference(
            constraint.arguments[1], f"{instruction.mnemonic} range count"
        )
        if count_field in bases_by_count:
            raise ValueError(
                f"{instruction.mnemonic}: repeated range count {count_field!r}"
            )
        bases_by_count[count_field] = base_field
    return bases_by_count


def _project_bitstream_constraint(
    instruction: Instruction,
    descriptor: Descriptor,
    constraint: RuleUse,
) -> VmPacketConstraint:
    if len(constraint.arguments) != 7:
        raise ValueError(f"{instruction.mnemonic}: malformed bitstream constraint")
    mode = constraint.arguments[0]
    maximum_bit_count = constraint.arguments[1]
    if mode not in ("pack", "unpack") or not isinstance(maximum_bit_count, int):
        raise ValueError(f"{instruction.mnemonic}: malformed bitstream mode")
    field_width = _field_reference(
        constraint.arguments[2], f"{instruction.mnemonic} field width"
    )
    source_count = _field_reference(
        constraint.arguments[3], f"{instruction.mnemonic} source count"
    )
    result_count = _field_reference(
        constraint.arguments[4], f"{instruction.mnemonic} result count"
    )
    source_width = _field_reference(
        constraint.arguments[5], f"{instruction.mnemonic} source width"
    )
    result_width = _field_reference(
        constraint.arguments[6], f"{instruction.mnemonic} result width"
    )
    bases_by_count = _bitstream_range_bases(instruction)
    if set(bases_by_count) != {source_count, result_count}:
        raise ValueError(
            f"{instruction.mnemonic}: bitstream ranges do not match shape counts"
        )
    kind = (
        VmPacketConstraintKind.INTEGER_BITSTREAM_PACK
        if mode == "pack"
        else VmPacketConstraintKind.INTEGER_BITSTREAM_UNPACK
    )
    return VmPacketConstraint(
        kind,
        (
            _descriptor_operand_ordinal(
                instruction, descriptor, bases_by_count[result_count]
            ),
            _descriptor_operand_ordinal(
                instruction, descriptor, bases_by_count[source_count]
            ),
            _descriptor_immediate_ordinal(instruction, descriptor, field_width),
            _descriptor_immediate_ordinal(instruction, descriptor, source_count),
            _descriptor_immediate_ordinal(instruction, descriptor, result_count),
            _descriptor_immediate_ordinal(instruction, descriptor, source_width),
            _descriptor_immediate_ordinal(instruction, descriptor, result_width),
        ),
        maximum_bit_count,
    )


def _project_constraints() -> tuple[
    tuple[VmPacketConstraint, ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int, int], ...],
    tuple[VmDeferredConstraint, ...],
    tuple[VmDeferredConstraint, ...],
]:
    constraints: list[VmPacketConstraint] = []
    ranges: list[tuple[int, int]] = []
    packed_masks: list[tuple[int, int, int, int]] = []
    target_dependent: list[VmDeferredConstraint] = []
    module_dependent: list[VmDeferredConstraint] = []
    for descriptor_ordinal, (projection, descriptor) in enumerate(
        zip(
            VM_INSTRUCTION_PROJECTIONS,
            VM_CORE_DESCRIPTOR_SET.descriptors,
            strict=True,
        )
    ):
        instruction = projection.instruction
        constraint_start = len(constraints)

        for field in instruction.fields:
            allowed_range = _rule_use(field, ALLOWED_RANGE.entity_id)
            if allowed_range is not None:
                if len(allowed_range.arguments) != 2:
                    raise ValueError(
                        f"{instruction.mnemonic}.{field.name}: malformed allowed range"
                    )
                minimum = int(allowed_range.arguments[0])
                if minimum != 0:
                    if minimum != 1:
                        raise ValueError(
                            f"{instruction.mnemonic}.{field.name}: positive immediate "
                            "minimum is not representable by the verifier"
                        )
                    constraints.append(
                        VmPacketConstraint(
                            VmPacketConstraintKind.IMMEDIATE_NONZERO,
                            (
                                _descriptor_immediate_ordinal(
                                    instruction, descriptor, field.name
                                ),
                            ),
                        )
                    )

            if _rule_use(field, PACKED_SELECTORS.entity_id) is not None:
                packed_mask = _packed_selector_mask(instruction, field)
                try:
                    mask_ordinal = packed_masks.index(packed_mask)
                except ValueError:
                    mask_ordinal = len(packed_masks)
                    packed_masks.append(packed_mask)
                constraints.append(
                    VmPacketConstraint(
                        VmPacketConstraintKind.PACKED_IMMEDIATE_MASK,
                        (
                            _descriptor_immediate_ordinal(
                                instruction, descriptor, field.name
                            ),
                            mask_ordinal,
                        ),
                    )
                )

        for constraint in instruction.constraints:
            if constraint.rule_id == VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id:
                if len(constraint.arguments) != 3:
                    raise ValueError(
                        f"{instruction.mnemonic}: malformed memory-format range"
                    )
                base_field = _field_reference(
                    constraint.arguments[0],
                    f"{instruction.mnemonic} memory-format base",
                )
                format_field = _field_reference(
                    constraint.arguments[1],
                    f"{instruction.mnemonic} memory-format selector",
                )
                maximum_unit_count = int(constraint.arguments[2])
                if maximum_unit_count != max(VM_MEMORY_FORMAT_UNIT_COUNTS):
                    raise ValueError(
                        f"{instruction.mnemonic}: memory-format maximum drifted"
                    )
                constraints.append(
                    VmPacketConstraint(
                        VmPacketConstraintKind.MEMORY_FORMAT_UNIT_COUNT,
                        (
                            _descriptor_operand_ordinal(
                                instruction, descriptor, base_field
                            ),
                            _descriptor_immediate_ordinal(
                                instruction, descriptor, format_field
                            ),
                        ),
                        maximum_unit_count,
                    )
                )
            elif constraint.rule_id == INTEGER_BITSTREAM_SHAPE.entity_id:
                constraints.append(
                    _project_bitstream_constraint(instruction, descriptor, constraint)
                )
            elif constraint.rule_id in (
                VALUE_REGISTER_RANGE.entity_id,
                PACKED_SELECTOR_ALLOWED_PAIRS.entity_id,
                FIELDS_DISTINCT.entity_id,
            ):
                # These are consumed by a related packet constraint or by the
                # generic Low allocation contract.
                continue
            elif constraint.rule_id == PACKED_SELECTOR_TARGET_SUPPORTED.entity_id:
                target_dependent.append(
                    VmDeferredConstraint(descriptor_ordinal, constraint.rule_id)
                )
            elif constraint.rule_id == FUNCTION_ADDRESS.entity_id:
                module_dependent.append(
                    VmDeferredConstraint(descriptor_ordinal, constraint.rule_id)
                )
            else:
                raise ValueError(
                    f"{instruction.mnemonic}: unclassified record constraint "
                    f"{constraint.rule_id}"
                )

        constraint_count = len(constraints) - constraint_start
        if constraint_start > 0xFF or constraint_count > 0xFF:
            raise ValueError("VM packet constraint range exceeds u8")
        ranges.append((constraint_start, constraint_count))

    if len(constraints) > 0xFF or len(packed_masks) > 0xFF:
        raise ValueError("VM packet verifier tables exceed u8 ordinals")
    return (
        tuple(constraints),
        tuple(ranges),
        tuple(packed_masks),
        tuple(target_dependent),
        tuple(module_dependent),
    )


(
    VM_PACKET_CONSTRAINTS,
    VM_PACKET_CONSTRAINT_RANGES,
    VM_PACKED_IMMEDIATE_MASKS,
    VM_TARGET_DEPENDENT_CONSTRAINTS,
    VM_MODULE_DEPENDENT_CONSTRAINTS,
) = _project_constraints()
