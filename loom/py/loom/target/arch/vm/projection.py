# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM ISA projection into target-Low descriptors."""

from __future__ import annotations

import dataclasses
import enum
import re
from pathlib import Path

from model.isa import (
    ControlFlow,
    Instruction,
    InstructionField,
    InstructionFieldRole,
    RefOwnership,
    StateAccess,
    StateResource,
    Suspension,
)
from model.isa.core.abi import INSTRUCTIONS as ABI_INSTRUCTIONS
from model.isa.core.buffer import INSTRUCTIONS as BUFFER_INSTRUCTIONS
from model.isa.core.constant import INSTRUCTIONS as CONSTANT_INSTRUCTIONS
from model.isa.core.control import INSTRUCTIONS as CONTROL_INSTRUCTIONS
from model.isa.core.conversion import INSTRUCTIONS as CONVERSION_INSTRUCTIONS
from model.isa.core.float import INSTRUCTIONS as FLOAT_INSTRUCTIONS
from model.isa.core.function import INSTRUCTIONS as FUNCTION_INSTRUCTIONS
from model.isa.core.globals import INSTRUCTIONS as GLOBAL_INSTRUCTIONS
from model.isa.core.integer import INSTRUCTIONS as INTEGER_INSTRUCTIONS
from model.isa.core.ref import INSTRUCTIONS as REF_INSTRUCTIONS
from model.isa.core.stack import INSTRUCTIONS as STACK_INSTRUCTIONS
from model.isa.core.value import INSTRUCTIONS as VALUE_INSTRUCTIONS
from model.isa.selectors import (
    SELECTOR_TABLES,
    SELECTOR_VALUES,
)
from model.isa.validation import (
    ABI_SLOT,
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    CONSTANT_POOL_ORDINAL,
    CONSTRAINT_MEMBER,
    FIELDS_DISTINCT,
    FUNCTION_ADDRESS,
    FUNCTION_LOCAL_ORDINAL,
    GLOBAL_ORDINAL,
    IMPORT_ORDINAL_OPTIONAL,
    INTEGER_BITSTREAM_SHAPE,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    LOCAL_BYTES_REPEATED_BASE,
    LOCAL_BYTES_REPEATED_COUNT,
    PACKED_SELECTOR_ALLOWED_PAIRS,
    PACKED_SELECTOR_TARGET_SUPPORTED,
    PACKED_SELECTORS,
    REF_SLOT,
    REGISTER_FUNCTION,
    REGISTER_REF,
    REGISTER_VALUE,
    RODATA_OFFSET,
    RODATA_ORDINAL,
    RODATA_STATIC_OFFSET,
    SELECTOR,
    VALUE_REGISTER_RANGE,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
)
from model.schema import ANY_BITS, SCALAR_ENCODINGS, EntityReference, FieldReference

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmImmediateFlag,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorCarrier,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EnumDomain,
    EnumValue,
    Immediate,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

_VALUE_REGISTER_CLASS = "vm.value"
_REF_REGISTER_CLASS = "vm.ref"
_FUNCTION_REGISTER_CLASS = "vm.function"
_VALUE_REGISTER_ALTERNATIVES = (RegClassAlt(_VALUE_REGISTER_CLASS),)
_REF_REGISTER_ALTERNATIVES = (RegClassAlt(_REF_REGISTER_CLASS),)
_FUNCTION_REGISTER_ALTERNATIVES = (RegClassAlt(_FUNCTION_REGISTER_CLASS),)
_EXECUTE_RESOURCE = "vm.execute"
_CONSTANT_SCHEDULE_CLASS = "vm.constant"
_EXECUTE_SCHEDULE_CLASS = "vm.execute"

_CORE_INSTRUCTIONS = (
    *ABI_INSTRUCTIONS,
    *BUFFER_INSTRUCTIONS,
    *CONSTANT_INSTRUCTIONS,
    *CONTROL_INSTRUCTIONS,
    *CONVERSION_INSTRUCTIONS,
    *FLOAT_INSTRUCTIONS,
    *FUNCTION_INSTRUCTIONS,
    *GLOBAL_INSTRUCTIONS,
    *INTEGER_INSTRUCTIONS,
    *REF_INSTRUCTIONS,
    *STACK_INSTRUCTIONS,
    *VALUE_INSTRUCTIONS,
)
VM_CORE_INSTRUCTIONS = tuple(sorted(_CORE_INSTRUCTIONS, key=lambda value: value.opcode))
_INSTRUCTIONS_BY_MNEMONIC = {
    instruction.mnemonic: instruction for instruction in _CORE_INSTRUCTIONS
}
_ENCODINGS_BY_ID = {encoding.entity_id: encoding for encoding in SCALAR_ENCODINGS}
_SELECTOR_TABLES_BY_ID = {table.entity_id: table for table in SELECTOR_TABLES}
_SELECTOR_VALUES_BY_TABLE_ID = {
    table_id: tuple(value for value in SELECTOR_VALUES if value.table_id == table_id)
    for table_id in _SELECTOR_TABLES_BY_ID
}
_ENCODING_SUFFIX = re.compile(r"_(?:(?:[irufv]\d+)(?:le)?|le)$")

_REGISTER_ALTERNATIVES_BY_RULE_ID = {
    REGISTER_VALUE.entity_id: _VALUE_REGISTER_ALTERNATIVES,
    REGISTER_REF.entity_id: _REF_REGISTER_ALTERNATIVES,
    REGISTER_FUNCTION.entity_id: _FUNCTION_REGISTER_ALTERNATIVES,
}
_ORDINAL_RULE_IDS = frozenset(
    rule.entity_id
    for rule in (
        ABI_SLOT,
        CONSTANT_POOL_ORDINAL,
        FUNCTION_LOCAL_ORDINAL,
        GLOBAL_ORDINAL,
        IMPORT_ORDINAL_OPTIONAL,
        REF_SLOT,
        RODATA_ORDINAL,
    )
)
_UNSIGNED_IMMEDIATE_RULE_IDS = frozenset(
    rule.entity_id
    for rule in (
        CONSTRAINT_MEMBER,
        LOCAL_BYTES_FIXED_BASE,
        LOCAL_BYTES_RANGE_BASE,
        LOCAL_BYTES_RANGE_LENGTH,
        LOCAL_BYTES_RANGE_MEMORY_FORMAT,
        LOCAL_BYTES_REPEATED_BASE,
        LOCAL_BYTES_REPEATED_COUNT,
        RODATA_OFFSET,
        RODATA_STATIC_OFFSET,
    )
)
_PLAIN_IMMEDIATE_RULE_IDS = frozenset((ANY_BITS.entity_id, PACKED_SELECTORS.entity_id))
_TARGET_VERIFY_CONSTRAINT_RULE_IDS = frozenset(
    rule.entity_id
    for rule in (
        INTEGER_BITSTREAM_SHAPE,
        PACKED_SELECTOR_ALLOWED_PAIRS,
        PACKED_SELECTOR_TARGET_SUPPORTED,
        VALUE_REGISTER_RANGE,
        VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
    )
)
_MODULE_VERIFY_CONSTRAINT_RULE_IDS = frozenset((FUNCTION_ADDRESS.entity_id,))
_NONMUTATING_REF_OPERAND_OWNERSHIP = frozenset(
    (
        RefOwnership.BORROW,
        RefOwnership.DIAGNOSTIC_BORROW,
        RefOwnership.INSPECT,
    )
)
_MEMORY_SPACE_BY_STATE_RESOURCE = {
    StateResource.INVOCATION_ARGUMENTS: MemorySpace.STACK,
    StateResource.INVOCATION_RESULTS: MemorySpace.STACK,
    StateResource.FRAME_LOCALS: MemorySpace.STACK,
    StateResource.PROCESS_GLOBALS: MemorySpace.GLOBAL,
    StateResource.BUFFER: MemorySpace.GENERIC,
}


@dataclasses.dataclass(frozen=True, slots=True)
class VmInstructionProjection:
    """One fixed-size packet instruction visible to target-Low."""

    instruction: Instruction

    def __post_init__(self) -> None:
        if self.instruction.control_flow not in (
            ControlFlow.SEQUENTIAL,
            ControlFlow.FAIL,
        ):
            raise ValueError(
                f"{self.instruction.mnemonic}: ordinary Low packet projection "
                "requires sequential or failing control flow"
            )
        if self.instruction.suspension is not Suspension.NEVER:
            raise ValueError(
                f"{self.instruction.mnemonic}: sequential instruction cannot suspend"
            )

    @property
    def key(self) -> str:
        return f"vm.{self.instruction.mnemonic}"

    @property
    def mnemonic(self) -> str:
        return self.instruction.mnemonic


class VmModuleResourceKind(enum.Enum):
    """Physical VM module-resource kind projected into C."""

    VALUE_IMMUTABLE = "VALUE_IMMUTABLE"
    VALUE_MUTABLE = "VALUE_MUTABLE"
    REF_IMMUTABLE = "REF_IMMUTABLE"
    REF_MUTABLE = "REF_MUTABLE"
    FUNCTION_IMMUTABLE = "FUNCTION_IMMUTABLE"
    FUNCTION_MUTABLE = "FUNCTION_MUTABLE"
    RODATA = "RODATA"


@dataclasses.dataclass(frozen=True, slots=True)
class VmModuleResource:
    """Physical resource kind and the ISA instructions that access it."""

    kind: VmModuleResourceKind
    load_descriptor_key: str
    store_descriptor_key: str | None
    store_preserve_descriptor_key: str | None


def _instruction(mnemonic: str) -> Instruction:
    return _INSTRUCTIONS_BY_MNEMONIC[mnemonic]


VM_INSTRUCTION_PROJECTIONS = tuple(
    VmInstructionProjection(instruction)
    for instruction in sorted(_CORE_INSTRUCTIONS, key=lambda value: value.opcode)
    if instruction.control_flow in (ControlFlow.SEQUENTIAL, ControlFlow.FAIL)
)

# Control-flow records are emitted from Low structural operations because they
# own successor edges rather than ordinary descriptor operands. Fixed-size fail
# packets have no successor and remain ordinary descriptors. Together the two
# projections partition the complete Core ISA.
VM_STRUCTURAL_INSTRUCTIONS = tuple(
    instruction
    for instruction in VM_CORE_INSTRUCTIONS
    if instruction.control_flow not in (ControlFlow.SEQUENTIAL, ControlFlow.FAIL)
)
VM_ABI_INSTRUCTIONS = tuple(
    instruction
    for instruction in VM_CORE_INSTRUCTIONS
    if instruction.family_id == "core.family.abi"
)


def _validate_core_instruction_partition() -> None:
    descriptor_instructions = tuple(
        projection.instruction for projection in VM_INSTRUCTION_PROJECTIONS
    )
    partitions = (
        descriptor_instructions,
        VM_STRUCTURAL_INSTRUCTIONS,
    )
    partitioned_ids = [
        instruction.entity_id for partition in partitions for instruction in partition
    ]
    core_ids = [instruction.entity_id for instruction in VM_CORE_INSTRUCTIONS]
    if len(partitioned_ids) != len(set(partitioned_ids)):
        raise ValueError("VM Core instruction projection partitions overlap")
    if set(partitioned_ids) != set(core_ids):
        missing = sorted(set(core_ids) - set(partitioned_ids))
        extra = sorted(set(partitioned_ids) - set(core_ids))
        raise ValueError(
            "VM Core instruction projection is incomplete: "
            f"missing={missing}, extra={extra}"
        )


_validate_core_instruction_partition()


def _field_name(name: str) -> str:
    return _ENCODING_SUFFIX.sub("", name)


def _rule_use(field: InstructionField, rule_id: str):
    return next(
        (rule_use for rule_use in field.validation if rule_use.rule_id == rule_id),
        None,
    )


def _register_alternatives(field: InstructionField) -> tuple[RegClassAlt, ...]:
    alternatives = tuple(
        _REGISTER_ALTERNATIVES_BY_RULE_ID[rule_use.rule_id]
        for rule_use in field.validation
        if rule_use.rule_id in _REGISTER_ALTERNATIVES_BY_RULE_ID
    )
    if len(alternatives) != 1:
        raise ValueError(
            f"{field.name}: register field must select exactly one VM bank"
        )
    return alternatives[0]


def _allowed_values_domain_name(values: tuple[int, ...]) -> str:
    return "vm.allowed." + "_".join(str(value) for value in values)


def _selector_table_id(field: InstructionField) -> str | None:
    rule_use = _rule_use(field, SELECTOR.entity_id)
    if rule_use is None:
        return None
    if len(rule_use.arguments) != 1 or not isinstance(
        rule_use.arguments[0], EntityReference
    ):
        raise ValueError(f"{field.name}: malformed selector validation")
    return rule_use.arguments[0].entity_id


def _immediate(
    field: InstructionField,
    element_index: int = 0,
    *,
    record_ordinal: bool = False,
) -> Immediate:
    encoding = _ENCODINGS_BY_ID[field.encoding_id]
    bit_width = encoding.byte_length * 8
    field_name = _field_name(field.name)
    if field.array_length != 1:
        field_name = f"{field_name}_{element_index}"
    encoding_field_id = field.offset + element_index * encoding.byte_length

    selector_table_id = _selector_table_id(field)
    allowed_values_rule = _rule_use(field, ALLOWED_VALUES.entity_id)
    allowed_range_rule = _rule_use(field, ALLOWED_RANGE.entity_id)
    recognized_rule_ids = {
        *(rule_id for rule_id in _ORDINAL_RULE_IDS),
        *(rule_id for rule_id in _UNSIGNED_IMMEDIATE_RULE_IDS),
        *(rule_id for rule_id in _PLAIN_IMMEDIATE_RULE_IDS),
        ALLOWED_RANGE.entity_id,
        ALLOWED_VALUES.entity_id,
        SELECTOR.entity_id,
    }
    unknown_rule_ids = tuple(
        rule_use.rule_id
        for rule_use in field.validation
        if rule_use.rule_id not in recognized_rule_ids
    )
    if unknown_rule_ids:
        raise ValueError(
            f"{field.name}: unsupported immediate validation "
            + ", ".join(unknown_rule_ids)
        )

    enum_domain = None
    if selector_table_id is not None:
        kind = ImmediateKind.ENUM
        enum_domain = selector_table_id
        signed_min = 0
        unsigned_max = (2**bit_width) - 1
    elif allowed_values_rule is not None:
        values = tuple(int(value) for value in allowed_values_rule.arguments[0])
        kind = ImmediateKind.ENUM
        enum_domain = _allowed_values_domain_name(values)
        signed_min = min(values)
        unsigned_max = max(values)
    elif record_ordinal or any(
        rule_use.rule_id in _ORDINAL_RULE_IDS for rule_use in field.validation
    ):
        kind = ImmediateKind.ORDINAL
        signed_min = 0
        unsigned_max = (2**bit_width) - 1
    else:
        is_signed = encoding.c_type.startswith("int")
        minimum = -(2 ** (bit_width - 1)) if is_signed else 0
        maximum = (2 ** (bit_width - 1)) - 1 if is_signed else (2**bit_width) - 1
        if allowed_range_rule is not None:
            minimum, maximum = (
                int(allowed_range_rule.arguments[0]),
                int(allowed_range_rule.arguments[1]),
            )
        kind = ImmediateKind.SIGNED if is_signed else ImmediateKind.UNSIGNED
        signed_min = minimum if is_signed else 0
        unsigned_max = maximum

    return Immediate(
        field_name=field_name,
        kind=kind,
        bit_width=bit_width,
        encoding_field_id=encoding_field_id,
        enum_domain=enum_domain,
        signed_min=signed_min,
        unsigned_max=unsigned_max,
    )


def _variable_value_register_limits(instruction: Instruction) -> dict[str, int]:
    limits: dict[str, int] = {}
    has_integer_bitstream_shape = any(
        constraint.rule_id == INTEGER_BITSTREAM_SHAPE.entity_id
        for constraint in instruction.constraints
    )
    for constraint in instruction.constraints:
        if constraint.rule_id not in (
            VALUE_REGISTER_RANGE.entity_id,
            VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id,
        ):
            continue
        if not constraint.arguments or not isinstance(
            constraint.arguments[0], FieldReference
        ):
            raise ValueError(f"{instruction.mnemonic}: malformed value-register range")
        field_name = constraint.arguments[0].field_name
        if constraint.rule_id == VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id:
            if len(constraint.arguments) != 3:
                raise ValueError(
                    f"{instruction.mnemonic}: malformed memory-format "
                    "value-register range"
                )
            maximum_unit_count = int(constraint.arguments[2])
        elif has_integer_bitstream_shape:
            bitstream_constraint = next(
                constraint
                for constraint in instruction.constraints
                if constraint.rule_id == INTEGER_BITSTREAM_SHAPE.entity_id
            )
            if len(bitstream_constraint.arguments) != 7:
                raise ValueError(
                    f"{instruction.mnemonic}: malformed integer bitstream shape"
                )
            # The narrowest logical field is one bit, so neither register run
            # can contain more cells than the complete stream has bits.
            maximum_unit_count = int(bitstream_constraint.arguments[1])
        else:
            raise ValueError(
                f"{instruction.mnemonic}: value-register range has no "
                "structured maximum-unit constraint"
            )
        previous_limit = limits.setdefault(field_name, maximum_unit_count)
        if previous_limit != maximum_unit_count:
            raise ValueError(
                f"{instruction.mnemonic}: value-register range {field_name} "
                "has conflicting maximum-unit constraints"
            )
    return limits


def _state_effect(effect) -> Effect:
    if effect.access in (StateAccess.READ, StateAccess.WRITE):
        memory_space = _MEMORY_SPACE_BY_STATE_RESOURCE.get(effect.resource)
        if memory_space is None:
            raise ValueError(f"unsupported Core state resource {effect.resource.value}")
        return Effect(
            EffectKind.READ if effect.access is StateAccess.READ else EffectKind.WRITE,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
        )
    if effect.access in (StateAccess.UNKNOWN, StateAccess.ALLOCATE):
        return Effect(
            EffectKind.CALL,
            flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
        )
    if effect.access in (StateAccess.RELEASE, StateAccess.SYNCHRONIZE):
        return Effect(EffectKind.BARRIER, flags=(EffectFlag.ORDERED,))
    raise ValueError(f"unsupported Core state access {effect.access.value}")


def _instruction_effects(instruction: Instruction) -> tuple[Effect, ...]:
    effects: list[Effect] = []
    if instruction.control_flow is not ControlFlow.SEQUENTIAL:
        effects.append(Effect(EffectKind.CONTROL, flags=(EffectFlag.ORDERED,)))
    for state_effect in instruction.state_effects:
        effect = _state_effect(state_effect)
        if effect not in effects:
            effects.append(effect)

    mutates_ref_operand = any(
        field.role is InstructionFieldRole.OPERAND
        and field.runtime_ref_policy is not None
        and field.runtime_ref_policy.ownership not in _NONMUTATING_REF_OPERAND_OWNERSHIP
        for field in instruction.fields
    )
    if mutates_ref_operand:
        barrier = Effect(EffectKind.BARRIER, flags=(EffectFlag.ORDERED,))
        if barrier not in effects:
            effects.append(barrier)
    return tuple(effects)


def _descriptor_constraints(instruction: Instruction) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    operand_indices_by_field: dict[str, int] = {}
    operand_index = 0
    for field in instruction.fields:
        if field.role in (
            InstructionFieldRole.RESULT,
            InstructionFieldRole.OPERAND,
        ):
            operand_indices_by_field[field.name] = operand_index
            operand_index += 1
    for constraint in instruction.constraints:
        if constraint.rule_id in (
            *_TARGET_VERIFY_CONSTRAINT_RULE_IDS,
            *_MODULE_VERIFY_CONSTRAINT_RULE_IDS,
        ):
            continue
        if constraint.rule_id != FIELDS_DISTINCT.entity_id:
            raise ValueError(
                f"{instruction.mnemonic}: unsupported record constraint "
                f"{constraint.rule_id}"
            )
        lhs, rhs = constraint.arguments
        if not isinstance(lhs, FieldReference) or not isinstance(rhs, FieldReference):
            raise ValueError(f"{instruction.mnemonic}: malformed distinct fields")
        lhs_index = operand_indices_by_field[lhs.field_name]
        rhs_index = operand_indices_by_field[rhs.field_name]
        constraints.append(Constraint(ConstraintKind.EARLY_CLOBBER, lhs_index))
        if rhs_index == lhs_index:
            raise ValueError(f"{instruction.mnemonic}: self-distinct field")
    return tuple(constraints)


def _record_ordinal_field_names(instruction: Instruction) -> frozenset[str]:
    field_names: set[str] = set()
    for constraint in instruction.constraints:
        if constraint.rule_id != FUNCTION_ADDRESS.entity_id:
            continue
        if len(constraint.arguments) != 3:
            raise ValueError(f"{instruction.mnemonic}: malformed function address")
        for argument in constraint.arguments[1:]:
            if not isinstance(argument, FieldReference):
                raise ValueError(
                    f"{instruction.mnemonic}: malformed function address ordinal"
                )
            field_names.add(argument.field_name)
    return frozenset(field_names)


def _instruction_classes(instruction: Instruction) -> tuple[InstructionClass, ...]:
    classes: list[InstructionClass] = []
    resources = {effect.resource for effect in instruction.state_effects}
    if StateResource.BUFFER in resources:
        classes.append(InstructionClass.GENERIC_MEMORY)
    if StateResource.FRAME_LOCALS in resources:
        classes.append(InstructionClass.PRIVATE_MEMORY)
    if StateResource.PROCESS_GLOBALS in resources:
        classes.append(InstructionClass.GLOBAL_MEMORY)
    if ".atomic." in instruction.mnemonic:
        classes.append(InstructionClass.ATOMIC)
    if instruction.family_id == "core.family.conversion":
        classes.append(InstructionClass.CONVERSION)
    elif instruction.family_id == "core.family.control":
        classes.append(InstructionClass.CONTROL)
    elif instruction.family_id in (
        "core.family.function",
        "core.family.ref",
        "core.family.value",
    ):
        classes.append(InstructionClass.REGISTER_MOVE)
    elif instruction.family_id == "core.family.constant":
        classes.append(InstructionClass.OTHER)
    elif not classes:
        classes.append(InstructionClass.SCALAR_ALU)
    return tuple(classes)


def _enum_domains() -> tuple[EnumDomain, ...]:
    selector_table_ids = {
        table_id
        for projection in VM_INSTRUCTION_PROJECTIONS
        for field in projection.instruction.fields
        if (table_id := _selector_table_id(field)) is not None
    }
    domains = [
        EnumDomain(
            name=table_id,
            values=tuple(
                EnumValue(value.name, value.value)
                for value in _SELECTOR_VALUES_BY_TABLE_ID[table_id]
            ),
        )
        for table_id in sorted(selector_table_ids)
    ]
    allowed_value_sets = {
        tuple(int(value) for value in rule_use.arguments[0])
        for projection in VM_INSTRUCTION_PROJECTIONS
        for field in projection.instruction.fields
        if (rule_use := _rule_use(field, ALLOWED_VALUES.entity_id)) is not None
    }
    domains.extend(
        EnumDomain(
            name=_allowed_values_domain_name(values),
            values=tuple(EnumValue(str(value), value) for value in values),
        )
        for values in sorted(allowed_value_sets)
    )
    return tuple(domains)


VM_ENUM_DOMAINS = _enum_domains()


def _descriptor(projection: VmInstructionProjection) -> Descriptor:
    instruction = projection.instruction
    variable_register_limits = _variable_value_register_limits(instruction)
    record_ordinal_field_names = _record_ordinal_field_names(instruction)
    operands: list[Operand] = []
    immediates: list[Immediate] = []
    asm_results: list[str] = []
    asm_operands: list[str] = []
    asm_immediates: list[AsmImmediate] = []
    for field in instruction.fields:
        name = _field_name(field.name)
        if field.role in (
            InstructionFieldRole.RESULT,
            InstructionFieldRole.OPERAND,
        ):
            variable_register_limit = variable_register_limits.get(field.name)
            is_variable_register = variable_register_limit is not None
            operands.append(
                Operand(
                    name,
                    OperandRole.RESULT
                    if field.role is InstructionFieldRole.RESULT
                    else OperandRole.OPERAND,
                    _register_alternatives(field),
                    flags=(
                        (OperandFlag.VARIABLE_UNIT_COUNT,)
                        if is_variable_register
                        else ()
                    ),
                    unit_count=variable_register_limit or 1,
                    encoding_field_id=field.offset,
                )
            )
            if field.role is InstructionFieldRole.RESULT:
                asm_results.append(name)
            else:
                asm_operands.append(name)
        elif field.role in (
            InstructionFieldRole.IMMEDIATE,
            InstructionFieldRole.CONSTRAINT_MEMBER,
        ):
            for element_index in range(field.array_length):
                immediate = _immediate(
                    field,
                    element_index,
                    record_ordinal=field.name in record_ordinal_field_names,
                )
                immediates.append(immediate)
                asm_immediates.append(
                    AsmImmediate(
                        immediate.field_name,
                        flags=(AsmImmediateFlag.ENUM_TOKEN,)
                        if immediate.kind is ImmediateKind.ENUM
                        else (),
                    )
                )
        elif field.role is not InstructionFieldRole.PADDING:
            raise ValueError(
                f"{instruction.mnemonic}: field {field.name} has unsupported "
                f"projection role {field.role.value}"
            )

    effects = _instruction_effects(instruction)
    side_effecting = any(
        effect.kind in (EffectKind.WRITE, EffectKind.CALL, EffectKind.BARRIER)
        for effect in effects
    )
    has_results = any(operand.role is OperandRole.RESULT for operand in operands)
    is_terminator = instruction.control_flow is not ControlFlow.SEQUENTIAL
    if is_terminator:
        flags = (
            DescriptorFlag.SIDE_EFFECTING,
            DescriptorFlag.TERMINATOR,
            DescriptorFlag.NO_RETURN,
        )
    elif side_effecting:
        flags = (DescriptorFlag.SIDE_EFFECTING,)
    elif has_results:
        flags = (DescriptorFlag.DEAD_REMOVABLE,)
    else:
        flags = ()
    is_constant = instruction.family_id == "core.family.constant"
    asm_forms = (
        AsmForm(
            results=tuple(asm_results),
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
    )
    return Descriptor(
        key=projection.key,
        mnemonic=projection.mnemonic,
        semantic_tag=projection.mnemonic,
        operands=tuple(operands),
        schedule_class=(
            _CONSTANT_SCHEDULE_CLASS if is_constant else _EXECUTE_SCHEDULE_CLASS
        ),
        op_kind=DescriptorOpKind.CONST if is_constant else DescriptorOpKind.OP,
        immediates=tuple(immediates),
        asm_forms=asm_forms,
        effects=effects,
        constraints=_descriptor_constraints(instruction),
        encoding_id=instruction.opcode,
        flags=flags,
        instruction_classes=_instruction_classes(instruction),
    )


def _switch_descriptor(instruction: Instruction) -> Descriptor:
    """Projects control.switch onto its structural Low carrier."""

    if instruction.control_flow is not ControlFlow.SWITCH:
        raise ValueError(f"{instruction.mnemonic}: expected switch control flow")
    selector_fields = tuple(
        field
        for field in instruction.fields
        if field.role is InstructionFieldRole.OPERAND
    )
    if len(selector_fields) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: structural switch requires one selector"
        )
    selector_field = selector_fields[0]
    selector_name = _field_name(selector_field.name)
    return Descriptor(
        key=f"vm.{instruction.mnemonic}",
        mnemonic=instruction.mnemonic,
        semantic_tag=instruction.mnemonic,
        operands=(
            Operand(
                selector_name,
                OperandRole.OPERAND,
                _register_alternatives(selector_field),
                unit_count=1,
                encoding_field_id=selector_field.offset,
            ),
        ),
        schedule_class=_EXECUTE_SCHEDULE_CLASS,
        carrier=DescriptorCarrier.SWITCH,
        asm_forms=(AsmForm(operands=(selector_name,)),),
        effects=_instruction_effects(instruction),
        encoding_id=instruction.opcode,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        instruction_classes=(InstructionClass.CONTROL,),
    )


def _yield_descriptor(instruction: Instruction) -> Descriptor:
    """Projects an explicit suspension onto its structural Low branch."""

    if instruction.control_flow is not ControlFlow.YIELD:
        raise ValueError(f"{instruction.mnemonic}: expected yield control flow")
    if instruction.suspension is not Suspension.ALWAYS:
        raise ValueError(f"{instruction.mnemonic}: explicit yield must always suspend")
    if instruction.byte_length != 8:
        raise ValueError(f"{instruction.mnemonic}: structural yield must be 8 bytes")
    target_fields = tuple(
        field
        for field in instruction.fields
        if field.role is InstructionFieldRole.IMMEDIATE
    )
    if len(target_fields) != 1 or target_fields[0].name != "target_rel32":
        raise ValueError(
            f"{instruction.mnemonic}: structural yield requires one s32 target"
        )
    return Descriptor(
        key=f"vm.{instruction.mnemonic}",
        mnemonic=instruction.mnemonic,
        semantic_tag=instruction.mnemonic,
        operands=(),
        schedule_class=_EXECUTE_SCHEDULE_CLASS,
        carrier=DescriptorCarrier.BRANCH,
        asm_forms=(AsmForm(),),
        effects=_instruction_effects(instruction),
        encoding_id=instruction.opcode,
        flags=(
            DescriptorFlag.SIDE_EFFECTING,
            DescriptorFlag.TERMINATOR,
            DescriptorFlag.MAY_YIELD,
        ),
        instruction_classes=(InstructionClass.CONTROL,),
    )


_CONTROL_YIELD_INSTRUCTION = _instruction("control.yield.s32")
_CONTROL_SWITCH_INSTRUCTION = _instruction("control.switch")

VM_PACKET_DESCRIPTORS = tuple(
    _descriptor(projection) for projection in VM_INSTRUCTION_PROJECTIONS
)


VM_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="vm.core",
    target_key="vm",
    feature_key="vm.core.0",
    c_header_path=Path("loom/src/loom/target/arch/vm/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/vm/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_VM_DESCRIPTORS_H_",
    public_header="loom/target/arch/vm/descriptors.h",
    function_name="loom_vm_core_descriptor_set",
    c_table_prefix="VmCore",
    c_enum_prefix="VM_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _VALUE_REGISTER_CLASS,
            alloc_unit_bits=64,
            spill_slot_space=SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL,),
            target_bank_id=1,
            allocatable_count=256,
            alias_set_id=1,
        ),
        RegClass(
            _REF_REGISTER_CLASS,
            alloc_unit_bits=128,
            spill_slot_space=SpillSlotSpace.PRIVATE,
            flags=(
                RegClassFlag.PHYSICAL,
                RegClassFlag.REFERENCE,
            ),
            target_bank_id=2,
            allocatable_count=256,
            alias_set_id=2,
        ),
        RegClass(
            _FUNCTION_REGISTER_CLASS,
            alloc_unit_bits=128,
            spill_slot_space=SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL,),
            target_bank_id=3,
            allocatable_count=256,
            alias_set_id=3,
        ),
    ),
    resources=(
        Resource(
            _EXECUTE_RESOURCE,
            capacity_per_cycle=1,
            kind=ResourceKind.SCALAR_ALU,
        ),
    ),
    schedule_classes=(
        ScheduleClass(
            _CONSTANT_SCHEDULE_CLASS,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
            instruction_classes=(InstructionClass.OTHER,),
        ),
        ScheduleClass(
            _EXECUTE_SCHEDULE_CLASS,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_EXECUTE_RESOURCE, cycles=1, units=1),),
            flags=(
                ScheduleClassFlag.MAY_LOAD,
                ScheduleClassFlag.MAY_STORE,
                ScheduleClassFlag.MAY_CALL,
                ScheduleClassFlag.CONTROL,
            ),
        ),
    ),
    enum_domains=VM_ENUM_DOMAINS,
    descriptors=(
        *VM_PACKET_DESCRIPTORS,
        _yield_descriptor(_CONTROL_YIELD_INSTRUCTION),
        _switch_descriptor(_CONTROL_SWITCH_INSTRUCTION),
    ),
)

_DESCRIPTORS_BY_KEY = {
    descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
}


def _require_module_resource_descriptor(
    descriptor_key: str,
    *,
    operand_count: int,
    result_count: int,
    ordinal_immediate: bool,
) -> None:
    descriptor = _DESCRIPTORS_BY_KEY.get(descriptor_key)
    if descriptor is None:
        raise ValueError(
            f"VM module resource names unknown descriptor {descriptor_key!r}"
        )
    actual_operand_count = sum(
        operand.role is OperandRole.OPERAND for operand in descriptor.operands
    )
    actual_result_count = sum(
        operand.role is OperandRole.RESULT for operand in descriptor.operands
    )
    if (
        actual_operand_count != operand_count
        or actual_result_count != result_count
        or len(descriptor.immediates) != (1 if ordinal_immediate else 0)
        or (
            ordinal_immediate
            and descriptor.immediates[0].kind is not ImmediateKind.ORDINAL
        )
    ):
        immediate_contract = "one ordinal" if ordinal_immediate else "no immediates"
        raise ValueError(
            f"{descriptor_key}: VM module resource descriptor must have "
            f"{operand_count} operands, {result_count} results, and "
            f"{immediate_contract}"
        )


def _module_resource(
    kind: VmModuleResourceKind,
    load_descriptor_key: str,
    store_descriptor_key: str | None,
    store_preserve_descriptor_key: str | None = None,
) -> VmModuleResource:
    _require_module_resource_descriptor(
        load_descriptor_key,
        operand_count=0,
        result_count=1,
        ordinal_immediate=True,
    )
    if store_descriptor_key is not None:
        _require_module_resource_descriptor(
            store_descriptor_key,
            operand_count=1,
            result_count=0,
            ordinal_immediate=True,
        )
    if store_preserve_descriptor_key is not None:
        if store_descriptor_key is None:
            raise ValueError(
                f"{kind.value}: store preservation requires a store descriptor"
            )
        _require_module_resource_descriptor(
            store_preserve_descriptor_key,
            operand_count=1,
            result_count=1,
            ordinal_immediate=False,
        )
    return VmModuleResource(
        kind,
        load_descriptor_key,
        store_descriptor_key,
        store_preserve_descriptor_key,
    )


VM_MODULE_RESOURCES = (
    _module_resource(
        VmModuleResourceKind.VALUE_IMMUTABLE,
        "vm.global.value.immutable.load",
        "vm.global.value.immutable.store",
    ),
    _module_resource(
        VmModuleResourceKind.VALUE_MUTABLE,
        "vm.global.value.mutable.load",
        "vm.global.value.mutable.store",
    ),
    _module_resource(
        VmModuleResourceKind.REF_IMMUTABLE,
        "vm.global.ref.immutable.load.borrow",
        "vm.global.ref.immutable.store.move",
        "vm.ref.retain",
    ),
    _module_resource(
        VmModuleResourceKind.REF_MUTABLE,
        "vm.global.ref.mutable.load.retain",
        "vm.global.ref.mutable.store.move",
        "vm.ref.retain",
    ),
    _module_resource(
        VmModuleResourceKind.FUNCTION_IMMUTABLE,
        "vm.global.func.immutable.load",
        "vm.global.func.immutable.store",
    ),
    _module_resource(
        VmModuleResourceKind.FUNCTION_MUTABLE,
        "vm.global.func.mutable.load",
        "vm.global.func.mutable.store",
    ),
    _module_resource(
        VmModuleResourceKind.RODATA,
        "vm.buffer.rodata.load",
        None,
    ),
)
