# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM ISA projection used by Loom source-to-Low lowering."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from model.isa import ControlFlow, Instruction, InstructionFieldRole, Suspension
from model.isa.core.constant import INSTRUCTIONS as CONSTANT_INSTRUCTIONS
from model.isa.core.conversion import INSTRUCTIONS as CONVERSION_INSTRUCTIONS
from model.isa.core.integer import INSTRUCTIONS as INTEGER_INSTRUCTIONS
from model.isa.selectors import SELECTOR_TABLES_BY_NAME, SELECTOR_VALUES
from model.schema import SCALAR_ENCODINGS

from loom.dialect.index import defs as index_defs
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dsl import Op
from loom.ir import ScalarType
from loom.scalar_type import ScalarTypeKind
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    EncodingFieldValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)
from loom.verify import type_satisfies_constraint

_VALUE_REGISTER_CLASS = "vm.value"
_VALUE_REGISTER_ALTERNATIVES = (RegClassAlt(_VALUE_REGISTER_CLASS),)
_EXECUTE_RESOURCE = "vm.execute"
_CONSTANT_SCHEDULE_CLASS = "vm.constant"
_EXECUTE_SCHEDULE_CLASS = "vm.execute"

_INSTRUCTIONS_BY_MNEMONIC = {
    instruction.mnemonic: instruction
    for instruction in (
        *CONSTANT_INSTRUCTIONS,
        *CONVERSION_INSTRUCTIONS,
        *INTEGER_INSTRUCTIONS,
    )
}
_ENCODINGS_BY_ID = {encoding.entity_id: encoding for encoding in SCALAR_ENCODINGS}
_SELECTOR_VALUES_BY_KEY = {
    (value.table_id, value.name): value.value for value in SELECTOR_VALUES
}
_ENCODING_SUFFIX = re.compile(r"_(?:[iuf]\d+(?:le)?|v8)$")


@dataclasses.dataclass(frozen=True, slots=True)
class VmInstructionProjection:
    """One physical instruction form visible to target-Low."""

    instruction: Instruction
    selector_table: str | None = None
    selector_name: str | None = None

    def __post_init__(self) -> None:
        has_selector = self.selector_table is not None or self.selector_name is not None
        if has_selector and (self.selector_table is None or self.selector_name is None):
            raise ValueError("VM selector table and value must be specified together")
        selector_fields = tuple(
            field for field in self.instruction.fields if field.name == "selector_u8"
        )
        if bool(selector_fields) != has_selector:
            raise ValueError(
                f"{self.instruction.mnemonic}: selector projection does not match "
                "fields"
            )
        if self.instruction.control_flow is not ControlFlow.SEQUENTIAL:
            raise ValueError(
                f"{self.instruction.mnemonic}: initial Low projection requires "
                "sequential control flow"
            )
        if self.instruction.suspension is not Suspension.NEVER:
            raise ValueError(
                f"{self.instruction.mnemonic}: initial Low projection cannot suspend"
            )
        if self.instruction.state_effects:
            raise ValueError(
                f"{self.instruction.mnemonic}: pure Low projection has state effects"
            )

    @property
    def key(self) -> str:
        suffix = (
            f".{self.selector_name.replace('.', '_')}"
            if self.selector_name is not None
            else ""
        )
        return f"vm.{self.instruction.mnemonic}{suffix}"

    @property
    def mnemonic(self) -> str:
        return self.key.removeprefix("vm.")

    @property
    def selector_value(self) -> int | None:
        if self.selector_table is None:
            return None
        table = SELECTOR_TABLES_BY_NAME[self.selector_table]
        return _SELECTOR_VALUES_BY_KEY[(table.entity_id, self.selector_name)]


@dataclasses.dataclass(frozen=True, slots=True)
class VmSourceLowering:
    """One generated concrete source signature selecting a VM descriptor."""

    source_op: Op
    operand_types: tuple[ScalarTypeKind, ...]
    result_types: tuple[ScalarTypeKind, ...]
    descriptor_key: str


def _instruction(mnemonic: str) -> Instruction:
    return _INSTRUCTIONS_BY_MNEMONIC[mnemonic]


VM_INSTRUCTION_PROJECTIONS = (
    VmInstructionProjection(_instruction("constant.zero")),
    VmInstructionProjection(_instruction("constant.s16")),
    VmInstructionProjection(_instruction("constant.i32")),
    VmInstructionProjection(_instruction("constant.i64")),
    VmInstructionProjection(_instruction("integer.add.i32")),
    VmInstructionProjection(_instruction("integer.add.i64")),
    VmInstructionProjection(_instruction("integer.mul.i32")),
    VmInstructionProjection(_instruction("integer.mul.i64")),
    VmInstructionProjection(
        _instruction("conversion.integer"),
        "integer.convert",
        "s32.to.i64",
    ),
    VmInstructionProjection(
        _instruction("conversion.integer"),
        "integer.convert",
        "u32.to.i64",
    ),
    VmInstructionProjection(
        _instruction("conversion.float.extend"),
        "float.extend",
        "bf16.to.f32",
    ),
    VmInstructionProjection(
        _instruction("conversion.float.to.integer"),
        "float.to.integer",
        "f32.to.u32",
    ),
)


def _field_name(name: str) -> str:
    return _ENCODING_SUFFIX.sub("", name)


def _immediate(field, *, default_value: int | None = None) -> Immediate:
    encoding = _ENCODINGS_BY_ID[field.encoding_id]
    bit_width = encoding.byte_length * 8
    is_signed = encoding.c_type.startswith("int")
    flags = (ImmediateFlag.DEFAULT_VALUE,) if default_value is not None else ()
    return Immediate(
        field_name=_field_name(field.name),
        kind=ImmediateKind.SIGNED if is_signed else ImmediateKind.UNSIGNED,
        flags=flags,
        bit_width=bit_width,
        encoding_field_id=field.offset,
        signed_min=-(2 ** (bit_width - 1)) if is_signed else 0,
        unsigned_max=(2 ** (bit_width - (1 if is_signed else 0))) - 1,
        default_value=default_value or 0,
    )


def _descriptor(projection: VmInstructionProjection) -> Descriptor:
    instruction = projection.instruction
    operands: list[Operand] = []
    immediates: list[Immediate] = []
    encoding_field_values: list[EncodingFieldValue] = []
    asm_results: list[str] = []
    asm_operands: list[str] = []
    asm_immediates: list[AsmImmediate] = []
    for field in instruction.fields:
        name = _field_name(field.name)
        if field.role is InstructionFieldRole.RESULT:
            operands.append(
                Operand(
                    name,
                    OperandRole.RESULT,
                    _VALUE_REGISTER_ALTERNATIVES,
                    encoding_field_id=field.offset,
                )
            )
            asm_results.append(name)
        elif field.role is InstructionFieldRole.OPERAND:
            operands.append(
                Operand(
                    name,
                    OperandRole.OPERAND,
                    _VALUE_REGISTER_ALTERNATIVES,
                    encoding_field_id=field.offset,
                )
            )
            asm_operands.append(name)
        elif field.role is InstructionFieldRole.IMMEDIATE:
            default_value = (
                projection.selector_value if field.name == "selector_u8" else None
            )
            immediate = _immediate(field, default_value=default_value)
            immediates.append(immediate)
            if default_value is None:
                asm_immediates.append(AsmImmediate(immediate.field_name))
        elif field.role is InstructionFieldRole.PADDING:
            encoding_field_values.append(EncodingFieldValue(field.offset, 0))
        else:
            raise ValueError(
                f"{instruction.mnemonic}: field {field.name} has unsupported "
                f"projection role {field.role.value}"
            )

    is_constant = instruction.mnemonic.startswith("constant.")
    is_conversion = instruction.mnemonic.startswith("conversion.")
    instruction_class = (
        InstructionClass.OTHER
        if is_constant
        else InstructionClass.CONVERSION
        if is_conversion
        else InstructionClass.SCALAR_ALU
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
        encoding_field_values=tuple(encoding_field_values),
        asm_forms=(
            AsmForm(
                results=tuple(asm_results),
                operands=tuple(asm_operands),
                immediates=tuple(asm_immediates),
            ),
        ),
        encoding_id=instruction.opcode,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        instruction_classes=(instruction_class,),
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
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            target_bank_id=1,
            allocatable_count=256,
            alias_set_id=1,
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
        ),
    ),
    descriptors=tuple(
        _descriptor(projection) for projection in VM_INSTRUCTION_PROJECTIONS
    ),
)

_DESCRIPTORS_BY_KEY = {
    descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
}


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
    descriptor_key: str,
    *,
    operand_count: int,
    result_count: int,
) -> None:
    descriptor = _DESCRIPTORS_BY_KEY.get(descriptor_key)
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
    if descriptor.op_kind is not DescriptorOpKind.OP:
        raise ValueError(
            f"{descriptor_key}: source operation projection requires a low.op "
            "descriptor"
        )
    missing_default_immediates = tuple(
        immediate.field_name
        for immediate in descriptor.immediates
        if ImmediateFlag.DEFAULT_VALUE not in immediate.flags
    )
    if missing_default_immediates:
        raise ValueError(
            f"{descriptor_key}: source operation projection cannot synthesize "
            "required immediates " + ", ".join(missing_default_immediates)
        )


def _require_concrete_source_types(
    source_op: Op,
    operand_types: tuple[ScalarTypeKind, ...],
    result_types: tuple[ScalarTypeKind, ...],
) -> None:
    fields = (*source_op.operands, *source_op.results)
    scalar_types = (*operand_types, *result_types)
    for field, scalar_type in zip(fields, scalar_types, strict=True):
        if not type_satisfies_constraint(
            ScalarType(scalar_type), field.type_constraint
        ):
            raise ValueError(
                f"{source_op.name}: {scalar_type.name} does not satisfy the "
                f"{field.name} type constraint"
            )


def _source_lowering(
    source_op: Op,
    operand_types: tuple[ScalarTypeKind, ...],
    result_types: tuple[ScalarTypeKind, ...],
    descriptor_key: str,
) -> VmSourceLowering:
    _require_fixed_source_shape(
        source_op,
        operand_count=len(operand_types),
        result_count=len(result_types),
    )
    _require_concrete_source_types(source_op, operand_types, result_types)
    _require_descriptor_shape(
        descriptor_key,
        operand_count=len(operand_types),
        result_count=len(result_types),
    )
    return VmSourceLowering(source_op, operand_types, result_types, descriptor_key)


def _same_type_binary(
    source_op: Op,
    descriptor_by_type: dict[ScalarTypeKind, str],
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
            (scalar_type, scalar_type),
            (scalar_type,),
            descriptor_key,
        )
        for scalar_type, descriptor_key in descriptor_by_type.items()
    )


def _cast(
    source_op: Op,
    descriptor_by_type_pair: dict[tuple[ScalarTypeKind, ScalarTypeKind], str],
) -> tuple[VmSourceLowering, ...]:
    """Projects a unary Loom cast onto concrete VM instructions."""

    _require_fixed_source_shape(source_op, operand_count=1, result_count=1)
    if not descriptor_by_type_pair:
        raise ValueError(f"{source_op.name}: cast projection has no type cases")
    return tuple(
        _source_lowering(
            source_op,
            (source_type,),
            (result_type,),
            descriptor_key,
        )
        for (
            source_type,
            result_type,
        ), descriptor_key in descriptor_by_type_pair.items()
    )


VM_SOURCE_LOWERINGS = (
    *_same_type_binary(
        index_defs.index_add,
        {
            ScalarTypeKind.INDEX: "vm.integer.add.i64",
            ScalarTypeKind.OFFSET: "vm.integer.add.i64",
        },
    ),
    *_same_type_binary(
        index_defs.index_mul,
        {ScalarTypeKind.INDEX: "vm.integer.mul.i64"},
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_addi,
        {
            ScalarTypeKind.I32: "vm.integer.add.i32",
            ScalarTypeKind.I64: "vm.integer.add.i64",
        },
    ),
    *_same_type_binary(
        scalar_arithmetic.scalar_muli,
        {
            ScalarTypeKind.I32: "vm.integer.mul.i32",
            ScalarTypeKind.I64: "vm.integer.mul.i64",
        },
    ),
    *_cast(
        scalar_conversion.scalar_extf,
        {
            (ScalarTypeKind.BF16, ScalarTypeKind.F32): (
                "vm.conversion.float.extend.bf16_to_f32"
            ),
        },
    ),
    *_cast(
        scalar_conversion.scalar_fptoui,
        {
            (ScalarTypeKind.F32, ScalarTypeKind.I32): (
                "vm.conversion.float.to.integer.f32_to_u32"
            ),
        },
    ),
    *_cast(
        index_defs.index_cast,
        {
            (ScalarTypeKind.I32, ScalarTypeKind.INDEX): (
                "vm.conversion.integer.s32_to_i64"
            ),
            (ScalarTypeKind.I32, ScalarTypeKind.OFFSET): (
                "vm.conversion.integer.u32_to_i64"
            ),
        },
    ),
)
