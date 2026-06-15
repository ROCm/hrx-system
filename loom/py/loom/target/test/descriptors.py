# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the backend-independent test-low target."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandForm,
    OperandFormMatch,
    OperandFormMatchKind,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    RegisterPart,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

_REG_I32 = "test.i32"
_REG_I8 = "test.i8"
_REG_F32 = "test.f32"
_REG_I64 = "test.i64"
_REG_PTR = "test.ptr"
_REG_PHYS = "test.phys"
_REG_SPECIAL = "test.special"
_REG_ALIAS32 = "test.alias32"
_REG_ALIAS64 = "test.alias64"

_REG_PART_I32_LOW16 = "test.i32.low16"
_REG_PART_I32_HIGH16 = "test.i32.high16"

_RESOURCE_SCALAR = "test.scalar"
_RESOURCE_VECTOR = "test.vector"
_RESOURCE_ADDRESS = "test.address"
_RESOURCE_LOAD = "test.load"
_RESOURCE_STORE = "test.store"
_RESOURCE_CALL = "test.call"
_RESOURCE_CONTROL = "test.control"

_SCHEDULE_CONST = "test.const"
_SCHEDULE_SCALAR_ALU = "test.scalar.alu"
_SCHEDULE_VECTOR_ALU = "test.vector.alu"
_SCHEDULE_LOAD = "test.load"
_SCHEDULE_STORE = "test.store"
_SCHEDULE_CALL = "test.call"
_SCHEDULE_CONTROL = "test.control"

_I32_ALT = (RegClassAlt(_REG_I32),)
_I8_ALT = (RegClassAlt(_REG_I8),)
_F32_ALT = (RegClassAlt(_REG_F32),)
_I32_I64_ALT = (RegClassAlt(_REG_I32), RegClassAlt(_REG_I64))
_PTR_ALT = (RegClassAlt(_REG_PTR),)
_PHYS_ALT = (RegClassAlt(_REG_PHYS),)
_SPECIAL_ALT = (RegClassAlt(_REG_SPECIAL),)


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str | AsmImmediate, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            results=results,
            operands=operands,
            immediates=tuple(
                immediate
                if isinstance(immediate, AsmImmediate)
                else AsmImmediate(immediate)
                for immediate in immediates
            ),
        ),
    )


def _i32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I32_ALT)


def _i32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I32_ALT)


def _i32_low16_result(field_name: str = "dst") -> Operand:
    return Operand(
        field_name, OperandRole.RESULT, _I32_ALT, register_part=_REG_PART_I32_LOW16
    )


def _i32_low16_operand(field_name: str) -> Operand:
    return Operand(
        field_name, OperandRole.OPERAND, _I32_ALT, register_part=_REG_PART_I32_LOW16
    )


def _i32_high16_result(field_name: str = "dst") -> Operand:
    return Operand(
        field_name, OperandRole.RESULT, _I32_ALT, register_part=_REG_PART_I32_HIGH16
    )


def _i32_high16_operand(field_name: str) -> Operand:
    return Operand(
        field_name, OperandRole.OPERAND, _I32_ALT, register_part=_REG_PART_I32_HIGH16
    )


def _i32_i64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I32_I64_ALT)


def _i32_i64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I32_I64_ALT)


def _f32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _F32_ALT)


def _f32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _F32_ALT)


def _i32_predicate(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _I32_ALT)


def _ptr_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _PTR_ALT)


def _v4i32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I32_ALT, unit_count=4)


def _v4i32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I32_ALT, unit_count=4)


def _phys_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PHYS_ALT)


def _phys_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _PHYS_ALT)


def _special_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _SPECIAL_ALT)


def _special_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SPECIAL_ALT)


def _special_state_read(field_name: str = "state_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SPECIAL_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
    )


def _special_state_write(field_name: str = "state_out") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SPECIAL_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
    )


_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_LANE_I32X4_IMMEDIATE = Immediate(
    "lane",
    ImmediateKind.UNSIGNED,
    bit_width=2,
    unsigned_max=3,
)

_SHUFFLE_CONTROL_IMMEDIATE = Immediate(
    "shuffle_control",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=255,
)

_SHUFFLE_BYTE_IMMEDIATES = tuple(
    Immediate(
        f"lane{i}",
        ImmediateKind.UNSIGNED,
        bit_width=8,
        unsigned_max=15,
    )
    for i in range(16)
)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_TRUE_BLOCK_IMMEDIATE = Immediate(
    "true_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_FALSE_BLOCK_IMMEDIATE = Immediate(
    "false_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_CALLEE_IMMEDIATE = Immediate(
    "callee_ordinal",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_CALL_EFFECT = Effect(
    EffectKind.CALL,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_CONVERGENT_EFFECT = Effect(
    EffectKind.CONVERGENT,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_TIED_RESULT_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)

TEST_LOW_CONST_I32_DESCRIPTOR = Descriptor(
    key="test.const.i32",
    mnemonic="test.const.i32",
    semantic_tag="integer.const.i32",
    operands=(_i32_result(),),
    immediates=(_I32_VALUE_IMMEDIATE,),
    asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
    schedule_class=_SCHEDULE_CONST,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ADD_I32_DESCRIPTOR = Descriptor(
    key="test.add.i32",
    mnemonic="test.add.i32",
    semantic_tag="integer.add.i32",
    operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_MUL_I32_DESCRIPTOR = Descriptor(
    key="test.mul.i32",
    mnemonic="test.mul.i32",
    semantic_tag="integer.mul.i32",
    operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_TIED_ANY_DESCRIPTOR = Descriptor(
    key="test.tied.any",
    mnemonic="test.tied.any",
    semantic_tag="test.tied.any",
    operands=(_i32_i64_result(), _i32_i64_operand("src")),
    constraints=_TIED_RESULT_CONSTRAINTS,
    asm_forms=_asm(results=("dst",), operands=("src",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ADD_F32_DESCRIPTOR = Descriptor(
    key="test.add.f32",
    mnemonic="test.add.f32",
    semantic_tag="float.add.f32",
    operands=(_f32_result(), _f32_operand("lhs"), _f32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_SUB_F32_DESCRIPTOR = Descriptor(
    key="test.sub.f32",
    mnemonic="test.sub.f32",
    semantic_tag="float.sub.f32",
    operands=(_f32_result(), _f32_operand("lhs"), _f32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_MUL_F32_DESCRIPTOR = Descriptor(
    key="test.mul.f32",
    mnemonic="test.mul.f32",
    semantic_tag="float.mul.f32",
    operands=(_f32_result(), _f32_operand("lhs"), _f32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_CMP_EQ_I32_DESCRIPTOR = Descriptor(
    key="test.cmp.eq.i32",
    mnemonic="test.cmp.eq.i32",
    semantic_tag="integer.cmp.eq.i32",
    operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_SELECT_I32_DESCRIPTOR = Descriptor(
    key="test.select.i32",
    mnemonic="test.select.i32",
    semantic_tag="integer.select.i32",
    operands=(
        _i32_result(),
        _i32_predicate("condition"),
        _i32_operand("true_value"),
        _i32_operand("false_value"),
    ),
    asm_forms=_asm(
        results=("dst",),
        operands=("condition", "true_value", "false_value"),
    ),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ADD_V4I32_DESCRIPTOR = Descriptor(
    key="test.add.v4i32",
    mnemonic="test.add.v4i32",
    semantic_tag="vector.add.i32x4",
    operands=(
        _v4i32_result(),
        _v4i32_operand("lhs"),
        _v4i32_operand("rhs"),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_DOT4I_S8S8_DESCRIPTOR = Descriptor(
    key="test.dot4i.s8s8",
    mnemonic="test.dot4i.s8s8",
    semantic_tag="integer.dot4.s8s8",
    operands=(
        Operand("dst", OperandRole.RESULT, _I32_ALT, unit_count=4),
        Operand("lhs", OperandRole.OPERAND, _I8_ALT, unit_count=16),
        Operand("rhs", OperandRole.OPERAND, _I8_ALT, unit_count=16),
        Operand("acc", OperandRole.OPERAND, _I32_ALT, unit_count=4),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs", "acc")),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_MMA_I32_2X2X2_DESCRIPTOR = Descriptor(
    key="test.mma.i32.2x2x2",
    mnemonic="test.mma.i32.2x2x2",
    semantic_tag="matrix.mma.i32.2x2x2",
    operands=(
        Operand("dst", OperandRole.RESULT, _I32_ALT, unit_count=4),
        Operand("a", OperandRole.OPERAND, _I32_ALT, unit_count=4),
        Operand("b", OperandRole.OPERAND, _I32_ALT, unit_count=4),
        Operand("acc", OperandRole.OPERAND, _I32_ALT, unit_count=4),
    ),
    asm_forms=_asm(results=("dst",), operands=("a", "b", "acc")),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR = Descriptor(
    key="test.shuffle.v4i32",
    mnemonic="test.shuffle.v4i32",
    semantic_tag="vector.shuffle.i32x4",
    operands=(
        Operand("dst", OperandRole.RESULT, _I32_ALT, unit_count=4),
        Operand("source", OperandRole.OPERAND, _I32_ALT, unit_count=4),
    ),
    immediates=(_SHUFFLE_CONTROL_IMMEDIATE,),
    asm_forms=_asm(
        results=("dst",),
        operands=("source",),
        immediates=("shuffle_control",),
    ),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR = Descriptor(
    key="test.from_elements.v4i32",
    mnemonic="test.from_elements.v4i32",
    semantic_tag="vector.from_elements.i32x4",
    operands=(
        _v4i32_result(),
        _i32_operand("lane0"),
        _i32_operand("lane1"),
        _i32_operand("lane2"),
        _i32_operand("lane3"),
    ),
    asm_forms=_asm(
        results=("dst",),
        operands=("lane0", "lane1", "lane2", "lane3"),
    ),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR = Descriptor(
    key="test.cmp.eq.v4i32",
    mnemonic="test.cmp.eq.v4i32",
    semantic_tag="vector.cmp.eq.i32x4",
    operands=(
        _v4i32_result(),
        _v4i32_operand("lhs"),
        _v4i32_operand("rhs"),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR = Descriptor(
    key="test.extract_lane.i32",
    mnemonic="test.extract_lane.i32",
    semantic_tag="vector.extract.i32x4",
    operands=(_i32_result(), _v4i32_operand("source")),
    immediates=(_LANE_I32X4_IMMEDIATE,),
    asm_forms=_asm(results=("dst",), operands=("source",), immediates=("lane",)),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_SHUFFLE_BYTES_DESCRIPTOR = Descriptor(
    key="test.shuffle.bytes",
    mnemonic="test.shuffle.bytes",
    semantic_tag="vector.shuffle.i8x16",
    operands=(
        _v4i32_result(),
        _v4i32_operand("lhs"),
        _v4i32_operand("rhs"),
    ),
    immediates=_SHUFFLE_BYTE_IMMEDIATES,
    asm_forms=_asm(
        results=("dst",),
        operands=("lhs", "rhs"),
        immediates=tuple(
            immediate.field_name for immediate in _SHUFFLE_BYTE_IMMEDIATES
        ),
    ),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_CONVERGENT_I32_DESCRIPTOR = Descriptor(
    key="test.convergent.i32",
    mnemonic="test.convergent.i32",
    semantic_tag="test.convergent.i32",
    operands=(_i32_result(), _i32_operand("input")),
    effects=(_CONVERGENT_EFFECT,),
    asm_forms=_asm(results=("dst",), operands=("input",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
)

TEST_LOW_AMBIGUOUS_DESCRIPTOR = Descriptor(
    key="test.ambiguous",
    mnemonic="test.ambiguous",
    semantic_tag="test.ambiguous",
    operands=(_i32_i64_result(),),
    asm_forms=_asm(results=("dst",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_READ_LOW16_I32_DESCRIPTOR = Descriptor(
    key="test.read.low16.i32",
    mnemonic="test.read.low16.i32",
    semantic_tag="test.read.low16.i32",
    operands=(_i32_result(), _i32_low16_operand("src")),
    asm_forms=_asm(results=("dst",), operands=("src",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_READ_HIGH16_I32_DESCRIPTOR = Descriptor(
    key="test.read.high16.i32",
    mnemonic="test.read.high16.i32",
    semantic_tag="test.read.high16.i32",
    operands=(_i32_result(), _i32_high16_operand("src")),
    asm_forms=_asm(results=("dst",), operands=("src",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR = Descriptor(
    key="test.write.low16.i32",
    mnemonic="test.write.low16.i32",
    semantic_tag="test.write.low16.i32",
    operands=(_i32_low16_result(), _ptr_resource("address")),
    asm_forms=_asm(results=("dst",), operands=("address",)),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_TIED_WRITE_LOW16_I32_DESCRIPTOR = Descriptor(
    key="test.write.low16.tied.i32",
    mnemonic="test.write.low16.tied.i32",
    semantic_tag="test.write.low16.tied.i32",
    operands=(
        _i32_low16_result(),
        _i32_high16_operand("src"),
        _ptr_resource("address"),
    ),
    constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    asm_forms=_asm(results=("dst",), operands=("src", "address")),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_WRITE_HIGH16_I32_DESCRIPTOR = Descriptor(
    key="test.write.high16.i32",
    mnemonic="test.write.high16.i32",
    semantic_tag="test.write.high16.i32",
    operands=(
        _i32_high16_result(),
        _i32_low16_operand("src"),
        _ptr_resource("address"),
    ),
    constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    asm_forms=_asm(results=("dst",), operands=("src", "address")),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_SPV_OP_IADD_I32_DESCRIPTOR = Descriptor(
    key="test.spv.op_iadd.i32",
    mnemonic="OpIAdd",
    semantic_tag="spirv.op_iadd.i32",
    operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ADD_PHYS_DESCRIPTOR = Descriptor(
    key="test.add.phys",
    mnemonic="test.add.phys",
    semantic_tag="test.physical.add",
    operands=(_phys_result(), _phys_operand("lhs"), _phys_operand("rhs")),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_VECTOR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ADD_SPECIAL_DESCRIPTOR = Descriptor(
    key="test.add.special",
    mnemonic="test.add.special",
    semantic_tag="test.special.add",
    operands=(
        _special_result(),
        _special_operand("lhs"),
        _special_operand("rhs"),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_STATE_ADD_I32_DESCRIPTOR = Descriptor(
    key="test.state.add.i32",
    mnemonic="test.state.add.i32",
    semantic_tag="test.state.add.i32",
    operands=(
        _i32_result(),
        _i32_operand("lhs"),
        _i32_operand("rhs"),
        _special_state_write(),
        _special_state_read(),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
    operand_forms=(
        OperandForm(
            replacement_descriptor="test.state.add.i32.rhs_zero",
            matches=(
                OperandFormMatch(
                    source_operand="rhs",
                    match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                    match_i64=0,
                ),
            ),
        ),
    ),
)

TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR = Descriptor(
    key="test.state.add.i32.rhs_zero",
    mnemonic="test.state.add.i32.rhs_zero",
    semantic_tag="test.state.add.i32",
    operands=(
        _i32_result(),
        _i32_operand("lhs"),
        _special_state_write(),
        _special_state_read(),
    ),
    asm_forms=_asm(results=("dst",), operands=("lhs",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_LOAD_V4I32_DESCRIPTOR = Descriptor(
    key="test.load.v4i32",
    mnemonic="test.load.v4i32",
    semantic_tag="memory.load.v128",
    operands=(
        Operand("dst", OperandRole.RESULT, _I32_ALT, unit_count=4),
        _ptr_resource("address"),
    ),
    asm_forms=_asm(results=("dst",), operands=("address",)),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_LOAD_V4F32_DESCRIPTOR = Descriptor(
    key="test.load.v4f32",
    mnemonic="test.load.v4f32",
    semantic_tag="memory.load.v128.f32",
    operands=(
        Operand("dst", OperandRole.RESULT, _F32_ALT, unit_count=4),
        _ptr_resource("address"),
    ),
    asm_forms=_asm(results=("dst",), operands=("address",)),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR = Descriptor(
    key="test.load.index.v4i32",
    mnemonic="test.load.index.v4i32",
    semantic_tag="memory.load.index.v128",
    operands=(
        Operand("dst", OperandRole.RESULT, _I32_ALT, unit_count=4),
        _ptr_resource("address"),
        _i32_operand("index"),
    ),
    asm_forms=_asm(results=("dst",), operands=("address", "index")),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_LOAD_INDEX_V4F32_DESCRIPTOR = Descriptor(
    key="test.load.index.v4f32",
    mnemonic="test.load.index.v4f32",
    semantic_tag="memory.load.index.v128.f32",
    operands=(
        Operand("dst", OperandRole.RESULT, _F32_ALT, unit_count=4),
        _ptr_resource("address"),
        _i32_operand("index"),
    ),
    asm_forms=_asm(results=("dst",), operands=("address", "index")),
    effects=(_LOAD_EFFECT,),
    schedule_class=_SCHEDULE_LOAD,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_STORE_V4I32_DESCRIPTOR = Descriptor(
    key="test.store.v4i32",
    mnemonic="test.store.v4i32",
    semantic_tag="memory.store.v128",
    operands=(
        _ptr_resource("address"),
        Operand("value", OperandRole.OPERAND, _I32_ALT, unit_count=4),
    ),
    asm_forms=_asm(operands=("address", "value")),
    effects=(_STORE_EFFECT,),
    schedule_class=_SCHEDULE_STORE,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_STORE_V4F32_DESCRIPTOR = Descriptor(
    key="test.store.v4f32",
    mnemonic="test.store.v4f32",
    semantic_tag="memory.store.v128.f32",
    operands=(
        _ptr_resource("address"),
        Operand("value", OperandRole.OPERAND, _F32_ALT, unit_count=4),
    ),
    asm_forms=_asm(operands=("address", "value")),
    effects=(_STORE_EFFECT,),
    schedule_class=_SCHEDULE_STORE,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_STORE_INDEX_V4I32_DESCRIPTOR = Descriptor(
    key="test.store.index.v4i32",
    mnemonic="test.store.index.v4i32",
    semantic_tag="memory.store.index.v128",
    operands=(
        _ptr_resource("address"),
        _i32_operand("index"),
        Operand("value", OperandRole.OPERAND, _I32_ALT, unit_count=4),
    ),
    asm_forms=_asm(operands=("address", "index", "value")),
    effects=(_STORE_EFFECT,),
    schedule_class=_SCHEDULE_STORE,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_STORE_INDEX_V4F32_DESCRIPTOR = Descriptor(
    key="test.store.index.v4f32",
    mnemonic="test.store.index.v4f32",
    semantic_tag="memory.store.index.v128.f32",
    operands=(
        _ptr_resource("address"),
        _i32_operand("index"),
        Operand("value", OperandRole.OPERAND, _F32_ALT, unit_count=4),
    ),
    asm_forms=_asm(operands=("address", "index", "value")),
    effects=(_STORE_EFFECT,),
    schedule_class=_SCHEDULE_STORE,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_CALL_I32_DESCRIPTOR = Descriptor(
    key="test.call.i32",
    mnemonic="test.call.i32",
    semantic_tag="call.import.i32",
    operands=(_i32_result(), _i32_operand("arg0")),
    immediates=(_CALLEE_IMMEDIATE,),
    asm_forms=_asm(
        results=("dst",),
        operands=("arg0",),
        immediates=(AsmImmediate("callee_ordinal", name="callee"),),
    ),
    effects=(_CALL_EFFECT,),
    schedule_class=_SCHEDULE_CALL,
    flags=(DescriptorFlag.SIDE_EFFECTING,),
)

TEST_LOW_BR_DESCRIPTOR = Descriptor(
    key="test.br",
    mnemonic="test.br",
    semantic_tag="control.branch",
    operands=(),
    immediates=(_TARGET_BLOCK_IMMEDIATE,),
    asm_forms=_asm(immediates=("target_block",)),
    effects=(_CONTROL_EFFECT,),
    schedule_class=_SCHEDULE_CONTROL,
    flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
)

TEST_LOW_COND_BR_I32_DESCRIPTOR = Descriptor(
    key="test.cond_br.i32",
    mnemonic="test.cond_br.i32",
    semantic_tag="control.cond_branch.i32",
    operands=(_i32_predicate("cond"),),
    immediates=(_TRUE_BLOCK_IMMEDIATE, _FALSE_BLOCK_IMMEDIATE),
    asm_forms=_asm(operands=("cond",), immediates=("true_block", "false_block")),
    effects=(_CONTROL_EFFECT,),
    schedule_class=_SCHEDULE_CONTROL,
    flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
)

TEST_LOW_RETURN_I32_DESCRIPTOR = Descriptor(
    key="test.return.i32",
    mnemonic="test.return.i32",
    semantic_tag="control.return.i32",
    operands=(_i32_operand("value"),),
    asm_forms=_asm(operands=("value",)),
    effects=(_CONTROL_EFFECT,),
    schedule_class=_SCHEDULE_CONTROL,
    flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
)

TEST_LOW_RETURN_VOID_DESCRIPTOR = Descriptor(
    key="test.return.void",
    mnemonic="test.return.void",
    semantic_tag="control.return.void",
    operands=(),
    asm_forms=_asm(),
    effects=(_CONTROL_EFFECT,),
    schedule_class=_SCHEDULE_CONTROL,
    flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
)

TEST_LOW_ALT_CONST_I32_DESCRIPTOR = Descriptor(
    key="test.alt.const.i32",
    mnemonic="test.alt.const.i32",
    semantic_tag="test.alt.integer.const.i32",
    operands=(_i32_result(),),
    immediates=(_I32_VALUE_IMMEDIATE,),
    asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
    schedule_class=_SCHEDULE_CONST,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_ALT_NEG_I32_DESCRIPTOR = Descriptor(
    key="test.alt.neg.i32",
    mnemonic="test.alt.neg.i32",
    semantic_tag="test.alt.integer.neg.i32",
    operands=(_i32_result(), _i32_operand("value")),
    asm_forms=_asm(results=("dst",), operands=("value",)),
    schedule_class=_SCHEDULE_SCALAR_ALU,
    flags=(DescriptorFlag.DEAD_REMOVABLE,),
)

TEST_LOW_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="test.low.core",
    target_key="test.low",
    feature_key="test.low.v1",
    c_header_path=Path("loom/src/loom/target/test/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/test/descriptors.c"),
    header_guard="LOOM_TARGET_TEST_DESCRIPTORS_H_",
    public_header="loom/target/test/descriptors.h",
    function_name="loom_test_low_core_descriptor_set",
    c_table_prefix="TestLowCore",
    c_enum_prefix="TEST_LOW_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I32,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            full_register_part_mask=0x3,
        ),
        RegClass(
            _REG_I8, 8, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_PTR, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_PHYS,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=32,
        ),
        RegClass(
            _REG_SPECIAL,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_ALIAS32,
            32,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=1,
            alias_set_id=1,
        ),
        RegClass(
            _REG_ALIAS64,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=1,
            alias_set_id=1,
        ),
    ),
    register_parts=(
        RegisterPart(_REG_PART_I32_LOW16, _REG_I32, 0x1),
        RegisterPart(_REG_PART_I32_HIGH16, _REG_I32, 0x2),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VECTOR, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_ADDRESS, capacity_per_cycle=1, kind=ResourceKind.ADDRESS),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_CALL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_ALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=2,
            issue_uses=(IssueUse(_RESOURCE_VECTOR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1, stage=0),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=1, stage=1),
            ),
            hazards=(
                Hazard(
                    HazardKind.MIN_DISTANCE,
                    resource=_RESOURCE_LOAD,
                    producer_stage=1,
                    consumer_stage=1,
                    distance=2,
                ),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=2,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1, stage=0),
                IssueUse(_RESOURCE_STORE, cycles=1, units=1, stage=1),
            ),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CALL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_CALL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_CALL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CONTROL,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    descriptors=(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        TEST_LOW_ADD_I32_DESCRIPTOR,
        TEST_LOW_CONVERGENT_I32_DESCRIPTOR,
        TEST_LOW_MUL_I32_DESCRIPTOR,
        TEST_LOW_ADD_F32_DESCRIPTOR,
        TEST_LOW_SUB_F32_DESCRIPTOR,
        TEST_LOW_MUL_F32_DESCRIPTOR,
        TEST_LOW_AMBIGUOUS_DESCRIPTOR,
        TEST_LOW_TIED_ANY_DESCRIPTOR,
        TEST_LOW_READ_LOW16_I32_DESCRIPTOR,
        TEST_LOW_READ_HIGH16_I32_DESCRIPTOR,
        TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR,
        TEST_LOW_TIED_WRITE_LOW16_I32_DESCRIPTOR,
        TEST_LOW_WRITE_HIGH16_I32_DESCRIPTOR,
        TEST_LOW_SPV_OP_IADD_I32_DESCRIPTOR,
        TEST_LOW_CMP_EQ_I32_DESCRIPTOR,
        TEST_LOW_SELECT_I32_DESCRIPTOR,
        TEST_LOW_ADD_V4I32_DESCRIPTOR,
        TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
        TEST_LOW_MMA_I32_2X2X2_DESCRIPTOR,
        TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR,
        TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
        TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
        TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR,
        TEST_LOW_SHUFFLE_BYTES_DESCRIPTOR,
        TEST_LOW_ADD_PHYS_DESCRIPTOR,
        TEST_LOW_ADD_SPECIAL_DESCRIPTOR,
        TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
        TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
        TEST_LOW_LOAD_V4I32_DESCRIPTOR,
        TEST_LOW_LOAD_V4F32_DESCRIPTOR,
        TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
        TEST_LOW_LOAD_INDEX_V4F32_DESCRIPTOR,
        TEST_LOW_STORE_V4I32_DESCRIPTOR,
        TEST_LOW_STORE_V4F32_DESCRIPTOR,
        TEST_LOW_STORE_INDEX_V4I32_DESCRIPTOR,
        TEST_LOW_STORE_INDEX_V4F32_DESCRIPTOR,
        TEST_LOW_CALL_I32_DESCRIPTOR,
        TEST_LOW_BR_DESCRIPTOR,
        TEST_LOW_COND_BR_I32_DESCRIPTOR,
        TEST_LOW_RETURN_I32_DESCRIPTOR,
        TEST_LOW_RETURN_VOID_DESCRIPTOR,
    ),
)


TEST_LOW_ALT_DESCRIPTOR_SET = DescriptorSet(
    key="test.low.alt",
    target_key="test.low",
    feature_key="test.low.v1.alt",
    c_header_path=Path("loom/src/loom/target/test/alt_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/test/alt_descriptors.c"),
    header_guard="LOOM_TARGET_TEST_ALT_DESCRIPTORS_H_",
    public_header="loom/target/test/alt_descriptors.h",
    function_name="loom_test_low_alt_descriptor_set",
    c_table_prefix="TestLowAlt",
    c_enum_prefix="TEST_LOW_ALT",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    descriptors=(
        TEST_LOW_ALT_CONST_I32_DESCRIPTOR,
        TEST_LOW_ALT_NEG_I32_DESCRIPTOR,
    ),
)
