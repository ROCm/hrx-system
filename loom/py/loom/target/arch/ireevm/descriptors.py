# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the IREE VM low target."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
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

_REG_I32 = "ireevm.i32"
_REG_I64 = "ireevm.i64"
_REG_F32 = "ireevm.f32"
_REG_F64 = "ireevm.f64"
_REG_REF = "ireevm.ref"

_SCHEDULE_CONST = "ireevm.const"
_SCHEDULE_ALU = "ireevm.alu"
_SCHEDULE_MATH = "ireevm.math"
_SCHEDULE_CONVERT = "ireevm.convert"
_SCHEDULE_REF = "ireevm.ref"
_SCHEDULE_BUFFER_LOAD = "ireevm.buffer.load"
_SCHEDULE_BUFFER_STORE = "ireevm.buffer.store"
_SCHEDULE_CONTROL = "ireevm.control"

_RESOURCE_ALU = "ireevm.alu"
_RESOURCE_MATH = "ireevm.math"
_RESOURCE_REF = "ireevm.ref"
_RESOURCE_BUFFER_LOAD = "ireevm.buffer.load"
_RESOURCE_BUFFER_STORE = "ireevm.buffer.store"
_RESOURCE_CONTROL = "ireevm.control"

_SCALAR_ALIAS_SET = 1
_REF_ALIAS_SET = 2
_SCALAR_TARGET_BANK = 1
_REF_TARGET_BANK = 2

_REG_BY_TYPE = {
    "i32": _REG_I32,
    "i64": _REG_I64,
    "f32": _REG_F32,
    "f64": _REG_F64,
    "ref": _REG_REF,
}
_UNIT_COUNT_BY_TYPE = {
    "i32": 1,
    "i64": 2,
    "f32": 1,
    "f64": 2,
    "ref": 1,
}
_ALT_BY_TYPE = {
    type_name: (RegClassAlt(reg_class),)
    for type_name, reg_class in _REG_BY_TYPE.items()
}


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(field_name) for field_name in immediates),
        ),
    )


def _value(
    type_name: str,
    role: OperandRole,
    field_name: str,
) -> Operand:
    return Operand(
        field_name,
        role,
        _ALT_BY_TYPE[type_name],
        unit_count=_UNIT_COUNT_BY_TYPE[type_name],
    )


def _result(type_name: str, field_name: str = "dst") -> Operand:
    return _value(type_name, OperandRole.RESULT, field_name)


def _operand(type_name: str, field_name: str) -> Operand:
    return _value(type_name, OperandRole.OPERAND, field_name)


def _predicate_i32(field_name: str = "condition") -> Operand:
    return _value("i32", OperandRole.PREDICATE, field_name)


_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_I64_VALUE_IMMEDIATE = Immediate(
    "i64_value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_F32_BITS_IMMEDIATE = Immediate(
    "f32_bits",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_F64_BITS_IMMEDIATE = Immediate(
    "f64_bits",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
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

# VM bytecode opcodes from runtime/src/iree/vm/bytecode/utils/generated/op_table.h.
_OP_PREFIX_F32 = 0xE0
_OP_PREFIX_F64 = 0xE1

_FEATURE_EXT_F32 = 1 << 0
_FEATURE_EXT_F64 = 1 << 1


def _feature_mask_words(encoding_id: int) -> tuple[int, ...]:
    prefix = encoding_id >> 8
    if prefix == _OP_PREFIX_F32:
        return (_FEATURE_EXT_F32,)
    if prefix == _OP_PREFIX_F64:
        return (_FEATURE_EXT_F64,)
    return ()


def _ext_f32(subopcode: int) -> int:
    return (_OP_PREFIX_F32 << 8) | subopcode


def _ext_f64(subopcode: int) -> int:
    return (_OP_PREFIX_F64 << 8) | subopcode


_OP_CONST_I32 = 0x0D
_OP_CONST_I64 = 0x0F
_OP_SELECT_I32 = 0x1C
_OP_SELECT_I64 = 0x1D
_OP_ADD_I32 = 0x22
_OP_SUB_I32 = 0x23
_OP_MUL_I32 = 0x24
_OP_DIV_S_I32 = 0x25
_OP_DIV_U_I32 = 0x26
_OP_REM_S_I32 = 0x27
_OP_REM_U_I32 = 0x28
_OP_FMA_I32 = 0x29
_OP_ADD_I64 = 0x2A
_OP_SUB_I64 = 0x2B
_OP_MUL_I64 = 0x2C
_OP_DIV_S_I64 = 0x2D
_OP_DIV_U_I64 = 0x2E
_OP_REM_S_I64 = 0x2F
_OP_REM_U_I64 = 0x30
_OP_FMA_I64 = 0x31
_OP_NOT_I32 = 0x32
_OP_AND_I32 = 0x33
_OP_OR_I32 = 0x34
_OP_XOR_I32 = 0x35
_OP_NOT_I64 = 0x36
_OP_AND_I64 = 0x37
_OP_OR_I64 = 0x38
_OP_XOR_I64 = 0x39
_OP_SHL_I32 = 0x3A
_OP_SHR_S_I32 = 0x3B
_OP_SHR_U_I32 = 0x3C
_OP_SHL_I64 = 0x3D
_OP_SHR_S_I64 = 0x3E
_OP_SHR_U_I64 = 0x3F
_OP_TRUNC_I32_I8 = 0x40
_OP_TRUNC_I32_I16 = 0x41
_OP_TRUNC_I64_I32 = 0x42
_OP_EXT_I8_I32_S = 0x43
_OP_EXT_I8_I32_U = 0x44
_OP_EXT_I16_I32_S = 0x45
_OP_EXT_I16_I32_U = 0x46
_OP_EXT_I32_I64_S = 0x47
_OP_EXT_I32_I64_U = 0x48
_OP_CMP_EQ_I32 = 0x49
_OP_CMP_NE_I32 = 0x4A
_OP_CMP_LT_S_I32 = 0x4B
_OP_CMP_LT_U_I32 = 0x4C
_OP_CMP_NZ_I32 = 0x4D
_OP_CMP_EQ_I64 = 0x4E
_OP_CMP_NE_I64 = 0x4F
_OP_CMP_LT_S_I64 = 0x50
_OP_CMP_LT_U_I64 = 0x51
_OP_CMP_NZ_I64 = 0x52
_OP_BRANCH = 0x56
_OP_COND_BRANCH = 0x57
_OP_RETURN = 0x5A
_OP_BUFFER_LOAD_I8_U = 0x62
_OP_BUFFER_LOAD_I8_S = 0x63
_OP_BUFFER_LOAD_I16_U = 0x64
_OP_BUFFER_LOAD_I16_S = 0x65
_OP_BUFFER_LOAD_I32 = 0x66
_OP_BUFFER_LOAD_I64 = 0x67
_OP_BUFFER_STORE_I8 = 0x68
_OP_BUFFER_STORE_I16 = 0x69
_OP_BUFFER_STORE_I32 = 0x6A
_OP_BUFFER_STORE_I64 = 0x6B
_OP_BUFFER_LENGTH = 0x6E
_OP_CTLZ_I32 = 0x75
_OP_CTLZ_I64 = 0x76
_OP_ABS_I32 = 0x77
_OP_ABS_I64 = 0x78
_OP_MIN_S_I32 = 0x7A
_OP_MIN_U_I32 = 0x7B
_OP_MAX_S_I32 = 0x7C
_OP_MAX_U_I32 = 0x7D
_OP_MIN_S_I64 = 0x7E
_OP_MIN_U_I64 = 0x7F
_OP_MAX_S_I64 = 0x80
_OP_MAX_U_I64 = 0x81
_OP_DISCARD_REFS = 0x85
_OP_ASSIGN_REF = 0x86

_OP_CONST_F32 = _ext_f32(0x05)
_OP_SELECT_F32 = _ext_f32(0x08)
_OP_ADD_F32 = _ext_f32(0x0A)
_OP_SUB_F32 = _ext_f32(0x0B)
_OP_MUL_F32 = _ext_f32(0x0C)
_OP_DIV_F32 = _ext_f32(0x0D)
_OP_REM_F32 = _ext_f32(0x0E)
_OP_FMA_F32 = _ext_f32(0x0F)
_OP_ABS_F32 = _ext_f32(0x10)
_OP_NEG_F32 = _ext_f32(0x11)
_OP_CEIL_F32 = _ext_f32(0x12)
_OP_FLOOR_F32 = _ext_f32(0x13)
_OP_CAST_SI32_F32 = _ext_f32(0x14)
_OP_CAST_UI32_F32 = _ext_f32(0x15)
_OP_CAST_F32_SI32 = _ext_f32(0x16)
_OP_CAST_F32_UI32 = _ext_f32(0x17)
_OP_BITCAST_I32_F32 = _ext_f32(0x18)
_OP_BITCAST_F32_I32 = _ext_f32(0x19)
_OP_ATAN_F32 = _ext_f32(0x1A)
_OP_ATAN2_F32 = _ext_f32(0x1B)
_OP_COS_F32 = _ext_f32(0x1C)
_OP_SIN_F32 = _ext_f32(0x1D)
_OP_EXP_F32 = _ext_f32(0x1E)
_OP_EXP2_F32 = _ext_f32(0x1F)
_OP_EXPM1_F32 = _ext_f32(0x20)
_OP_LOG_F32 = _ext_f32(0x21)
_OP_LOG10_F32 = _ext_f32(0x22)
_OP_LOG1P_F32 = _ext_f32(0x23)
_OP_LOG2_F32 = _ext_f32(0x24)
_OP_POW_F32 = _ext_f32(0x25)
_OP_RSQRT_F32 = _ext_f32(0x26)
_OP_SQRT_F32 = _ext_f32(0x27)
_OP_TANH_F32 = _ext_f32(0x28)
_OP_ERF_F32 = _ext_f32(0x29)
_OP_CMP_EQ_O_F32 = _ext_f32(0x2A)
_OP_CMP_EQ_U_F32 = _ext_f32(0x2B)
_OP_CMP_NE_O_F32 = _ext_f32(0x2C)
_OP_CMP_NE_U_F32 = _ext_f32(0x2D)
_OP_CMP_LT_O_F32 = _ext_f32(0x2E)
_OP_CMP_LT_U_F32 = _ext_f32(0x2F)
_OP_CMP_LE_O_F32 = _ext_f32(0x30)
_OP_CMP_LE_U_F32 = _ext_f32(0x31)
_OP_CMP_NAN_F32 = _ext_f32(0x32)
_OP_BUFFER_LOAD_F32 = _ext_f32(0x33)
_OP_BUFFER_STORE_F32 = _ext_f32(0x34)
_OP_ROUND_F32 = _ext_f32(0x36)
_OP_MIN_F32 = _ext_f32(0x37)
_OP_MAX_F32 = _ext_f32(0x38)
_OP_ROUND_EVEN_F32 = _ext_f32(0x39)
_OP_CAST_F32_SI64 = _ext_f32(0x3A)
_OP_CAST_F32_UI64 = _ext_f32(0x3B)
_OP_CAST_SI64_F32 = _ext_f32(0x3C)
_OP_CAST_UI64_F32 = _ext_f32(0x3D)

_OP_CONST_F64 = _ext_f64(0x05)
_OP_SELECT_F64 = _ext_f64(0x08)
_OP_ADD_F64 = _ext_f64(0x0A)
_OP_SUB_F64 = _ext_f64(0x0B)
_OP_MUL_F64 = _ext_f64(0x0C)
_OP_DIV_F64 = _ext_f64(0x0D)
_OP_REM_F64 = _ext_f64(0x0E)
_OP_FMA_F64 = _ext_f64(0x0F)
_OP_ABS_F64 = _ext_f64(0x10)
_OP_NEG_F64 = _ext_f64(0x11)
_OP_CEIL_F64 = _ext_f64(0x12)
_OP_FLOOR_F64 = _ext_f64(0x13)
_OP_TRUNC_F64_F32 = _ext_f64(0x14)
_OP_EXT_F32_F64 = _ext_f64(0x15)
_OP_CAST_SI32_F64 = _ext_f64(0x16)
_OP_CAST_UI32_F64 = _ext_f64(0x17)
_OP_CAST_F64_SI32 = _ext_f64(0x18)
_OP_CAST_F64_UI32 = _ext_f64(0x19)
_OP_CAST_SI64_F64 = _ext_f64(0x1A)
_OP_CAST_UI64_F64 = _ext_f64(0x1B)
_OP_CAST_F64_SI64 = _ext_f64(0x1C)
_OP_CAST_F64_UI64 = _ext_f64(0x1D)
_OP_BITCAST_I64_F64 = _ext_f64(0x1E)
_OP_BITCAST_F64_I64 = _ext_f64(0x1F)
_OP_ATAN_F64 = _ext_f64(0x20)
_OP_ATAN2_F64 = _ext_f64(0x21)
_OP_COS_F64 = _ext_f64(0x22)
_OP_SIN_F64 = _ext_f64(0x23)
_OP_EXP_F64 = _ext_f64(0x24)
_OP_EXP2_F64 = _ext_f64(0x25)
_OP_EXPM1_F64 = _ext_f64(0x26)
_OP_LOG_F64 = _ext_f64(0x27)
_OP_LOG10_F64 = _ext_f64(0x28)
_OP_LOG1P_F64 = _ext_f64(0x29)
_OP_LOG2_F64 = _ext_f64(0x2A)
_OP_POW_F64 = _ext_f64(0x2B)
_OP_RSQRT_F64 = _ext_f64(0x2C)
_OP_SQRT_F64 = _ext_f64(0x2D)
_OP_TANH_F64 = _ext_f64(0x2E)
_OP_ERF_F64 = _ext_f64(0x2F)
_OP_CMP_EQ_O_F64 = _ext_f64(0x30)
_OP_CMP_EQ_U_F64 = _ext_f64(0x31)
_OP_CMP_NE_O_F64 = _ext_f64(0x32)
_OP_CMP_NE_U_F64 = _ext_f64(0x33)
_OP_CMP_LT_O_F64 = _ext_f64(0x34)
_OP_CMP_LT_U_F64 = _ext_f64(0x35)
_OP_CMP_LE_O_F64 = _ext_f64(0x36)
_OP_CMP_LE_U_F64 = _ext_f64(0x37)
_OP_CMP_NAN_F64 = _ext_f64(0x38)
_OP_BUFFER_LOAD_F64 = _ext_f64(0x39)
_OP_BUFFER_STORE_F64 = _ext_f64(0x3A)
_OP_ROUND_F64 = _ext_f64(0x3C)
_OP_MIN_F64 = _ext_f64(0x3D)
_OP_MAX_F64 = _ext_f64(0x3E)
_OP_ROUND_EVEN_F64 = _ext_f64(0x3F)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_REF_READ_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.VM_REF,
    flags=(EffectFlag.DEPENDENCY,),
)

_REF_WRITE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.VM_REF,
    flags=(EffectFlag.DEPENDENCY,),
)

_BUFFER_READ_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
)

_BUFFER_WRITE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
)


def _const_descriptor(
    type_name: str,
    *,
    immediate: Immediate,
    encoding_id: int,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.const.{type_name}",
        mnemonic=f"ireevm.const.{type_name}",
        semantic_tag=f"{type_name}.const",
        operands=(_result(type_name),),
        immediates=(immediate,),
        asm_forms=_asm(results=("dst",), immediates=(immediate.field_name,)),
        schedule_class=_SCHEDULE_CONST,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _unary_descriptor(
    stem: str,
    type_name: str,
    *,
    encoding_id: int,
    schedule_class: str = _SCHEDULE_ALU,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.{stem}.{type_name}",
        mnemonic=f"ireevm.{stem}.{type_name}",
        semantic_tag=f"{stem}.{type_name}",
        operands=(_result(type_name), _operand(type_name, "input")),
        asm_forms=_asm(results=("dst",), operands=("input",)),
        schedule_class=schedule_class,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _binary_descriptor(
    stem: str,
    type_name: str,
    *,
    encoding_id: int,
    schedule_class: str = _SCHEDULE_ALU,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.{stem}.{type_name}",
        mnemonic=f"ireevm.{stem}.{type_name}",
        semantic_tag=f"{stem}.{type_name}",
        operands=(
            _result(type_name),
            _operand(type_name, "lhs"),
            _operand(type_name, "rhs"),
        ),
        asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
        schedule_class=schedule_class,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _ternary_descriptor(
    stem: str,
    type_name: str,
    *,
    encoding_id: int,
    schedule_class: str = _SCHEDULE_ALU,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.{stem}.{type_name}",
        mnemonic=f"ireevm.{stem}.{type_name}",
        semantic_tag=f"{stem}.{type_name}",
        operands=(
            _result(type_name),
            _operand(type_name, "a"),
            _operand(type_name, "b"),
            _operand(type_name, "c"),
        ),
        asm_forms=_asm(results=("dst",), operands=("a", "b", "c")),
        schedule_class=schedule_class,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _shift_descriptor(stem: str, type_name: str, *, encoding_id: int) -> Descriptor:
    return Descriptor(
        key=f"ireevm.{stem}.{type_name}",
        mnemonic=f"ireevm.{stem}.{type_name}",
        semantic_tag=f"{stem}.{type_name}",
        operands=(
            _result(type_name),
            _operand(type_name, "input"),
            _operand("i32", "amount"),
        ),
        asm_forms=_asm(results=("dst",), operands=("input", "amount")),
        schedule_class=_SCHEDULE_ALU,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _convert_descriptor(
    stem: str,
    source_type: str,
    result_type: str,
    *,
    encoding_id: int,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.{stem}.{source_type}.{result_type}",
        mnemonic=f"ireevm.{stem}.{source_type}.{result_type}",
        semantic_tag=f"{stem}.{source_type}.{result_type}",
        operands=(_result(result_type), _operand(source_type, "input")),
        asm_forms=_asm(results=("dst",), operands=("input",)),
        schedule_class=_SCHEDULE_CONVERT,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _cmp_descriptor(
    stem: str,
    operand_type: str,
    *,
    encoding_id: int,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.cmp.{stem}.{operand_type}",
        mnemonic=f"ireevm.cmp.{stem}.{operand_type}",
        semantic_tag=f"cmp.{stem}.{operand_type}",
        operands=(
            _result("i32"),
            _operand(operand_type, "lhs"),
            _operand(operand_type, "rhs"),
        ),
        asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
        schedule_class=_SCHEDULE_ALU,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _cmp_nz_descriptor(type_name: str, *, encoding_id: int) -> Descriptor:
    return Descriptor(
        key=f"ireevm.cmp.nz.{type_name}",
        mnemonic=f"ireevm.cmp.nz.{type_name}",
        semantic_tag=f"cmp.nz.{type_name}",
        operands=(_result("i32"), _operand(type_name, "input")),
        asm_forms=_asm(results=("dst",), operands=("input",)),
        schedule_class=_SCHEDULE_ALU,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _cmp_nan_descriptor(type_name: str, *, encoding_id: int) -> Descriptor:
    return Descriptor(
        key=f"ireevm.cmp.nan.{type_name}",
        mnemonic=f"ireevm.cmp.nan.{type_name}",
        semantic_tag=f"cmp.nan.{type_name}",
        operands=(_result("i32"), _operand(type_name, "input")),
        asm_forms=_asm(results=("dst",), operands=("input",)),
        schedule_class=_SCHEDULE_ALU,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _select_descriptor(type_name: str, *, encoding_id: int) -> Descriptor:
    return Descriptor(
        key=f"ireevm.select.{type_name}",
        mnemonic=f"ireevm.select.{type_name}",
        semantic_tag=f"select.{type_name}",
        operands=(
            _result(type_name),
            _predicate_i32(),
            _operand(type_name, "true_value"),
            _operand(type_name, "false_value"),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("condition", "true_value", "false_value"),
        ),
        schedule_class=_SCHEDULE_ALU,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _buffer_length_descriptor() -> Descriptor:
    return Descriptor(
        key="ireevm.buffer.length",
        mnemonic="ireevm.buffer.length",
        semantic_tag="buffer.length",
        operands=(_result("i64"), _operand("ref", "buffer")),
        asm_forms=_asm(results=("dst",), operands=("buffer",)),
        effects=(_BUFFER_READ_EFFECT,),
        schedule_class=_SCHEDULE_BUFFER_LOAD,
        feature_mask_words=_feature_mask_words(_OP_BUFFER_LENGTH),
        encoding_id=_OP_BUFFER_LENGTH,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_descriptor(
    suffix: str,
    result_type: str,
    *,
    encoding_id: int,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.buffer.load.{suffix}",
        mnemonic=f"ireevm.buffer.load.{suffix}",
        semantic_tag=f"buffer.load.{suffix}",
        operands=(
            _result(result_type),
            _operand("ref", "buffer"),
            _operand("i64", "element_offset"),
        ),
        asm_forms=_asm(results=("dst",), operands=("buffer", "element_offset")),
        effects=(_BUFFER_READ_EFFECT,),
        schedule_class=_SCHEDULE_BUFFER_LOAD,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_descriptor(
    suffix: str,
    value_type: str,
    *,
    encoding_id: int,
) -> Descriptor:
    return Descriptor(
        key=f"ireevm.buffer.store.{suffix}",
        mnemonic=f"ireevm.buffer.store.{suffix}",
        semantic_tag=f"buffer.store.{suffix}",
        operands=(
            _operand("ref", "buffer"),
            _operand("i64", "element_offset"),
            _operand(value_type, "value"),
        ),
        asm_forms=_asm(operands=("value", "buffer", "element_offset")),
        effects=(_BUFFER_WRITE_EFFECT,),
        schedule_class=_SCHEDULE_BUFFER_STORE,
        feature_mask_words=_feature_mask_words(encoding_id),
        encoding_id=encoding_id,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


_CONST_DESCRIPTORS = (
    _const_descriptor("i32", immediate=_I32_VALUE_IMMEDIATE, encoding_id=_OP_CONST_I32),
    _const_descriptor("i64", immediate=_I64_VALUE_IMMEDIATE, encoding_id=_OP_CONST_I64),
    _const_descriptor("f32", immediate=_F32_BITS_IMMEDIATE, encoding_id=_OP_CONST_F32),
    _const_descriptor("f64", immediate=_F64_BITS_IMMEDIATE, encoding_id=_OP_CONST_F64),
)

_SELECT_DESCRIPTORS = (
    _select_descriptor("i32", encoding_id=_OP_SELECT_I32),
    _select_descriptor("i64", encoding_id=_OP_SELECT_I64),
    _select_descriptor("f32", encoding_id=_OP_SELECT_F32),
    _select_descriptor("f64", encoding_id=_OP_SELECT_F64),
)

_INTEGER_DESCRIPTORS = (
    _binary_descriptor("add", "i32", encoding_id=_OP_ADD_I32),
    _binary_descriptor("sub", "i32", encoding_id=_OP_SUB_I32),
    _binary_descriptor("mul", "i32", encoding_id=_OP_MUL_I32),
    _binary_descriptor("div.s", "i32", encoding_id=_OP_DIV_S_I32),
    _binary_descriptor("div.u", "i32", encoding_id=_OP_DIV_U_I32),
    _binary_descriptor("rem.s", "i32", encoding_id=_OP_REM_S_I32),
    _binary_descriptor("rem.u", "i32", encoding_id=_OP_REM_U_I32),
    _ternary_descriptor("fma", "i32", encoding_id=_OP_FMA_I32),
    _unary_descriptor("abs", "i32", encoding_id=_OP_ABS_I32),
    _binary_descriptor("min.s", "i32", encoding_id=_OP_MIN_S_I32),
    _binary_descriptor("min.u", "i32", encoding_id=_OP_MIN_U_I32),
    _binary_descriptor("max.s", "i32", encoding_id=_OP_MAX_S_I32),
    _binary_descriptor("max.u", "i32", encoding_id=_OP_MAX_U_I32),
    _binary_descriptor("add", "i64", encoding_id=_OP_ADD_I64),
    _binary_descriptor("sub", "i64", encoding_id=_OP_SUB_I64),
    _binary_descriptor("mul", "i64", encoding_id=_OP_MUL_I64),
    _binary_descriptor("div.s", "i64", encoding_id=_OP_DIV_S_I64),
    _binary_descriptor("div.u", "i64", encoding_id=_OP_DIV_U_I64),
    _binary_descriptor("rem.s", "i64", encoding_id=_OP_REM_S_I64),
    _binary_descriptor("rem.u", "i64", encoding_id=_OP_REM_U_I64),
    _ternary_descriptor("fma", "i64", encoding_id=_OP_FMA_I64),
    _unary_descriptor("abs", "i64", encoding_id=_OP_ABS_I64),
    _binary_descriptor("min.s", "i64", encoding_id=_OP_MIN_S_I64),
    _binary_descriptor("min.u", "i64", encoding_id=_OP_MIN_U_I64),
    _binary_descriptor("max.s", "i64", encoding_id=_OP_MAX_S_I64),
    _binary_descriptor("max.u", "i64", encoding_id=_OP_MAX_U_I64),
)

_BITWISE_DESCRIPTORS = (
    _unary_descriptor("not", "i32", encoding_id=_OP_NOT_I32),
    _binary_descriptor("and", "i32", encoding_id=_OP_AND_I32),
    _binary_descriptor("or", "i32", encoding_id=_OP_OR_I32),
    _binary_descriptor("xor", "i32", encoding_id=_OP_XOR_I32),
    _shift_descriptor("shl", "i32", encoding_id=_OP_SHL_I32),
    _shift_descriptor("shr.s", "i32", encoding_id=_OP_SHR_S_I32),
    _shift_descriptor("shr.u", "i32", encoding_id=_OP_SHR_U_I32),
    _unary_descriptor("ctlz", "i32", encoding_id=_OP_CTLZ_I32),
    _unary_descriptor("not", "i64", encoding_id=_OP_NOT_I64),
    _binary_descriptor("and", "i64", encoding_id=_OP_AND_I64),
    _binary_descriptor("or", "i64", encoding_id=_OP_OR_I64),
    _binary_descriptor("xor", "i64", encoding_id=_OP_XOR_I64),
    _shift_descriptor("shl", "i64", encoding_id=_OP_SHL_I64),
    _shift_descriptor("shr.s", "i64", encoding_id=_OP_SHR_S_I64),
    _shift_descriptor("shr.u", "i64", encoding_id=_OP_SHR_U_I64),
    _unary_descriptor("ctlz", "i64", encoding_id=_OP_CTLZ_I64),
)

_INTEGER_CONVERSION_DESCRIPTORS = (
    _convert_descriptor("trunc", "i32", "i32", encoding_id=_OP_TRUNC_I32_I8),
    _convert_descriptor("trunc.i16", "i32", "i32", encoding_id=_OP_TRUNC_I32_I16),
    _convert_descriptor("trunc", "i64", "i32", encoding_id=_OP_TRUNC_I64_I32),
    _convert_descriptor("ext.s.i8", "i32", "i32", encoding_id=_OP_EXT_I8_I32_S),
    _convert_descriptor("ext.u.i8", "i32", "i32", encoding_id=_OP_EXT_I8_I32_U),
    _convert_descriptor("ext.s.i16", "i32", "i32", encoding_id=_OP_EXT_I16_I32_S),
    _convert_descriptor("ext.u.i16", "i32", "i32", encoding_id=_OP_EXT_I16_I32_U),
    _convert_descriptor("ext.s", "i32", "i64", encoding_id=_OP_EXT_I32_I64_S),
    _convert_descriptor("ext.u", "i32", "i64", encoding_id=_OP_EXT_I32_I64_U),
)

_INTEGER_COMPARE_DESCRIPTORS = (
    _cmp_descriptor("eq", "i32", encoding_id=_OP_CMP_EQ_I32),
    _cmp_descriptor("ne", "i32", encoding_id=_OP_CMP_NE_I32),
    _cmp_descriptor("lt.s", "i32", encoding_id=_OP_CMP_LT_S_I32),
    _cmp_descriptor("lt.u", "i32", encoding_id=_OP_CMP_LT_U_I32),
    _cmp_nz_descriptor("i32", encoding_id=_OP_CMP_NZ_I32),
    _cmp_descriptor("eq", "i64", encoding_id=_OP_CMP_EQ_I64),
    _cmp_descriptor("ne", "i64", encoding_id=_OP_CMP_NE_I64),
    _cmp_descriptor("lt.s", "i64", encoding_id=_OP_CMP_LT_S_I64),
    _cmp_descriptor("lt.u", "i64", encoding_id=_OP_CMP_LT_U_I64),
    _cmp_nz_descriptor("i64", encoding_id=_OP_CMP_NZ_I64),
)

_FLOAT_ARITHMETIC_DESCRIPTORS = (
    _binary_descriptor("add", "f32", encoding_id=_OP_ADD_F32),
    _binary_descriptor("sub", "f32", encoding_id=_OP_SUB_F32),
    _binary_descriptor("mul", "f32", encoding_id=_OP_MUL_F32),
    _binary_descriptor("div", "f32", encoding_id=_OP_DIV_F32),
    _binary_descriptor("rem", "f32", encoding_id=_OP_REM_F32),
    _ternary_descriptor("fma", "f32", encoding_id=_OP_FMA_F32),
    _unary_descriptor("abs", "f32", encoding_id=_OP_ABS_F32),
    _unary_descriptor("neg", "f32", encoding_id=_OP_NEG_F32),
    _unary_descriptor("ceil", "f32", encoding_id=_OP_CEIL_F32),
    _unary_descriptor("floor", "f32", encoding_id=_OP_FLOOR_F32),
    _unary_descriptor("round", "f32", encoding_id=_OP_ROUND_F32),
    _unary_descriptor("roundeven", "f32", encoding_id=_OP_ROUND_EVEN_F32),
    _binary_descriptor("min", "f32", encoding_id=_OP_MIN_F32),
    _binary_descriptor("max", "f32", encoding_id=_OP_MAX_F32),
    _binary_descriptor("add", "f64", encoding_id=_OP_ADD_F64),
    _binary_descriptor("sub", "f64", encoding_id=_OP_SUB_F64),
    _binary_descriptor("mul", "f64", encoding_id=_OP_MUL_F64),
    _binary_descriptor("div", "f64", encoding_id=_OP_DIV_F64),
    _binary_descriptor("rem", "f64", encoding_id=_OP_REM_F64),
    _ternary_descriptor("fma", "f64", encoding_id=_OP_FMA_F64),
    _unary_descriptor("abs", "f64", encoding_id=_OP_ABS_F64),
    _unary_descriptor("neg", "f64", encoding_id=_OP_NEG_F64),
    _unary_descriptor("ceil", "f64", encoding_id=_OP_CEIL_F64),
    _unary_descriptor("floor", "f64", encoding_id=_OP_FLOOR_F64),
    _unary_descriptor("round", "f64", encoding_id=_OP_ROUND_F64),
    _unary_descriptor("roundeven", "f64", encoding_id=_OP_ROUND_EVEN_F64),
    _binary_descriptor("min", "f64", encoding_id=_OP_MIN_F64),
    _binary_descriptor("max", "f64", encoding_id=_OP_MAX_F64),
)

_FLOAT_MATH_DESCRIPTORS = (
    _unary_descriptor(
        "atan", "f32", encoding_id=_OP_ATAN_F32, schedule_class=_SCHEDULE_MATH
    ),
    _binary_descriptor(
        "atan2", "f32", encoding_id=_OP_ATAN2_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "cos", "f32", encoding_id=_OP_COS_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "sin", "f32", encoding_id=_OP_SIN_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "exp", "f32", encoding_id=_OP_EXP_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "exp2", "f32", encoding_id=_OP_EXP2_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "expm1", "f32", encoding_id=_OP_EXPM1_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log", "f32", encoding_id=_OP_LOG_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log10", "f32", encoding_id=_OP_LOG10_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log1p", "f32", encoding_id=_OP_LOG1P_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log2", "f32", encoding_id=_OP_LOG2_F32, schedule_class=_SCHEDULE_MATH
    ),
    _binary_descriptor(
        "pow", "f32", encoding_id=_OP_POW_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "rsqrt", "f32", encoding_id=_OP_RSQRT_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "sqrt", "f32", encoding_id=_OP_SQRT_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "tanh", "f32", encoding_id=_OP_TANH_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "erf", "f32", encoding_id=_OP_ERF_F32, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "atan", "f64", encoding_id=_OP_ATAN_F64, schedule_class=_SCHEDULE_MATH
    ),
    _binary_descriptor(
        "atan2", "f64", encoding_id=_OP_ATAN2_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "cos", "f64", encoding_id=_OP_COS_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "sin", "f64", encoding_id=_OP_SIN_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "exp", "f64", encoding_id=_OP_EXP_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "exp2", "f64", encoding_id=_OP_EXP2_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "expm1", "f64", encoding_id=_OP_EXPM1_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log", "f64", encoding_id=_OP_LOG_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log10", "f64", encoding_id=_OP_LOG10_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log1p", "f64", encoding_id=_OP_LOG1P_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "log2", "f64", encoding_id=_OP_LOG2_F64, schedule_class=_SCHEDULE_MATH
    ),
    _binary_descriptor(
        "pow", "f64", encoding_id=_OP_POW_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "rsqrt", "f64", encoding_id=_OP_RSQRT_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "sqrt", "f64", encoding_id=_OP_SQRT_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "tanh", "f64", encoding_id=_OP_TANH_F64, schedule_class=_SCHEDULE_MATH
    ),
    _unary_descriptor(
        "erf", "f64", encoding_id=_OP_ERF_F64, schedule_class=_SCHEDULE_MATH
    ),
)

_FLOAT_CONVERSION_DESCRIPTORS = (
    _convert_descriptor("trunc", "f64", "f32", encoding_id=_OP_TRUNC_F64_F32),
    _convert_descriptor("ext", "f32", "f64", encoding_id=_OP_EXT_F32_F64),
    _convert_descriptor("cast.s", "i32", "f32", encoding_id=_OP_CAST_SI32_F32),
    _convert_descriptor("cast.u", "i32", "f32", encoding_id=_OP_CAST_UI32_F32),
    _convert_descriptor("cast.s", "f32", "i32", encoding_id=_OP_CAST_F32_SI32),
    _convert_descriptor("cast.u", "f32", "i32", encoding_id=_OP_CAST_F32_UI32),
    _convert_descriptor("bitcast", "i32", "f32", encoding_id=_OP_BITCAST_I32_F32),
    _convert_descriptor("bitcast", "f32", "i32", encoding_id=_OP_BITCAST_F32_I32),
    _convert_descriptor("cast.s", "f32", "i64", encoding_id=_OP_CAST_F32_SI64),
    _convert_descriptor("cast.u", "f32", "i64", encoding_id=_OP_CAST_F32_UI64),
    _convert_descriptor("cast.s", "i64", "f32", encoding_id=_OP_CAST_SI64_F32),
    _convert_descriptor("cast.u", "i64", "f32", encoding_id=_OP_CAST_UI64_F32),
    _convert_descriptor("cast.s", "i32", "f64", encoding_id=_OP_CAST_SI32_F64),
    _convert_descriptor("cast.u", "i32", "f64", encoding_id=_OP_CAST_UI32_F64),
    _convert_descriptor("cast.s", "f64", "i32", encoding_id=_OP_CAST_F64_SI32),
    _convert_descriptor("cast.u", "f64", "i32", encoding_id=_OP_CAST_F64_UI32),
    _convert_descriptor("cast.s", "i64", "f64", encoding_id=_OP_CAST_SI64_F64),
    _convert_descriptor("cast.u", "i64", "f64", encoding_id=_OP_CAST_UI64_F64),
    _convert_descriptor("cast.s", "f64", "i64", encoding_id=_OP_CAST_F64_SI64),
    _convert_descriptor("cast.u", "f64", "i64", encoding_id=_OP_CAST_F64_UI64),
    _convert_descriptor("bitcast", "i64", "f64", encoding_id=_OP_BITCAST_I64_F64),
    _convert_descriptor("bitcast", "f64", "i64", encoding_id=_OP_BITCAST_F64_I64),
)

_FLOAT_COMPARE_DESCRIPTORS = (
    _cmp_descriptor("eq.o", "f32", encoding_id=_OP_CMP_EQ_O_F32),
    _cmp_descriptor("eq.u", "f32", encoding_id=_OP_CMP_EQ_U_F32),
    _cmp_descriptor("ne.o", "f32", encoding_id=_OP_CMP_NE_O_F32),
    _cmp_descriptor("ne.u", "f32", encoding_id=_OP_CMP_NE_U_F32),
    _cmp_descriptor("lt.o", "f32", encoding_id=_OP_CMP_LT_O_F32),
    _cmp_descriptor("lt.u", "f32", encoding_id=_OP_CMP_LT_U_F32),
    _cmp_descriptor("le.o", "f32", encoding_id=_OP_CMP_LE_O_F32),
    _cmp_descriptor("le.u", "f32", encoding_id=_OP_CMP_LE_U_F32),
    _cmp_nan_descriptor("f32", encoding_id=_OP_CMP_NAN_F32),
    _cmp_descriptor("eq.o", "f64", encoding_id=_OP_CMP_EQ_O_F64),
    _cmp_descriptor("eq.u", "f64", encoding_id=_OP_CMP_EQ_U_F64),
    _cmp_descriptor("ne.o", "f64", encoding_id=_OP_CMP_NE_O_F64),
    _cmp_descriptor("ne.u", "f64", encoding_id=_OP_CMP_NE_U_F64),
    _cmp_descriptor("lt.o", "f64", encoding_id=_OP_CMP_LT_O_F64),
    _cmp_descriptor("lt.u", "f64", encoding_id=_OP_CMP_LT_U_F64),
    _cmp_descriptor("le.o", "f64", encoding_id=_OP_CMP_LE_O_F64),
    _cmp_descriptor("le.u", "f64", encoding_id=_OP_CMP_LE_U_F64),
    _cmp_nan_descriptor("f64", encoding_id=_OP_CMP_NAN_F64),
)

_REF_DESCRIPTORS = (
    Descriptor(
        key="ireevm.ref.retain",
        mnemonic="ireevm.ref.retain",
        semantic_tag="ref.retain",
        operands=(_result("ref"), _operand("ref", "resource")),
        asm_forms=_asm(results=("dst",), operands=("resource",)),
        effects=(_REF_READ_EFFECT, _REF_WRITE_EFFECT),
        schedule_class=_SCHEDULE_REF,
        encoding_id=_OP_ASSIGN_REF,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="ireevm.ref.release",
        mnemonic="ireevm.ref.release",
        semantic_tag="ref.release",
        operands=(_operand("ref", "resource"),),
        asm_forms=_asm(operands=("resource",)),
        effects=(_REF_WRITE_EFFECT,),
        schedule_class=_SCHEDULE_REF,
        encoding_id=_OP_DISCARD_REFS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
)

_BUFFER_DESCRIPTORS = (
    _buffer_length_descriptor(),
    _buffer_load_descriptor("i8.u", "i32", encoding_id=_OP_BUFFER_LOAD_I8_U),
    _buffer_load_descriptor("i8.s", "i32", encoding_id=_OP_BUFFER_LOAD_I8_S),
    _buffer_load_descriptor("i16.u", "i32", encoding_id=_OP_BUFFER_LOAD_I16_U),
    _buffer_load_descriptor("i16.s", "i32", encoding_id=_OP_BUFFER_LOAD_I16_S),
    _buffer_load_descriptor("i32", "i32", encoding_id=_OP_BUFFER_LOAD_I32),
    _buffer_load_descriptor("i64", "i64", encoding_id=_OP_BUFFER_LOAD_I64),
    _buffer_load_descriptor("f32", "f32", encoding_id=_OP_BUFFER_LOAD_F32),
    _buffer_load_descriptor("f64", "f64", encoding_id=_OP_BUFFER_LOAD_F64),
    _buffer_store_descriptor("i8", "i32", encoding_id=_OP_BUFFER_STORE_I8),
    _buffer_store_descriptor("i16", "i32", encoding_id=_OP_BUFFER_STORE_I16),
    _buffer_store_descriptor("i32", "i32", encoding_id=_OP_BUFFER_STORE_I32),
    _buffer_store_descriptor("i64", "i64", encoding_id=_OP_BUFFER_STORE_I64),
    _buffer_store_descriptor("f32", "f32", encoding_id=_OP_BUFFER_STORE_F32),
    _buffer_store_descriptor("f64", "f64", encoding_id=_OP_BUFFER_STORE_F64),
)

_CONTROL_DESCRIPTORS = (
    Descriptor(
        key="ireevm.br",
        mnemonic="ireevm.br",
        semantic_tag="control.branch",
        operands=(),
        immediates=(_TARGET_BLOCK_IMMEDIATE,),
        asm_forms=_asm(immediates=("target_block",)),
        effects=(_CONTROL_EFFECT,),
        schedule_class=_SCHEDULE_CONTROL,
        encoding_id=_OP_BRANCH,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
    ),
    Descriptor(
        key="ireevm.cond_br.i32",
        mnemonic="ireevm.cond_br.i32",
        semantic_tag="control.cond_branch.i32",
        operands=(_predicate_i32("condition"),),
        immediates=(_TRUE_BLOCK_IMMEDIATE, _FALSE_BLOCK_IMMEDIATE),
        asm_forms=_asm(
            operands=("condition",), immediates=("true_block", "false_block")
        ),
        effects=(_CONTROL_EFFECT,),
        schedule_class=_SCHEDULE_CONTROL,
        encoding_id=_OP_COND_BRANCH,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
    ),
    Descriptor(
        key="ireevm.return",
        mnemonic="ireevm.return",
        semantic_tag="control.return",
        operands=(),
        asm_forms=_asm(),
        effects=(_CONTROL_EFFECT,),
        schedule_class=_SCHEDULE_CONTROL,
        encoding_id=_OP_RETURN,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
    ),
)

IREEVM_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="ireevm.core",
    target_key="ireevm",
    feature_key="ireevm.v1",
    c_header_path=Path("loom/src/loom/target/arch/ireevm/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/ireevm/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_IREEVM_DESCRIPTORS_H_",
    public_header="loom/target/arch/ireevm/descriptors/descriptors.h",
    function_name="loom_ireevm_core_descriptor_set",
    c_table_prefix="IreeVmCore",
    c_enum_prefix="IREEVM_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I32,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            target_bank_id=_SCALAR_TARGET_BANK,
            alias_set_id=_SCALAR_ALIAS_SET,
        ),
        RegClass(
            _REG_I64,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            target_bank_id=_SCALAR_TARGET_BANK,
            alias_set_id=_SCALAR_ALIAS_SET,
        ),
        RegClass(
            _REG_F32,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            target_bank_id=_SCALAR_TARGET_BANK,
            alias_set_id=_SCALAR_ALIAS_SET,
        ),
        RegClass(
            _REG_F64,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            target_bank_id=_SCALAR_TARGET_BANK,
            alias_set_id=_SCALAR_ALIAS_SET,
        ),
        RegClass(
            _REG_REF,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.REFERENCE),
            target_bank_id=_REF_TARGET_BANK,
            alias_set_id=_REF_ALIAS_SET,
        ),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_MATH, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_REF, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_BUFFER_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_BUFFER_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_MATH,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_MATH, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CONVERT,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_REF,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_REF, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_BUFFER_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_BUFFER_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_BUFFER_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_BUFFER_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
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
        *_CONST_DESCRIPTORS,
        *_SELECT_DESCRIPTORS,
        *_INTEGER_DESCRIPTORS,
        *_BITWISE_DESCRIPTORS,
        *_INTEGER_CONVERSION_DESCRIPTORS,
        *_INTEGER_COMPARE_DESCRIPTORS,
        *_FLOAT_ARITHMETIC_DESCRIPTORS,
        *_FLOAT_MATH_DESCRIPTORS,
        *_FLOAT_CONVERSION_DESCRIPTORS,
        *_FLOAT_COMPARE_DESCRIPTORS,
        *_REF_DESCRIPTORS,
        *_BUFFER_DESCRIPTORS,
        *_CONTROL_DESCRIPTORS,
    ),
)
