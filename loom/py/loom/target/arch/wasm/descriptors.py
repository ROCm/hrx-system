# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the Wasm core+SIMD128 target."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
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

_REG_I32 = "wasm.i32"
_REG_I64 = "wasm.i64"
_REG_F32 = "wasm.f32"
_REG_F64 = "wasm.f64"
_REG_V128 = "wasm.v128"

_RESOURCE_SCALAR = "wasm.scalar"
_RESOURCE_SIMD = "wasm.simd"
_RESOURCE_LOAD = "wasm.load"
_RESOURCE_STORE = "wasm.store"
_RESOURCE_CONTROL = "wasm.control"

_SCHEDULE_CONST = "wasm.const"
_SCHEDULE_SCALAR_I32 = "wasm.scalar.i32"
_SCHEDULE_SCALAR_I64 = "wasm.scalar.i64"
_SCHEDULE_SCALAR_F32 = "wasm.scalar.f32"
_SCHEDULE_SCALAR_F64 = "wasm.scalar.f64"
_SCHEDULE_SIMD_I32X4 = "wasm.simd.i32x4"
_SCHEDULE_SIMD_F32X4 = "wasm.simd.f32x4"
_SCHEDULE_SIMD_I64X2 = "wasm.simd.i64x2"
_SCHEDULE_SIMD_F64X2 = "wasm.simd.f64x2"
_SCHEDULE_MEMORY_LOAD = "wasm.memory.load"
_SCHEDULE_MEMORY_STORE = "wasm.memory.store"
_SCHEDULE_CONTROL = "wasm.control"

_I32_ALT = (RegClassAlt(_REG_I32),)
_I64_ALT = (RegClassAlt(_REG_I64),)
_F32_ALT = (RegClassAlt(_REG_F32),)
_F64_ALT = (RegClassAlt(_REG_F64),)
_V128_ALT = (RegClassAlt(_REG_V128),)


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


def _i32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I32_ALT)


def _i32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I32_ALT)


def _i64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I64_ALT)


def _i64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I64_ALT)


def _i32_predicate(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _I32_ALT)


def _i32_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _I32_ALT)


def _f32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _F32_ALT)


def _f32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _F32_ALT)


def _f64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _F64_ALT)


def _f64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _F64_ALT)


def _v128_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _V128_ALT)


def _v128_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _V128_ALT)


def _v128_predicate(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _V128_ALT)


_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**32) - 1,
)

_I64_VALUE_IMMEDIATE = Immediate(
    "i64_value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_V128_LO_IMMEDIATE = Immediate(
    "lo64",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_V128_HI_IMMEDIATE = Immediate(
    "hi64",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_LANE_I32X4_IMMEDIATE = Immediate(
    "lane",
    ImmediateKind.UNSIGNED,
    bit_width=2,
    unsigned_max=3,
)

_LANE_I64X2_IMMEDIATE = Immediate(
    "lane",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_SHUFFLE_BYTE_IMMEDIATES = tuple(
    Immediate(
        f"lane{i}",
        ImmediateKind.UNSIGNED,
        bit_width=5,
        unsigned_max=31,
    )
    for i in range(16)
)

_MEMORY_OFFSET_IMMEDIATE = Immediate(
    "offset",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
    default_value=0,
)

_OP_BR = 0x0C
_OP_BR_IF = 0x0D
_OP_RETURN = 0x0F
_OP_SELECT = 0x1B
_OP_I32_LOAD = 0x28
_OP_I64_LOAD = 0x29
_OP_F32_LOAD = 0x2A
_OP_F64_LOAD = 0x2B
_OP_I32_LOAD8_U = 0x2D
_OP_I32_LOAD16_U = 0x2F
_OP_I32_STORE = 0x36
_OP_I64_STORE = 0x37
_OP_F32_STORE = 0x38
_OP_F64_STORE = 0x39
_OP_I32_STORE8 = 0x3A
_OP_I32_STORE16 = 0x3B
_OP_I32_CONST = 0x41
_OP_I64_CONST = 0x42
_OP_F32_CONST = 0x43
_OP_F64_CONST = 0x44
_OP_I32_EQ = 0x46
_OP_I32_NE = 0x47
_OP_I32_LT_S = 0x48
_OP_I32_LT_U = 0x49
_OP_I32_GT_S = 0x4A
_OP_I32_GT_U = 0x4B
_OP_I32_LE_S = 0x4C
_OP_I32_LE_U = 0x4D
_OP_I32_GE_S = 0x4E
_OP_I32_GE_U = 0x4F
_OP_I64_EQ = 0x51
_OP_I64_NE = 0x52
_OP_I64_LT_S = 0x53
_OP_I64_LT_U = 0x54
_OP_I64_GT_S = 0x55
_OP_I64_GT_U = 0x56
_OP_I64_LE_S = 0x57
_OP_I64_LE_U = 0x58
_OP_I64_GE_S = 0x59
_OP_I64_GE_U = 0x5A
_OP_I32_ADD = 0x6A
_OP_I32_SUB = 0x6B
_OP_I32_MUL = 0x6C
_OP_I32_REM_U = 0x70
_OP_I32_AND = 0x71
_OP_I32_OR = 0x72
_OP_I32_XOR = 0x73
_OP_I32_SHL = 0x74
_OP_I32_SHR_S = 0x75
_OP_I32_SHR_U = 0x76
_OP_I64_ADD = 0x7C
_OP_I64_SUB = 0x7D
_OP_I64_MUL = 0x7E
_OP_I64_REM_U = 0x82
_OP_I64_AND = 0x83
_OP_I64_OR = 0x84
_OP_I64_XOR = 0x85
_OP_I64_SHL = 0x86
_OP_I64_SHR_S = 0x87
_OP_I64_SHR_U = 0x88
_OP_F32_ADD = 0x92
_OP_I32_WRAP_I64 = 0xA7
_OP_I64_EXTEND_I32_S = 0xAC
_OP_I64_EXTEND_I32_U = 0xAD
_OP_I32_REINTERPRET_F32 = 0xBC
_OP_I64_REINTERPRET_F64 = 0xBD
_OP_F32_REINTERPRET_I32 = 0xBE
_OP_F64_REINTERPRET_I64 = 0xBF
_OP_SIMD_PREFIX = 0xFD


def _simd_encoding_id(subopcode: int) -> int:
    return (_OP_SIMD_PREFIX << 8) | subopcode


_OP_V128_LOAD = _simd_encoding_id(0x00)
_OP_V128_STORE = _simd_encoding_id(0x0B)
_OP_V128_CONST = _simd_encoding_id(0x0C)
_OP_I8X16_SHUFFLE = _simd_encoding_id(0x0D)
_OP_I32X4_SPLAT = _simd_encoding_id(0x11)
_OP_I64X2_SPLAT = _simd_encoding_id(0x12)
_OP_F32X4_SPLAT = _simd_encoding_id(0x13)
_OP_F64X2_SPLAT = _simd_encoding_id(0x14)
_OP_I32X4_EXTRACT_LANE = _simd_encoding_id(0x1B)
_OP_I32X4_REPLACE_LANE = _simd_encoding_id(0x1C)
_OP_I64X2_EXTRACT_LANE = _simd_encoding_id(0x1D)
_OP_I64X2_REPLACE_LANE = _simd_encoding_id(0x1E)
_OP_F32X4_EXTRACT_LANE = _simd_encoding_id(0x1F)
_OP_F32X4_REPLACE_LANE = _simd_encoding_id(0x20)
_OP_F64X2_EXTRACT_LANE = _simd_encoding_id(0x21)
_OP_F64X2_REPLACE_LANE = _simd_encoding_id(0x22)
_OP_I32X4_EQ = _simd_encoding_id(0x37)
_OP_I32X4_NE = _simd_encoding_id(0x38)
_OP_I32X4_LT_S = _simd_encoding_id(0x39)
_OP_I32X4_LT_U = _simd_encoding_id(0x3A)
_OP_I32X4_GT_S = _simd_encoding_id(0x3B)
_OP_I32X4_GT_U = _simd_encoding_id(0x3C)
_OP_I32X4_LE_S = _simd_encoding_id(0x3D)
_OP_I32X4_LE_U = _simd_encoding_id(0x3E)
_OP_I32X4_GE_S = _simd_encoding_id(0x3F)
_OP_I32X4_GE_U = _simd_encoding_id(0x40)
_OP_F32X4_EQ = _simd_encoding_id(0x41)
_OP_F32X4_LT = _simd_encoding_id(0x43)
_OP_F32X4_GT = _simd_encoding_id(0x44)
_OP_F32X4_LE = _simd_encoding_id(0x45)
_OP_F32X4_GE = _simd_encoding_id(0x46)
_OP_V128_BITSELECT = _simd_encoding_id(0x52)
_OP_I32X4_ADD = _simd_encoding_id(0xAE)
_OP_I32X4_SUB = _simd_encoding_id(0xB1)
_OP_I32X4_MUL = _simd_encoding_id(0xB5)
_OP_F32X4_ADD = _simd_encoding_id(0xE4)
_OP_F32X4_MUL = _simd_encoding_id(0xE6)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.WASM_MEMORY,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.WASM_MEMORY,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)


def _scalar_binary_descriptor(
    type_name: str,
    operation: str,
    semantic_tag: str,
    encoding_id: int,
    *,
    comparison: bool = False,
) -> Descriptor:
    register_class = _REG_I32 if type_name == "i32" else _REG_I64
    result_class = _REG_I32 if comparison else register_class
    return Descriptor(
        key=f"wasm.{type_name}.{operation}",
        mnemonic=f"{type_name}.{operation}",
        semantic_tag=semantic_tag,
        encoding_id=encoding_id,
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt(result_class),)),
            Operand("lhs", OperandRole.OPERAND, (RegClassAlt(register_class),)),
            Operand("rhs", OperandRole.OPERAND, (RegClassAlt(register_class),)),
        ),
        asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
        schedule_class=_SCHEDULE_SCALAR_I32
        if type_name == "i32"
        else _SCHEDULE_SCALAR_I64,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _scalar_memory_descriptors(
    type_name: str,
    register_class: str,
    width_bits: int,
    load_opcode: int,
    store_opcode: int,
    *,
    load_suffix: str = "",
    store_suffix: str = "",
) -> tuple[Descriptor, ...]:
    return (
        Descriptor(
            key=f"wasm.{type_name}.load{load_suffix}",
            mnemonic=f"{type_name}.load{load_suffix}",
            semantic_tag=f"memory.load{load_suffix}.{type_name}",
            encoding_id=load_opcode,
            operands=(
                Operand("dst", OperandRole.RESULT, (RegClassAlt(register_class),)),
                _i32_resource("address"),
            ),
            immediates=(_MEMORY_OFFSET_IMMEDIATE,),
            asm_forms=_asm(
                results=("dst",), operands=("address",), immediates=("offset",)
            ),
            effects=(
                Effect(
                    EffectKind.READ,
                    memory_space=MemorySpace.WASM_MEMORY,
                    flags=(EffectFlag.DEPENDENCY,),
                    width_bits=width_bits,
                ),
            ),
            schedule_class=_SCHEDULE_MEMORY_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key=f"wasm.{type_name}.store{store_suffix}",
            mnemonic=f"{type_name}.store{store_suffix}",
            semantic_tag=f"memory.store{store_suffix}.{type_name}",
            encoding_id=store_opcode,
            operands=(
                _i32_resource("address"),
                Operand("value", OperandRole.OPERAND, (RegClassAlt(register_class),)),
            ),
            immediates=(_MEMORY_OFFSET_IMMEDIATE,),
            asm_forms=_asm(operands=("address", "value"), immediates=("offset",)),
            effects=(
                Effect(
                    EffectKind.WRITE,
                    memory_space=MemorySpace.WASM_MEMORY,
                    flags=(EffectFlag.DEPENDENCY,),
                    width_bits=width_bits,
                ),
            ),
            schedule_class=_SCHEDULE_MEMORY_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    )


def _scalar_select_descriptor(
    type_name: str, register_class: str, schedule: str
) -> Descriptor:
    category = "float" if type_name.startswith("f") else "integer"
    return Descriptor(
        key=f"wasm.{type_name}.select",
        mnemonic=f"{type_name}.select",
        semantic_tag=f"{category}.select.{type_name}",
        encoding_id=_OP_SELECT,
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt(register_class),)),
            Operand("true_value", OperandRole.OPERAND, (RegClassAlt(register_class),)),
            Operand("false_value", OperandRole.OPERAND, (RegClassAlt(register_class),)),
            _i32_operand("condition"),
        ),
        asm_forms=_asm(
            results=("dst",), operands=("true_value", "false_value", "condition")
        ),
        schedule_class=schedule,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _float_const_descriptor(
    type_name: str, result: Operand, bit_width: int, encoding_id: int
) -> Descriptor:
    return Descriptor(
        key=f"wasm.{type_name}.const",
        mnemonic=f"{type_name}.const",
        semantic_tag=f"float.const.{type_name}",
        encoding_id=encoding_id,
        operands=(result,),
        op_kind=DescriptorOpKind.CONST,
        immediates=(
            Immediate(
                "bits",
                ImmediateKind.UNSIGNED,
                bit_width=bit_width,
                unsigned_max=(1 << bit_width) - 1,
            ),
        ),
        asm_forms=_asm(results=("dst",), immediates=("bits",)),
        schedule_class=_SCHEDULE_CONST,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _lane_descriptors(
    type_name: str,
    result: Operand,
    value: Operand,
    lane: Immediate,
    schedule: str,
    splat: int,
    extract: int,
    insert: int,
) -> tuple[Descriptor, ...]:
    return (
        Descriptor(
            key=f"wasm.{type_name}.splat",
            mnemonic=f"{type_name}.splat",
            semantic_tag=f"vector.splat.{type_name}",
            encoding_id=splat,
            operands=(_v128_result(), value),
            asm_forms=_asm(results=("dst",), operands=("value",)),
            schedule_class=schedule,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key=f"wasm.{type_name}.extract_lane",
            mnemonic=f"{type_name}.extract_lane",
            semantic_tag=f"vector.extract.{type_name}",
            encoding_id=extract,
            operands=(result, _v128_operand("source")),
            immediates=(lane,),
            asm_forms=_asm(
                results=("dst",), operands=("source",), immediates=("lane",)
            ),
            schedule_class=schedule,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key=f"wasm.{type_name}.replace_lane",
            mnemonic=f"{type_name}.replace_lane",
            semantic_tag=f"vector.insert.{type_name}",
            encoding_id=insert,
            operands=(_v128_result(), _v128_operand("dest"), value),
            immediates=(lane,),
            asm_forms=_asm(
                results=("dst",), operands=("dest", "value"), immediates=("lane",)
            ),
            schedule_class=schedule,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    )


WASM_CORE_SIMD128_DESCRIPTOR_SET = DescriptorSet(
    key="wasm.core.simd128",
    requires_structured_control_flow=True,
    target_key="wasm",
    feature_key="wasm.simd128.v1",
    c_header_path=Path("loom/src/loom/target/arch/wasm/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/wasm/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_WASM_DESCRIPTORS_H_",
    public_header="loom/target/arch/wasm/descriptors/descriptors.h",
    function_name="loom_wasm_core_simd128_descriptor_set",
    c_table_prefix="WasmCoreSimd128",
    c_enum_prefix="WASM_CORE_SIMD128",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_V128, 128, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_SIMD, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
            instruction_classes=(InstructionClass.OTHER,),
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_I32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_I64,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_F32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SIMD_I32X4,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SIMD, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SIMD_F32X4,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SIMD, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        *(
            ScheduleClass(
                key,
                latency_kind=LatencyKind.ESTIMATE,
                latency_cycles=1,
                issue_uses=(IssueUse(resource, cycles=1, units=1),),
                model_quality=ModelQuality.ESTIMATED,
            )
            for key, resource in (
                (_SCHEDULE_SCALAR_F64, _RESOURCE_SCALAR),
                (_SCHEDULE_SIMD_I64X2, _RESOURCE_SIMD),
                (_SCHEDULE_SIMD_F64X2, _RESOURCE_SIMD),
            )
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
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
        *(
            _scalar_select_descriptor(type_name, register_class, schedule)
            for type_name, register_class, schedule in (
                ("i32", _REG_I32, _SCHEDULE_SCALAR_I32),
                ("i64", _REG_I64, _SCHEDULE_SCALAR_I64),
                ("f32", _REG_F32, _SCHEDULE_SCALAR_F32),
                ("f64", _REG_F64, _SCHEDULE_SCALAR_F64),
            )
        ),
        _float_const_descriptor("f32", _f32_result(), 32, _OP_F32_CONST),
        _float_const_descriptor("f64", _f64_result(), 64, _OP_F64_CONST),
        Descriptor(
            key="wasm.i32.const",
            mnemonic="i32.const",
            semantic_tag="integer.const.i32",
            encoding_id=_OP_I32_CONST,
            operands=(_i32_result(),),
            op_kind=DescriptorOpKind.CONST,
            immediates=(_I32_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i64.const",
            mnemonic="i64.const",
            semantic_tag="integer.const.i64",
            encoding_id=_OP_I64_CONST,
            operands=(_i64_result(),),
            op_kind=DescriptorOpKind.CONST,
            immediates=(_I64_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("i64_value",)),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *(
            _scalar_binary_descriptor(
                "i32", operation, f"integer.{semantic}.i32", encoding
            )
            for operation, semantic, encoding in (
                ("add", "add", _OP_I32_ADD),
                ("sub", "sub", _OP_I32_SUB),
                ("mul", "mul", _OP_I32_MUL),
                ("rem_u", "remu", _OP_I32_REM_U),
                ("and", "and", _OP_I32_AND),
                ("or", "or", _OP_I32_OR),
                ("xor", "xor", _OP_I32_XOR),
                ("shl", "shl", _OP_I32_SHL),
                ("shr_s", "shrs", _OP_I32_SHR_S),
                ("shr_u", "shru", _OP_I32_SHR_U),
            )
        ),
        *(
            _scalar_binary_descriptor(
                "i32", operation, f"integer.cmp.{semantic}", encoding, comparison=True
            )
            for operation, semantic, encoding in (
                ("eq", "eq.i32", _OP_I32_EQ),
                ("ne", "ne.i32", _OP_I32_NE),
                ("lt_s", "lt.s32", _OP_I32_LT_S),
                ("lt_u", "lt.u32", _OP_I32_LT_U),
                ("gt_s", "gt.s32", _OP_I32_GT_S),
                ("gt_u", "gt.u32", _OP_I32_GT_U),
                ("le_s", "le.s32", _OP_I32_LE_S),
                ("le_u", "le.u32", _OP_I32_LE_U),
                ("ge_s", "ge.s32", _OP_I32_GE_S),
                ("ge_u", "ge.u32", _OP_I32_GE_U),
            )
        ),
        *(
            _scalar_binary_descriptor(
                "i64", operation, f"integer.{semantic}.i64", encoding
            )
            for operation, semantic, encoding in (
                ("add", "add", _OP_I64_ADD),
                ("sub", "sub", _OP_I64_SUB),
                ("mul", "mul", _OP_I64_MUL),
                ("rem_u", "remu", _OP_I64_REM_U),
                ("and", "and", _OP_I64_AND),
                ("or", "or", _OP_I64_OR),
                ("xor", "xor", _OP_I64_XOR),
                ("shl", "shl", _OP_I64_SHL),
                ("shr_s", "shrs", _OP_I64_SHR_S),
                ("shr_u", "shru", _OP_I64_SHR_U),
            )
        ),
        *(
            _scalar_binary_descriptor(
                "i64", operation, f"integer.cmp.{semantic}", encoding, comparison=True
            )
            for operation, semantic, encoding in (
                ("eq", "eq.i64", _OP_I64_EQ),
                ("ne", "ne.i64", _OP_I64_NE),
                ("lt_s", "lt.s64", _OP_I64_LT_S),
                ("lt_u", "lt.u64", _OP_I64_LT_U),
                ("gt_s", "gt.s64", _OP_I64_GT_S),
                ("gt_u", "gt.u64", _OP_I64_GT_U),
                ("le_s", "le.s64", _OP_I64_LE_S),
                ("le_u", "le.u64", _OP_I64_LE_U),
                ("ge_s", "ge.s64", _OP_I64_GE_S),
                ("ge_u", "ge.u64", _OP_I64_GE_U),
            )
        ),
        Descriptor(
            key="wasm.f32.add",
            mnemonic="f32.add",
            semantic_tag="float.add.f32",
            encoding_id=_OP_F32_ADD,
            operands=(_f32_result(), _f32_operand("lhs"), _f32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR_F32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32.reinterpret_f32",
            mnemonic="i32.reinterpret_f32",
            semantic_tag="bitcast.f32.i32",
            encoding_id=_OP_I32_REINTERPRET_F32,
            operands=(_i32_result(), _f32_operand("input")),
            asm_forms=_asm(results=("dst",), operands=("input",)),
            schedule_class=_SCHEDULE_SCALAR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32.wrap_i64",
            mnemonic="i32.wrap_i64",
            semantic_tag="integer.trunc.i64.i32",
            encoding_id=_OP_I32_WRAP_I64,
            operands=(_i32_result(), _i64_operand("input")),
            asm_forms=_asm(results=("dst",), operands=("input",)),
            schedule_class=_SCHEDULE_SCALAR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *(
            Descriptor(
                key=f"wasm.i64.extend_i32_{signedness}",
                mnemonic=f"i64.extend_i32_{signedness}",
                semantic_tag=f"integer.ext{signedness}i.i32.i64",
                encoding_id=encoding,
                operands=(_i64_result(), _i32_operand("input")),
                asm_forms=_asm(results=("dst",), operands=("input",)),
                schedule_class=_SCHEDULE_SCALAR_I64,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            for signedness, encoding in (
                ("s", _OP_I64_EXTEND_I32_S),
                ("u", _OP_I64_EXTEND_I32_U),
            )
        ),
        Descriptor(
            key="wasm.i64.reinterpret_f64",
            mnemonic="i64.reinterpret_f64",
            semantic_tag="bitcast.f64.i64",
            encoding_id=_OP_I64_REINTERPRET_F64,
            operands=(_i64_result(), _f64_operand("input")),
            asm_forms=_asm(results=("dst",), operands=("input",)),
            schedule_class=_SCHEDULE_SCALAR_I64,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32.reinterpret_i32",
            mnemonic="f32.reinterpret_i32",
            semantic_tag="bitcast.i32.f32",
            encoding_id=_OP_F32_REINTERPRET_I32,
            operands=(_f32_result(), _i32_operand("input")),
            asm_forms=_asm(results=("dst",), operands=("input",)),
            schedule_class=_SCHEDULE_SCALAR_F32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f64.reinterpret_i64",
            mnemonic="f64.reinterpret_i64",
            semantic_tag="bitcast.i64.f64",
            encoding_id=_OP_F64_REINTERPRET_I64,
            operands=(_f64_result(), _i64_operand("input")),
            asm_forms=_asm(results=("dst",), operands=("input",)),
            schedule_class=_SCHEDULE_SCALAR_F64,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.v128.const",
            mnemonic="v128.const",
            semantic_tag="vector.const.v128",
            encoding_id=_OP_V128_CONST,
            operands=(_v128_result(),),
            op_kind=DescriptorOpKind.CONST,
            immediates=(_V128_LO_IMMEDIATE, _V128_HI_IMMEDIATE),
            asm_forms=_asm(results=("dst",), immediates=("lo64", "hi64")),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i8x16.shuffle",
            mnemonic="i8x16.shuffle",
            semantic_tag="vector.shuffle.i8x16",
            encoding_id=_OP_I8X16_SHUFFLE,
            operands=(
                _v128_result(),
                _v128_operand("lhs"),
                _v128_operand("rhs"),
            ),
            immediates=_SHUFFLE_BYTE_IMMEDIATES,
            asm_forms=_asm(
                results=("dst",),
                operands=("lhs", "rhs"),
                immediates=tuple(
                    immediate.field_name for immediate in _SHUFFLE_BYTE_IMMEDIATES
                ),
            ),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_lane_descriptors(
            "i32x4",
            _i32_result(),
            _i32_operand("value"),
            _LANE_I32X4_IMMEDIATE,
            _SCHEDULE_SIMD_I32X4,
            _OP_I32X4_SPLAT,
            _OP_I32X4_EXTRACT_LANE,
            _OP_I32X4_REPLACE_LANE,
        ),
        *_lane_descriptors(
            "i64x2",
            _i64_result(),
            _i64_operand("value"),
            _LANE_I64X2_IMMEDIATE,
            _SCHEDULE_SIMD_I64X2,
            _OP_I64X2_SPLAT,
            _OP_I64X2_EXTRACT_LANE,
            _OP_I64X2_REPLACE_LANE,
        ),
        *_lane_descriptors(
            "f32x4",
            _f32_result(),
            _f32_operand("value"),
            _LANE_I32X4_IMMEDIATE,
            _SCHEDULE_SIMD_F32X4,
            _OP_F32X4_SPLAT,
            _OP_F32X4_EXTRACT_LANE,
            _OP_F32X4_REPLACE_LANE,
        ),
        *_lane_descriptors(
            "f64x2",
            _f64_result(),
            _f64_operand("value"),
            _LANE_I64X2_IMMEDIATE,
            _SCHEDULE_SIMD_F64X2,
            _OP_F64X2_SPLAT,
            _OP_F64X2_EXTRACT_LANE,
            _OP_F64X2_REPLACE_LANE,
        ),
        Descriptor(
            key="wasm.i32x4.eq",
            mnemonic="i32x4.eq",
            semantic_tag="vector.cmp.eq.i32x4",
            encoding_id=_OP_I32X4_EQ,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.ne",
            mnemonic="i32x4.ne",
            semantic_tag="vector.cmp.ne.i32x4",
            encoding_id=_OP_I32X4_NE,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.lt_s",
            mnemonic="i32x4.lt_s",
            semantic_tag="vector.cmp.slt.i32x4",
            encoding_id=_OP_I32X4_LT_S,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.lt_u",
            mnemonic="i32x4.lt_u",
            semantic_tag="vector.cmp.ult.i32x4",
            encoding_id=_OP_I32X4_LT_U,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.gt_s",
            mnemonic="i32x4.gt_s",
            semantic_tag="vector.cmp.sgt.i32x4",
            encoding_id=_OP_I32X4_GT_S,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.gt_u",
            mnemonic="i32x4.gt_u",
            semantic_tag="vector.cmp.ugt.i32x4",
            encoding_id=_OP_I32X4_GT_U,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.le_s",
            mnemonic="i32x4.le_s",
            semantic_tag="vector.cmp.sle.i32x4",
            encoding_id=_OP_I32X4_LE_S,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.le_u",
            mnemonic="i32x4.le_u",
            semantic_tag="vector.cmp.ule.i32x4",
            encoding_id=_OP_I32X4_LE_U,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.ge_s",
            mnemonic="i32x4.ge_s",
            semantic_tag="vector.cmp.sge.i32x4",
            encoding_id=_OP_I32X4_GE_S,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.ge_u",
            mnemonic="i32x4.ge_u",
            semantic_tag="vector.cmp.uge.i32x4",
            encoding_id=_OP_I32X4_GE_U,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.add",
            mnemonic="i32x4.add",
            semantic_tag="vector.add.i32x4",
            encoding_id=_OP_I32X4_ADD,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.sub",
            mnemonic="i32x4.sub",
            semantic_tag="vector.sub.i32x4",
            encoding_id=_OP_I32X4_SUB,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.mul",
            mnemonic="i32x4.mul",
            semantic_tag="vector.mul.i32x4",
            encoding_id=_OP_I32X4_MUL,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.add",
            mnemonic="f32x4.add",
            semantic_tag="vector.add.f32x4",
            encoding_id=_OP_F32X4_ADD,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.mul",
            mnemonic="f32x4.mul",
            semantic_tag="vector.mul.f32x4",
            encoding_id=_OP_F32X4_MUL,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.eq",
            mnemonic="f32x4.eq",
            semantic_tag="vector.cmp.oeq.f32x4",
            encoding_id=_OP_F32X4_EQ,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.lt",
            mnemonic="f32x4.lt",
            semantic_tag="vector.cmp.olt.f32x4",
            encoding_id=_OP_F32X4_LT,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.gt",
            mnemonic="f32x4.gt",
            semantic_tag="vector.cmp.ogt.f32x4",
            encoding_id=_OP_F32X4_GT,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.le",
            mnemonic="f32x4.le",
            semantic_tag="vector.cmp.ole.f32x4",
            encoding_id=_OP_F32X4_LE,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.ge",
            mnemonic="f32x4.ge",
            semantic_tag="vector.cmp.oge.f32x4",
            encoding_id=_OP_F32X4_GE,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.v128.bitselect",
            mnemonic="v128.bitselect",
            semantic_tag="vector.select.v128",
            encoding_id=_OP_V128_BITSELECT,
            operands=(
                _v128_result(),
                _v128_operand("true_value"),
                _v128_operand("false_value"),
                _v128_predicate("condition"),
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("true_value", "false_value", "condition"),
            ),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *(
            descriptor
            for type_name, register_class, width_bits, load_opcode, store_opcode in (
                ("i32", _REG_I32, 32, _OP_I32_LOAD, _OP_I32_STORE),
                ("i64", _REG_I64, 64, _OP_I64_LOAD, _OP_I64_STORE),
                ("f32", _REG_F32, 32, _OP_F32_LOAD, _OP_F32_STORE),
                ("f64", _REG_F64, 64, _OP_F64_LOAD, _OP_F64_STORE),
            )
            for descriptor in _scalar_memory_descriptors(
                type_name, register_class, width_bits, load_opcode, store_opcode
            )
        ),
        *(
            descriptor
            for width_bits, load_opcode, store_opcode in (
                (8, _OP_I32_LOAD8_U, _OP_I32_STORE8),
                (16, _OP_I32_LOAD16_U, _OP_I32_STORE16),
            )
            for descriptor in _scalar_memory_descriptors(
                "i32",
                _REG_I32,
                width_bits,
                load_opcode,
                store_opcode,
                load_suffix=f"{width_bits}_u",
                store_suffix=str(width_bits),
            )
        ),
        Descriptor(
            key="wasm.v128.load",
            mnemonic="v128.load",
            semantic_tag="memory.load.v128",
            encoding_id=_OP_V128_LOAD,
            operands=(_v128_result(), _i32_resource("address")),
            immediates=(_MEMORY_OFFSET_IMMEDIATE,),
            asm_forms=_asm(
                results=("dst",), operands=("address",), immediates=("offset",)
            ),
            effects=(_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="wasm.v128.store",
            mnemonic="v128.store",
            semantic_tag="memory.store.v128",
            encoding_id=_OP_V128_STORE,
            operands=(_i32_resource("address"), _v128_operand("value")),
            immediates=(_MEMORY_OFFSET_IMMEDIATE,),
            asm_forms=_asm(operands=("address", "value"), immediates=("offset",)),
            effects=(_STORE_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="wasm.br",
            mnemonic="br",
            semantic_tag="control.branch",
            encoding_id=_OP_BR,
            operands=(),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="wasm.br_if.i32",
            mnemonic="br_if",
            semantic_tag="control.cond_branch.i32",
            encoding_id=_OP_BR_IF,
            operands=(_i32_predicate("cond"),),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(operands=("cond",), immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="wasm.return.v128",
            mnemonic="return",
            semantic_tag="control.return.v128",
            encoding_id=_OP_RETURN,
            operands=(_v128_operand("value"),),
            asm_forms=_asm(operands=("value",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
    ),
)
