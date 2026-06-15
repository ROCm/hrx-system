# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the generic LLVMIR oracle target."""

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
    EnumDomain,
    EnumValue,
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

_REG_I1 = "llvmir.i1"
_REG_I8 = "llvmir.i8"
_REG_I16 = "llvmir.i16"
_REG_I32 = "llvmir.i32"
_REG_I64 = "llvmir.i64"
_REG_F16 = "llvmir.f16"
_REG_BF16 = "llvmir.bf16"
_REG_F32 = "llvmir.f32"
_REG_F64 = "llvmir.f64"
_REG_PTR = "llvmir.ptr"

_RESOURCE_ALU = "llvmir.alu"
_RESOURCE_LOAD = "llvmir.load"
_RESOURCE_STORE = "llvmir.store"
_RESOURCE_ATOMIC = "llvmir.atomic"

_SCHEDULE_CONST = "llvmir.const"
_SCHEDULE_ALU = "llvmir.alu"
_SCHEDULE_LOAD = "llvmir.load"
_SCHEDULE_STORE = "llvmir.store"
_SCHEDULE_ATOMIC = "llvmir.atomic"

_CACHE_SCOPE_ENUM = "llvmir.cache_scope"
_CACHE_TEMPORAL_ENUM = "llvmir.cache_temporal"
_ATOMIC_ORDERING_ENUM = "llvmir.atomic_ordering"
_ATOMIC_SCOPE_ENUM = "llvmir.atomic_scope"

_VECTOR_LANE_COUNTS = (2, 3, 4, 8, 16)
_STRUCTURAL_VECTOR_LANE_COUNTS = (*_VECTOR_LANE_COUNTS, 32)
_KERNEL_DIMENSIONS = ("x", "y", "z")
_STRUCTURAL_VECTOR_TYPES = (
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "f64",
)
_MEMORY_VALUE_TYPES = (
    ("i8", 1),
    ("i16", 2),
    ("i32", 4),
    ("i64", 8),
    ("f16", 2),
    ("bf16", 2),
    ("f32", 4),
    ("f64", 8),
)

_ATOMIC_INTEGER_VALUE_TYPES = (
    ("i8", 1),
    ("i16", 2),
    ("i32", 4),
    ("i64", 8),
)
_ATOMIC_FLOAT_VALUE_TYPES = (
    ("f32", 4),
    ("f64", 8),
)

_INTEGER_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)

_FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)

_SCALAR_CAST_SPECS = (
    ("trunc", "i32", "i8"),
    ("trunc", "i32", "i16"),
    ("trunc", "i64", "i32"),
    ("sext", "i8", "i32"),
    ("sext", "i16", "i32"),
    ("sext", "i32", "i64"),
    ("zext", "i8", "i32"),
    ("zext", "i16", "i32"),
    ("zext", "i32", "i64"),
    ("sitofp", "i8", "f32"),
    ("sitofp", "i32", "f32"),
    ("sitofp", "i64", "f64"),
    ("uitofp", "i8", "f32"),
    ("uitofp", "i32", "f32"),
    ("uitofp", "i64", "f64"),
    ("fptosi", "f32", "i32"),
    ("fptosi", "f64", "i64"),
    ("fptoui", "f32", "i32"),
    ("fptoui", "f64", "i64"),
    ("fptrunc", "f32", "f16"),
    ("fptrunc", "f32", "bf16"),
    ("fptrunc", "f64", "f32"),
    ("fpext", "f16", "f32"),
    ("fpext", "bf16", "f32"),
    ("fpext", "f32", "f64"),
    ("bitcast", "i16", "f16"),
    ("bitcast", "i16", "bf16"),
    ("bitcast", "f16", "i16"),
    ("bitcast", "bf16", "i16"),
    ("bitcast", "f16", "bf16"),
    ("bitcast", "bf16", "f16"),
    ("bitcast", "i32", "f32"),
    ("bitcast", "f32", "i32"),
    ("bitcast", "i64", "f64"),
    ("bitcast", "f64", "i64"),
)

_VECTOR_CAST_SPECS = (
    ("sext", "i32", "i64"),
    ("zext", "i32", "i64"),
    ("trunc", "i64", "i32"),
    ("sitofp", "i32", "f32"),
    ("uitofp", "i32", "f32"),
    ("fptosi", "f32", "i32"),
    ("fptoui", "f32", "i32"),
    ("fptrunc", "f32", "f16"),
    ("fptrunc", "f32", "bf16"),
    ("fpext", "f16", "f32"),
    ("fpext", "bf16", "f32"),
    ("bitcast", "i32", "f32"),
    ("bitcast", "f32", "i32"),
    ("bitcast", "i64", "f64"),
    ("bitcast", "f64", "i64"),
)
_BITCAST_RESHAPE_SPECS = (
    ("i32", 1, "i8", 4),
    ("i32", 2, "i8", 8),
    ("f16", 2, "i32", 1),
    ("bf16", 2, "i32", 1),
)

_ALT_BY_TYPE = {
    "i1": (RegClassAlt(_REG_I1),),
    "i8": (RegClassAlt(_REG_I8),),
    "i16": (RegClassAlt(_REG_I16),),
    "i32": (RegClassAlt(_REG_I32),),
    "i64": (RegClassAlt(_REG_I64),),
    "f16": (RegClassAlt(_REG_F16),),
    "bf16": (RegClassAlt(_REG_BF16),),
    "f32": (RegClassAlt(_REG_F32),),
    "f64": (RegClassAlt(_REG_F64),),
    "ptr": (RegClassAlt(_REG_PTR),),
}

_UNIT_BITS_BY_TYPE = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
    "ptr": 64,
}


def _asm(
    *,
    mnemonic: str,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(field_name) for field_name in immediates),
        ),
    )


def _value(
    type_name: str,
    role: OperandRole,
    field_name: str,
    unit_count: int = 1,
) -> Operand:
    return Operand(
        field_name,
        role,
        _ALT_BY_TYPE[type_name],
        unit_count=unit_count,
    )


def _result(type_name: str, suffix: str = "dst", unit_count: int = 1) -> Operand:
    return _value(type_name, OperandRole.RESULT, suffix, unit_count)


def _operand(type_name: str, field_name: str, unit_count: int = 1) -> Operand:
    return _value(type_name, OperandRole.OPERAND, field_name, unit_count)


def _predicate(
    type_name: str,
    field_name: str = "condition",
    unit_count: int = 1,
) -> Operand:
    return _value(type_name, OperandRole.PREDICATE, field_name, unit_count)


def _ptr_operand(field_name: str) -> Operand:
    return _value("ptr", OperandRole.RESOURCE, field_name)


def _descriptor_suffix(type_name: str, unit_count: int, *, vector: bool = False) -> str:
    if vector:
        return f"v{unit_count}{type_name}"
    return type_name if unit_count == 1 else f"v{unit_count}{type_name}"


_I64_VALUE_IMMEDIATE = Immediate(
    "value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_F32_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_F16_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    unsigned_max=(2**16) - 1,
)

_BF16_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    unsigned_max=(2**16) - 1,
)

_F64_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_BYTE_OFFSET_IMMEDIATE = Immediate(
    "byte_offset",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_BYTE_STRIDE_IMMEDIATE = Immediate(
    "byte_stride",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_BASE_ALIGNMENT_IMMEDIATE = Immediate(
    "base_alignment",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_CACHE_SCOPE_IMMEDIATE = Immediate(
    "cache_scope",
    ImmediateKind.ENUM,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    enum_domain=_CACHE_SCOPE_ENUM,
    default_value=0,
)

_CACHE_TEMPORAL_IMMEDIATE = Immediate(
    "cache_temporal",
    ImmediateKind.ENUM,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    enum_domain=_CACHE_TEMPORAL_ENUM,
    default_value=0,
)

_ATOMIC_ORDERING_IMMEDIATE = Immediate(
    "ordering",
    ImmediateKind.ENUM,
    enum_domain=_ATOMIC_ORDERING_ENUM,
)

_ATOMIC_SUCCESS_ORDERING_IMMEDIATE = Immediate(
    "success_ordering",
    ImmediateKind.ENUM,
    enum_domain=_ATOMIC_ORDERING_ENUM,
)

_ATOMIC_FAILURE_ORDERING_IMMEDIATE = Immediate(
    "failure_ordering",
    ImmediateKind.ENUM,
    enum_domain=_ATOMIC_ORDERING_ENUM,
)

_ATOMIC_SCOPE_IMMEDIATE = Immediate(
    "scope",
    ImmediateKind.ENUM,
    enum_domain=_ATOMIC_SCOPE_ENUM,
)

_FAST_MATH_FLAGS_IMMEDIATE = Immediate(
    "fast_math_flags",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=7,
    unsigned_max=0x7F,
    default_value=0,
)


def _lane_immediate(lane_count: int) -> Immediate:
    return Immediate(
        "lane",
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=lane_count - 1,
    )


def _offset_immediate(maximum: int) -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=maximum,
    )


def _shuffle_lane_immediate(name: str, lane_count: int) -> Immediate:
    return Immediate(
        name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=lane_count - 1,
    )


def _read_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _write_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.WRITE,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _atomic_effects(width_bits: int) -> tuple[Effect, ...]:
    return (
        _read_effect(width_bits),
        _write_effect(width_bits),
    )


def _const_i_descriptor(
    type_name: str,
    unit_count: int = 1,
    *,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.const.{suffix}",
        mnemonic="const",
        semantic_tag=f"llvmir.const.{suffix}",
        operands=(_result(type_name, unit_count=unit_count),),
        immediates=(_I64_VALUE_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=f"const.{suffix}",
            results=("dst",),
            immediates=("value",),
        ),
        schedule_class=_SCHEDULE_CONST,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _const_f_descriptor(
    type_name: str,
    immediate: Immediate,
    unit_count: int = 1,
    *,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.const.{suffix}",
        mnemonic="const",
        semantic_tag=f"llvmir.const.{suffix}",
        operands=(_result(type_name, unit_count=unit_count),),
        immediates=(immediate,),
        asm_forms=_asm(
            mnemonic=f"const.{suffix}",
            results=("dst",),
            immediates=("bits",),
        ),
        schedule_class=_SCHEDULE_CONST,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _constant_descriptors() -> tuple[Descriptor, ...]:
    return (
        _const_i_descriptor("i1"),
        _const_i_descriptor("i8"),
        _const_i_descriptor("i16"),
        _const_i_descriptor("i32"),
        _const_i_descriptor("i64"),
        _const_f_descriptor("f16", _F16_BITS_IMMEDIATE),
        _const_f_descriptor("bf16", _BF16_BITS_IMMEDIATE),
        _const_f_descriptor("f32", _F32_BITS_IMMEDIATE),
        _const_f_descriptor("f64", _F64_BITS_IMMEDIATE),
        *(
            _const_i_descriptor(type_name, lane_count, vector=True)
            for type_name in ("i8", "i16", "i32", "i64")
            for lane_count in _VECTOR_LANE_COUNTS
        ),
        *(
            _const_f_descriptor(type_name, immediate, lane_count, vector=True)
            for type_name, immediate in (
                ("f16", _F16_BITS_IMMEDIATE),
                ("bf16", _BF16_BITS_IMMEDIATE),
                ("f32", _F32_BITS_IMMEDIATE),
                ("f64", _F64_BITS_IMMEDIATE),
            )
            for lane_count in _VECTOR_LANE_COUNTS
        ),
    )


def _binary_descriptor(
    *,
    stem: str,
    type_name: str,
    semantic_stem: str,
    unit_count: int = 1,
    vector: bool = False,
    fast_math_flags: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.{stem}.{suffix}",
        mnemonic=stem,
        semantic_tag=f"llvmir.{semantic_stem}.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _operand(type_name, "lhs", unit_count=unit_count),
            _operand(type_name, "rhs", unit_count=unit_count),
        ),
        immediates=(_FAST_MATH_FLAGS_IMMEDIATE,) if fast_math_flags else (),
        asm_forms=_asm(
            mnemonic=f"{stem}.{suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _unary_descriptor(
    *,
    stem: str,
    type_name: str,
    semantic_stem: str,
    unit_count: int = 1,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.{stem}.{suffix}",
        mnemonic=stem,
        semantic_tag=f"llvmir.{semantic_stem}.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _operand(type_name, "input", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"{stem}.{suffix}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _ternary_descriptor(
    *,
    stem: str,
    type_name: str,
    semantic_stem: str,
    unit_count: int = 1,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.{stem}.{suffix}",
        mnemonic=stem,
        semantic_tag=f"llvmir.{semantic_stem}.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _operand(type_name, "a", unit_count=unit_count),
            _operand(type_name, "b", unit_count=unit_count),
            _operand(type_name, "c", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"{stem}.{suffix}",
            results=("dst",),
            operands=("a", "b", "c"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _compare_descriptor(
    *,
    predicate: str,
    type_name: str,
    unit_count: int = 1,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    result_unit_count = unit_count
    return Descriptor(
        key=f"llvmir.cmp.{predicate}.{suffix}",
        mnemonic="cmp",
        semantic_tag=f"llvmir.cmp.{predicate}.{suffix}",
        operands=(
            _result("i1", unit_count=result_unit_count),
            _operand(type_name, "lhs", unit_count=unit_count),
            _operand(type_name, "rhs", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"cmp.{predicate}.{suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _cast_descriptor(
    *,
    stem: str,
    source_type: str,
    result_type: str,
    unit_count: int = 1,
    vector: bool = False,
    source_unit_count: int | None = None,
    result_unit_count: int | None = None,
    source_vector: bool | None = None,
    result_vector: bool | None = None,
) -> Descriptor:
    source_unit_count = unit_count if source_unit_count is None else source_unit_count
    result_unit_count = unit_count if result_unit_count is None else result_unit_count
    source_vector = vector if source_vector is None else source_vector
    result_vector = vector if result_vector is None else result_vector
    source_suffix = _descriptor_suffix(
        source_type, source_unit_count, vector=source_vector
    )
    result_suffix = _descriptor_suffix(
        result_type, result_unit_count, vector=result_vector
    )
    return Descriptor(
        key=f"llvmir.{stem}.{source_suffix}.{result_suffix}",
        mnemonic=stem,
        semantic_tag=f"llvmir.{stem}.{source_suffix}.{result_suffix}",
        operands=(
            _result(result_type, unit_count=result_unit_count),
            _operand(source_type, "value", unit_count=source_unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"{stem}.{source_suffix}.{result_suffix}",
            results=("dst",),
            operands=("value",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _select_descriptor(
    type_name: str,
    unit_count: int = 1,
    *,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.select.{suffix}",
        mnemonic="select",
        semantic_tag=f"llvmir.select.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _predicate("i1", unit_count=unit_count),
            _operand(type_name, "true_value", unit_count=unit_count),
            _operand(type_name, "false_value", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"select.{suffix}",
            results=("dst",),
            operands=("condition", "true_value", "false_value"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _load_descriptor(
    type_name: str,
    *,
    unit_count: int = 1,
    indexed: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count)
    operands = [_result(type_name, unit_count=unit_count), _ptr_operand("ptr")]
    immediates = [
        _BYTE_OFFSET_IMMEDIATE,
        _CACHE_SCOPE_IMMEDIATE,
        _CACHE_TEMPORAL_IMMEDIATE,
    ]
    asm_operands = ["ptr"]
    asm_immediates = ["byte_offset"]
    key_suffix = f"indexed.{suffix}" if indexed else suffix
    mnemonic_suffix = f"indexed.{suffix}" if indexed else suffix
    if indexed:
        operands.append(_operand("i64", "index"))
        immediates.append(_BYTE_STRIDE_IMMEDIATE)
        asm_operands.append("index")
        asm_immediates.append("byte_stride")
    return Descriptor(
        key=f"llvmir.load.{key_suffix}",
        mnemonic="load",
        semantic_tag=f"llvmir.load.{key_suffix}",
        operands=tuple(operands),
        immediates=tuple(immediates),
        asm_forms=_asm(
            mnemonic=f"load.{mnemonic_suffix}",
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
        effects=(_read_effect(_UNIT_BITS_BY_TYPE[type_name] * unit_count),),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _store_descriptor(
    type_name: str,
    *,
    unit_count: int = 1,
    indexed: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count)
    operands = [
        _operand(type_name, "value", unit_count=unit_count),
        _ptr_operand("ptr"),
    ]
    immediates = [
        _BYTE_OFFSET_IMMEDIATE,
        _CACHE_SCOPE_IMMEDIATE,
        _CACHE_TEMPORAL_IMMEDIATE,
    ]
    asm_operands = ["value", "ptr"]
    asm_immediates = ["byte_offset"]
    key_suffix = f"indexed.{suffix}" if indexed else suffix
    mnemonic_suffix = f"indexed.{suffix}" if indexed else suffix
    if indexed:
        operands.append(_operand("i64", "index"))
        immediates.append(_BYTE_STRIDE_IMMEDIATE)
        asm_operands.append("index")
        asm_immediates.append("byte_stride")
    return Descriptor(
        key=f"llvmir.store.{key_suffix}",
        mnemonic="store",
        semantic_tag=f"llvmir.store.{key_suffix}",
        operands=tuple(operands),
        immediates=tuple(immediates),
        asm_forms=_asm(
            mnemonic=f"store.{mnemonic_suffix}",
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
        effects=(_write_effect(_UNIT_BITS_BY_TYPE[type_name] * unit_count),),
        schedule_class=_SCHEDULE_STORE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _atomic_descriptor(
    form: str,
    operation: str,
    type_name: str,
    *,
    indexed: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, 1)
    operands = []
    asm_results = ()
    if form in ("rmw", "cmpxchg"):
        operands.append(_result(type_name, "old"))
        asm_results = ("old",)
    if form == "cmpxchg":
        operands.extend(
            (
                _operand(type_name, "expected"),
                _operand(type_name, "replacement"),
                _ptr_operand("ptr"),
            )
        )
        immediates = [
            _BYTE_OFFSET_IMMEDIATE,
            _ATOMIC_SUCCESS_ORDERING_IMMEDIATE,
            _ATOMIC_FAILURE_ORDERING_IMMEDIATE,
            _ATOMIC_SCOPE_IMMEDIATE,
            _CACHE_SCOPE_IMMEDIATE,
            _CACHE_TEMPORAL_IMMEDIATE,
        ]
        asm_operands = ["expected", "replacement", "ptr"]
        asm_immediates = [
            "byte_offset",
            "success_ordering",
            "failure_ordering",
            "scope",
        ]
        key_suffix = f"indexed.{suffix}" if indexed else suffix
        mnemonic_suffix = f"indexed.{suffix}" if indexed else suffix
    else:
        operands.extend(
            (
                _operand(type_name, "value"),
                _ptr_operand("ptr"),
            )
        )
        immediates = [
            _BYTE_OFFSET_IMMEDIATE,
            _ATOMIC_ORDERING_IMMEDIATE,
            _ATOMIC_SCOPE_IMMEDIATE,
            _CACHE_SCOPE_IMMEDIATE,
            _CACHE_TEMPORAL_IMMEDIATE,
        ]
        asm_operands = ["value", "ptr"]
        asm_immediates = ["byte_offset", "ordering", "scope"]
        key_suffix = (
            f"indexed.{operation}.{suffix}" if indexed else f"{operation}.{suffix}"
        )
        mnemonic_suffix = (
            f"indexed.{operation}.{suffix}" if indexed else f"{operation}.{suffix}"
        )
    if indexed:
        operands.append(_operand("i64", "index"))
        immediates.append(_BYTE_STRIDE_IMMEDIATE)
        asm_operands.append("index")
        asm_immediates.append("byte_stride")
    return Descriptor(
        key=f"llvmir.atomic.{form}.{key_suffix}",
        mnemonic=f"atomic.{form}",
        semantic_tag=f"llvmir.atomic.{form}.{key_suffix}",
        operands=tuple(operands),
        immediates=tuple(immediates),
        asm_forms=_asm(
            mnemonic=f"atomic.{form}.{mnemonic_suffix}",
            results=asm_results,
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
        effects=_atomic_effects(_UNIT_BITS_BY_TYPE[type_name]),
        schedule_class=_SCHEDULE_ATOMIC,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _alloca_descriptor(memory_space: str) -> Descriptor:
    return Descriptor(
        key=f"llvmir.alloca.{memory_space}.i8",
        mnemonic="alloca",
        semantic_tag=f"llvmir.alloca.{memory_space}.i8",
        operands=(
            _result("ptr", "ptr"),
            _operand("i64", "byte_length"),
        ),
        immediates=(_BASE_ALIGNMENT_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=f"alloca.{memory_space}.i8",
            results=("ptr",),
            operands=("byte_length",),
            immediates=("base_alignment",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _arithmetic_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("add", "sub", "mul", "udiv", "sdiv", "urem", "srem")
        )
    for type_name in ("f32", "f64"):
        descriptors.extend(
            _unary_descriptor(
                stem=stem,
                type_name=type_name,
                semantic_stem=stem,
            )
            for stem in ("neg", "abs")
        )
        descriptors.extend(
            _binary_descriptor(
                stem=stem,
                type_name=type_name,
                semantic_stem=stem,
                fast_math_flags=True,
            )
            for stem in ("add", "sub", "mul", "div")
        )
        descriptors.extend(
            _binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem)
            for stem in ("minnum", "maxnum")
        )
        descriptors.append(
            _ternary_descriptor(
                stem="fma",
                type_name=type_name,
                semantic_stem="fma",
            )
        )
    for type_name in ("i32", "f32"):
        stems = (
            ("add", "sub", "mul", "div")
            if type_name == "f32"
            else ("add", "sub", "mul")
        )
        descriptors.extend(
            (
                _binary_descriptor(
                    stem=stem,
                    type_name=type_name,
                    semantic_stem=stem,
                    unit_count=lane_count,
                    vector=True,
                    fast_math_flags=type_name == "f32",
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for stem in stems
        )
    descriptors.extend(
        _ternary_descriptor(
            stem="fma",
            type_name="f32",
            semantic_stem="fma",
            unit_count=lane_count,
            vector=True,
        )
        for lane_count in _VECTOR_LANE_COUNTS
    )
    for stem in ("neg", "abs"):
        descriptors.extend(
            _unary_descriptor(
                stem=stem,
                type_name="f32",
                semantic_stem=stem,
                unit_count=lane_count,
                vector=True,
            )
            for lane_count in _VECTOR_LANE_COUNTS
        )
    for stem in ("minnum", "maxnum"):
        descriptors.extend(
            _binary_descriptor(
                stem=stem,
                type_name="f32",
                semantic_stem=stem,
                unit_count=lane_count,
                vector=True,
            )
            for lane_count in _VECTOR_LANE_COUNTS
        )
    return tuple(descriptors)


def _bitwise_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i1", "i8", "i16", "i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("and", "or", "xor")
        )
    for type_name in ("i8", "i16", "i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("shl", "lshr", "ashr")
        )
    for type_name, stems in (
        ("i1", ("and", "or", "xor")),
        ("i8", ("and", "or", "xor", "shl", "lshr", "ashr")),
        ("i16", ("and", "or", "xor", "shl", "lshr", "ashr")),
        ("i32", ("and", "or", "xor", "shl", "lshr", "ashr")),
    ):
        descriptors.extend(
            (
                _binary_descriptor(
                    stem=stem,
                    type_name=type_name,
                    semantic_stem=stem,
                    unit_count=lane_count,
                    vector=True,
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for stem in stems
        )
    return tuple(descriptors)


def _compare_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i32", "i64"):
        descriptors.extend(
            _compare_descriptor(predicate=predicate, type_name=type_name)
            for predicate in _INTEGER_PREDICATES
        )
    for type_name in ("f32", "f64"):
        descriptors.extend(
            _compare_descriptor(predicate=predicate, type_name=type_name)
            for predicate in _FLOAT_PREDICATES
        )
    for type_name in ("i32", "f32"):
        predicates = _INTEGER_PREDICATES if type_name == "i32" else _FLOAT_PREDICATES
        descriptors.extend(
            _compare_descriptor(
                predicate=predicate,
                type_name=type_name,
                unit_count=lane_count,
                vector=True,
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for predicate in predicates
        )
    return tuple(descriptors)


def _cast_descriptors() -> tuple[Descriptor, ...]:
    return (
        tuple(
            _cast_descriptor(
                stem=stem, source_type=source_type, result_type=result_type
            )
            for stem, source_type, result_type in _SCALAR_CAST_SPECS
        )
        + tuple(
            _cast_descriptor(
                stem=stem,
                source_type=source_type,
                result_type=result_type,
                unit_count=lane_count,
                vector=True,
            )
            for stem, source_type, result_type in _VECTOR_CAST_SPECS
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _cast_descriptor(
                stem="bitcast",
                source_type=source_type,
                result_type=result_type,
                source_unit_count=source_unit_count,
                result_unit_count=result_unit_count,
                source_vector=source_unit_count != 1,
                result_vector=result_unit_count != 1,
            )
            for (
                source_type,
                source_unit_count,
                result_type,
                result_unit_count,
            ) in _BITCAST_RESHAPE_SPECS
        )
    )


def _select_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _select_descriptor(type_name, unit_count=unit_count)
        for type_name, unit_count in (
            ("i1", 1),
            ("i32", 1),
            ("i64", 1),
            ("f32", 1),
            ("f64", 1),
        )
    ) + tuple(
        _select_descriptor(type_name, unit_count=lane_count, vector=True)
        for type_name in ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
        for lane_count in _VECTOR_LANE_COUNTS
    )


def _kernel_query_descriptor(query: str, dimension: str) -> Descriptor:
    return Descriptor(
        key=f"llvmir.kernel.{query}.{dimension}",
        mnemonic="kernel",
        semantic_tag=f"llvmir.kernel.{query}.{dimension}",
        operands=(_result("i64"),),
        asm_forms=_asm(
            mnemonic=f"kernel.{query}.{dimension}",
            results=("dst",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _kernel_scalar_query_descriptor(query: str) -> Descriptor:
    return Descriptor(
        key=f"llvmir.kernel.{query}",
        mnemonic="kernel",
        semantic_tag=f"llvmir.kernel.{query}",
        operands=(_result("i64"),),
        asm_forms=_asm(
            mnemonic=f"kernel.{query}",
            results=("dst",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _kernel_query_descriptors() -> tuple[Descriptor, ...]:
    dimensioned = tuple(
        _kernel_query_descriptor(query, dimension)
        for query in (
            "workitem_id",
            "workgroup_id",
            "workgroup_size",
            "workitem_dispatch_id",
        )
        for dimension in _KERNEL_DIMENSIONS
    )
    scalar = tuple(
        _kernel_scalar_query_descriptor(query)
        for query in (
            "subgroup_size",
            "subgroup_count",
        )
    )
    return dimensioned + scalar


def _splat_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.splat.{suffix}",
        mnemonic="splat",
        semantic_tag=f"llvmir.splat.{suffix}",
        operands=(
            _result(type_name, unit_count=lane_count),
            _operand(type_name, "value"),
        ),
        asm_forms=_asm(
            mnemonic=f"splat.{suffix}",
            results=("dst",),
            operands=("value",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _from_elements_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.from_elements.{suffix}",
        mnemonic="from_elements",
        semantic_tag=f"llvmir.from_elements.{suffix}",
        operands=(
            _result(type_name, unit_count=lane_count),
            *(
                _operand(type_name, f"lane{lane_index}")
                for lane_index in range(lane_count)
            ),
        ),
        asm_forms=_asm(
            mnemonic=f"from_elements.{suffix}",
            results=("dst",),
            operands=tuple(f"lane{lane_index}" for lane_index in range(lane_count)),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _extract_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.extract.{suffix}",
        mnemonic="extract",
        semantic_tag=f"llvmir.extract.{suffix}",
        operands=(
            _result(type_name),
            _operand(type_name, "source", unit_count=lane_count),
        ),
        immediates=(_lane_immediate(lane_count),),
        asm_forms=_asm(
            mnemonic=f"extract.{suffix}",
            results=("dst",),
            operands=("source",),
            immediates=("lane",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _insert_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.insert.{suffix}",
        mnemonic="insert",
        semantic_tag=f"llvmir.insert.{suffix}",
        operands=(
            _result(type_name, unit_count=lane_count),
            _operand(type_name, "dest", unit_count=lane_count),
            _operand(type_name, "value"),
        ),
        immediates=(_lane_immediate(lane_count),),
        asm_forms=_asm(
            mnemonic=f"insert.{suffix}",
            results=("dst",),
            operands=("dest", "value"),
            immediates=("lane",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _dynamic_insert_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.insert.dynamic.{suffix}",
        mnemonic="insert.dynamic",
        semantic_tag=f"llvmir.insert.dynamic.{suffix}",
        operands=(
            _result(type_name, unit_count=lane_count),
            _operand(type_name, "dest", unit_count=lane_count),
            _operand(type_name, "value"),
            _operand("i64", "index"),
        ),
        asm_forms=_asm(
            mnemonic=f"insert.dynamic.{suffix}",
            results=("dst",),
            operands=("dest", "value", "index"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _shuffle_descriptor(type_name: str, lane_count: int) -> Descriptor:
    suffix = _descriptor_suffix(type_name, lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.shuffle.{suffix}",
        mnemonic="shuffle",
        semantic_tag=f"llvmir.shuffle.{suffix}",
        operands=(
            _result(type_name, unit_count=lane_count),
            _operand(type_name, "source", unit_count=lane_count),
        ),
        immediates=tuple(
            _shuffle_lane_immediate(f"lane{lane_index}", lane_count)
            for lane_index in range(lane_count)
        ),
        asm_forms=_asm(
            mnemonic=f"shuffle.{suffix}",
            results=("dst",),
            operands=("source",),
            immediates=tuple(f"lane{lane_index}" for lane_index in range(lane_count)),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _concat_descriptor(
    type_name: str,
    *,
    input_lane_count: int,
    input_count: int,
) -> Descriptor:
    input_suffix = _descriptor_suffix(type_name, input_lane_count, vector=True)
    result_lane_count = input_lane_count * input_count
    result_suffix = _descriptor_suffix(type_name, result_lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.concat.{input_suffix}.{result_suffix}",
        mnemonic="concat",
        semantic_tag=f"llvmir.concat.{input_suffix}.{result_suffix}",
        operands=(
            _result(type_name, unit_count=result_lane_count),
            *(
                _operand(type_name, f"input{input_ordinal}", input_lane_count)
                for input_ordinal in range(input_count)
            ),
        ),
        asm_forms=_asm(
            mnemonic=f"concat.{input_suffix}.{result_suffix}",
            results=("dst",),
            operands=tuple(
                f"input{input_ordinal}" for input_ordinal in range(input_count)
            ),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _slice_descriptor(
    type_name: str,
    *,
    source_lane_count: int,
    result_lane_count: int,
) -> Descriptor:
    source_suffix = _descriptor_suffix(type_name, source_lane_count, vector=True)
    result_suffix = _descriptor_suffix(type_name, result_lane_count, vector=True)
    return Descriptor(
        key=f"llvmir.slice.{source_suffix}.{result_suffix}",
        mnemonic="slice",
        semantic_tag=f"llvmir.slice.{source_suffix}.{result_suffix}",
        operands=(
            _result(type_name, unit_count=result_lane_count),
            _operand(type_name, "source", unit_count=source_lane_count),
        ),
        immediates=(_offset_immediate(source_lane_count - result_lane_count),),
        asm_forms=_asm(
            mnemonic=f"slice.{source_suffix}.{result_suffix}",
            results=("dst",),
            operands=("source",),
            immediates=("offset",),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _structural_vector_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in _STRUCTURAL_VECTOR_TYPES:
        for lane_count in _STRUCTURAL_VECTOR_LANE_COUNTS:
            descriptors.append(_splat_descriptor(type_name, lane_count))
            descriptors.append(_from_elements_descriptor(type_name, lane_count))
            descriptors.append(_extract_descriptor(type_name, lane_count))
            descriptors.append(_insert_descriptor(type_name, lane_count))
            descriptors.append(_dynamic_insert_descriptor(type_name, lane_count))
            descriptors.append(_shuffle_descriptor(type_name, lane_count))
        descriptors.append(
            _slice_descriptor(type_name, source_lane_count=4, result_lane_count=2)
        )
        descriptors.append(
            _slice_descriptor(type_name, source_lane_count=16, result_lane_count=4)
        )
        descriptors.append(
            _concat_descriptor(type_name, input_lane_count=4, input_count=4)
        )
    return tuple(descriptors)


def _memory_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name, _ in _MEMORY_VALUE_TYPES:
        for indexed in (False, True):
            descriptors.append(
                _load_descriptor(
                    type_name,
                    indexed=indexed,
                )
            )
            descriptors.append(
                _store_descriptor(
                    type_name,
                    indexed=indexed,
                )
            )
        for unit_count in _VECTOR_LANE_COUNTS:
            for indexed in (False, True):
                descriptors.append(
                    _load_descriptor(
                        type_name,
                        unit_count=unit_count,
                        indexed=indexed,
                    )
                )
                descriptors.append(
                    _store_descriptor(
                        type_name,
                        unit_count=unit_count,
                        indexed=indexed,
                    )
                )
    return tuple(descriptors)


def _atomic_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name, _ in _ATOMIC_INTEGER_VALUE_TYPES:
        integer_reduce_operations = (
            "add",
            "sub",
            "and",
            "or",
            "xor",
            "min",
            "max",
            "umin",
            "umax",
        )
        descriptors.extend(
            _atomic_descriptor(
                "reduce",
                operation,
                type_name,
                indexed=indexed,
            )
            for operation in integer_reduce_operations
            for indexed in (False, True)
        )
        integer_rmw_operations = (
            "xchg",
            "add",
            "sub",
            "and",
            "or",
            "xor",
            "min",
            "max",
            "umin",
            "umax",
        )
        descriptors.extend(
            _atomic_descriptor("rmw", operation, type_name, indexed=indexed)
            for operation in integer_rmw_operations
            for indexed in (False, True)
        )
        descriptors.extend(
            _atomic_descriptor(
                "cmpxchg",
                "cmpxchg",
                type_name,
                indexed=indexed,
            )
            for indexed in (False, True)
        )
    for type_name, _ in _ATOMIC_FLOAT_VALUE_TYPES:
        float_reduce_operations = ("fadd", "fminimum", "fmaximum", "fmin", "fmax")
        descriptors.extend(
            _atomic_descriptor(
                "reduce",
                operation,
                type_name,
                indexed=indexed,
            )
            for operation in float_reduce_operations
            for indexed in (False, True)
        )
        float_rmw_operations = ("xchg", "fadd", "fminimum", "fmaximum", "fmin", "fmax")
        descriptors.extend(
            _atomic_descriptor("rmw", operation, type_name, indexed=indexed)
            for operation in float_rmw_operations
            for indexed in (False, True)
        )
    return tuple(descriptors)


def _allocation_descriptors() -> tuple[Descriptor, ...]:
    return (
        _alloca_descriptor("private"),
        _alloca_descriptor("workgroup"),
    )


LLVMIR_GENERIC_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="llvmir.generic.core",
    target_key="llvmir",
    feature_key="llvmir.generic.v1",
    c_header_path=Path("loom/src/loom/target/arch/llvmir/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/llvmir/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_LLVMIR_DESCRIPTORS_H_",
    public_header="loom/target/arch/llvmir/descriptors/descriptors.h",
    function_name="loom_llvmir_generic_core_descriptor_set",
    c_table_prefix="LlvmirGenericCore",
    c_enum_prefix="LLVMIR_GENERIC_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I1, 1, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I8, 8, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_BF16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_PTR, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_ATOMIC, capacity_per_cycle=1, kind=ResourceKind.STORE),
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
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            flags=(ScheduleClassFlag.MAY_LOAD,),
        ),
        ScheduleClass(
            _SCHEDULE_STORE,
            latency_kind=LatencyKind.VARIABLE,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            flags=(ScheduleClassFlag.MAY_STORE,),
        ),
        ScheduleClass(
            _SCHEDULE_ATOMIC,
            latency_kind=LatencyKind.VARIABLE,
            issue_uses=(IssueUse(_RESOURCE_ATOMIC, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
        ),
    ),
    enum_domains=(
        EnumDomain(
            _CACHE_SCOPE_ENUM,
            (
                EnumValue("cu", 0),
                EnumValue("se", 1),
                EnumValue("device", 2),
                EnumValue("system", 3),
            ),
        ),
        EnumDomain(
            _CACHE_TEMPORAL_ENUM,
            (
                EnumValue("regular", 0),
                EnumValue("non_temporal", 1),
                EnumValue("high_temporal", 2),
                EnumValue("last_use", 3),
                EnumValue("writeback", 4),
                EnumValue("non_temporal_regular", 5),
                EnumValue("regular_non_temporal", 6),
                EnumValue("non_temporal_high_temporal", 7),
                EnumValue("non_temporal_writeback", 8),
                EnumValue("bypass", 9),
            ),
        ),
        EnumDomain(
            _ATOMIC_ORDERING_ENUM,
            (
                EnumValue("relaxed", 0),
                EnumValue("acquire", 1),
                EnumValue("release", 2),
                EnumValue("acq_rel", 3),
                EnumValue("seq_cst", 4),
            ),
        ),
        EnumDomain(
            _ATOMIC_SCOPE_ENUM,
            (
                EnumValue("thread", 0),
                EnumValue("subgroup", 1),
                EnumValue("workgroup", 2),
                EnumValue("device", 3),
                EnumValue("system", 4),
            ),
        ),
    ),
    descriptors=(
        *_constant_descriptors(),
        *_arithmetic_descriptors(),
        *_bitwise_descriptors(),
        *_compare_descriptors(),
        *_cast_descriptors(),
        *_select_descriptors(),
        *_kernel_query_descriptors(),
        *_structural_vector_descriptors(),
        *_memory_descriptors(),
        *_atomic_descriptors(),
        *_allocation_descriptors(),
    ),
)
