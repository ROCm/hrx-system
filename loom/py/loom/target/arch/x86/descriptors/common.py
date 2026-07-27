# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for x86 target-low descriptor shards."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from loom.target.arch.x86.packed_dot_data import (
    CONTRACT_FLAG_SATURATING,
    NUMERIC_BF16,
    NUMERIC_F16,
    NUMERIC_F32,
    NUMERIC_I8,
    NUMERIC_I16,
    NUMERIC_I32,
    NUMERIC_U8,
    NUMERIC_U16,
    PackedDotDescriptor,
)
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    Effect,
    EffectFlag,
    EffectKind,
    EnumDomain,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    MemorySpace,
    Operand,
    OperandAddressMapKind,
    OperandRole,
    RegClass,
    RegClassAlt,
    Resource,
    ScheduleClass,
)

_T = TypeVar("_T", RegClass, Resource, ScheduleClass, EnumDomain)

_REG_GPR32 = "x86.gpr32"
_REG_GPR64 = "x86.gpr64"
_REG_XMM = "x86.xmm"
_REG_YMM = "x86.ymm"
_REG_ZMM = "x86.zmm"
_REG_K = "x86.k"

_RESOURCE_SCALAR = "x86.scalar"
_RESOURCE_VECTOR = "x86.vector"
_RESOURCE_DOT = "x86.vector.dot"
_RESOURCE_MASK = "x86.mask"
_RESOURCE_LOAD = "x86.load"
_RESOURCE_STORE = "x86.store"
_RESOURCE_ADDRESS = "x86.address"
_RESOURCE_CONTROL = "x86.control"

_SCHEDULE_SCALAR = "x86.scalar"
_SCHEDULE_ADDRESS = "x86.address"
_SCHEDULE_VECTOR_I32_XMM = "x86.vector.i32.128"
_SCHEDULE_VECTOR_I32_ZMM = "x86.vector.i32.512"
_SCHEDULE_VECTOR_F32_XMM = "x86.vector.f32.128"
_SCHEDULE_VECTOR_F32_ZMM = "x86.vector.f32.512"
_SCHEDULE_VECTOR_FMA_F32_XMM = "x86.vector.fma.f32.128"
_SCHEDULE_VECTOR_FMA_F32_ZMM = "x86.vector.fma.f32.512"
_SCHEDULE_VECTOR_COMPARE_XMM = "x86.vector.compare.128"
_SCHEDULE_VECTOR_COMPARE_ZMM = "x86.vector.compare.512"
_SCHEDULE_VECTOR_DOT_XMM = "x86.vector.dot.128"
_SCHEDULE_VECTOR_DOT_YMM = "x86.vector.dot.256"
_SCHEDULE_VECTOR_DOT_ZMM = "x86.vector.dot.512"
_SCHEDULE_MASK = "x86.mask"
_SCHEDULE_MEMORY_LOAD_GPR32 = "x86.memory.load.gpr32"
_SCHEDULE_MEMORY_LOAD_GPR64 = "x86.memory.load.gpr64"
_SCHEDULE_MEMORY_LOAD_XMM = "x86.memory.load.128"
_SCHEDULE_MEMORY_LOAD_ZMM = "x86.memory.load"
_SCHEDULE_MEMORY_STORE_GPR32 = "x86.memory.store.gpr32"
_SCHEDULE_MEMORY_STORE_GPR64 = "x86.memory.store.gpr64"
_SCHEDULE_MEMORY_STORE_XMM = "x86.memory.store.128"
_SCHEDULE_MEMORY_STORE_ZMM = "x86.memory.store"
_SCHEDULE_CONTROL = "x86.control"

_GPR32_ALT = (RegClassAlt(_REG_GPR32),)
_GPR64_ALT = (RegClassAlt(_REG_GPR64),)
_XMM_ALT = (RegClassAlt(_REG_XMM),)
_YMM_ALT = (RegClassAlt(_REG_YMM),)
_ZMM_ALT = (RegClassAlt(_REG_ZMM),)
_K_ALT = (RegClassAlt(_REG_K),)


def _low_subset_operand(operand: Operand, addressable_unit_count: int) -> Operand:
    return replace(
        operand,
        address_map_kind=OperandAddressMapKind.LOW_SUBSET,
        addressable_unit_count=addressable_unit_count,
    )


def _vector_lane_units(vector_bit_width: int) -> int:
    if vector_bit_width % 128 != 0:
        raise ValueError(f"unsupported x86 vector width {vector_bit_width}")
    return vector_bit_width // 128


def _vector_dot_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_DOT_XMM
        case 256:
            return _SCHEDULE_VECTOR_DOT_YMM
        case 512:
            return _SCHEDULE_VECTOR_DOT_ZMM
        case _:
            raise ValueError(f"unsupported x86 dot schedule width {vector_bit_width}")


def _vector_i32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_I32_XMM
        case 512:
            return _SCHEDULE_VECTOR_I32_ZMM
        case _:
            raise ValueError(f"unsupported x86 i32 vector width {vector_bit_width}")


def _vector_f32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_F32_XMM
        case 512:
            return _SCHEDULE_VECTOR_F32_ZMM
        case _:
            raise ValueError(f"unsupported x86 f32 vector width {vector_bit_width}")


def _vector_fma_f32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_FMA_F32_XMM
        case 512:
            return _SCHEDULE_VECTOR_FMA_F32_ZMM
        case _:
            raise ValueError(f"unsupported x86 f32 FMA vector width {vector_bit_width}")


def _vector_compare_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_COMPARE_XMM
        case 512:
            return _SCHEDULE_VECTOR_COMPARE_ZMM
        case _:
            raise ValueError(f"unsupported x86 compare vector width {vector_bit_width}")


def _memory_load_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_MEMORY_LOAD_XMM
        case 512:
            return _SCHEDULE_MEMORY_LOAD_ZMM
        case _:
            raise ValueError(f"unsupported x86 memory vector width {vector_bit_width}")


def _memory_store_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_MEMORY_STORE_XMM
        case 512:
            return _SCHEDULE_MEMORY_STORE_ZMM
        case _:
            raise ValueError(f"unsupported x86 memory vector width {vector_bit_width}")


def _asm(
    *,
    mnemonic: str | None = None,
    native_assembly_mnemonic: str | None = None,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    named_immediates: bool = False,
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            native_assembly_mnemonic=native_assembly_mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(
                AsmImmediate(field_name, name=field_name if named_immediates else None)
                for field_name in immediates
            ),
        ),
    )


def _gpr64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _GPR64_ALT)


def _gpr64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _GPR64_ALT)


def _gpr64_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _GPR64_ALT)


def _gpr32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _GPR32_ALT)


def _gpr32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _GPR32_ALT)


def _xmm_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _XMM_ALT)


def _xmm_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _XMM_ALT)


def _zmm_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _ZMM_ALT)


def _zmm_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _ZMM_ALT)


def _vector_reg_alt(vector_bit_width: int) -> tuple[RegClassAlt, ...]:
    match vector_bit_width:
        case 128:
            return _XMM_ALT
        case 256:
            return _YMM_ALT
        case 512:
            return _ZMM_ALT
        case _:
            raise ValueError(
                f"unsupported x86 vector register width {vector_bit_width}"
            )


def _vector_asm_mnemonic(mnemonic: str, vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return f"{mnemonic}.xmm"
        case 256:
            return f"{mnemonic}.ymm"
        case 512:
            return mnemonic
        case _:
            raise ValueError(f"unsupported x86 vector width {vector_bit_width}")


def _vector_result(vector_bit_width: int, field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _vector_reg_alt(vector_bit_width))


def _vector_operand(vector_bit_width: int, field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _vector_reg_alt(vector_bit_width))


def _k_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _K_ALT)


def _k_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _K_ALT)


_DISP32_IMMEDIATE = Immediate(
    "disp32",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_IMM32_IMMEDIATE = Immediate(
    "imm32",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_IMM64_IMMEDIATE = Immediate(
    "imm64",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63) + 1,
    unsigned_max=(2**63) - 1,
)

_SHIFT32_IMMEDIATE = Immediate(
    "shift",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=31,
)

_SHIFT64_IMMEDIATE = Immediate(
    "shift",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=63,
)

_LANE_I32X4_IMMEDIATE = Immediate(
    "lane",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=3,
)

_SHUFFLE_4X2_CONTROL_IMMEDIATE = Immediate(
    "control",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=255,
)

_INSERTPS_CONTROL_IMMEDIATE = Immediate(
    "control",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=255,
)

_ADDRESS_SCALE_ENUM = "x86.address.scale"

_ADDRESS_SCALE_IMMEDIATE = Immediate(
    "scale",
    ImmediateKind.ENUM,
    enum_domain=_ADDRESS_SCALE_ENUM,
    bit_width=8,
    unsigned_max=8,
)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=512,
)

_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=512,
)


def _load_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _store_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.WRITE,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)

_GPR_DESTRUCTIVE_LHS_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)

_PACKED_DOT_NUMERIC_TAGS = {
    NUMERIC_I8: "s8",
    NUMERIC_U8: "u8",
    NUMERIC_I16: "s16",
    NUMERIC_U16: "u16",
    NUMERIC_F16: "f16",
    NUMERIC_BF16: "bf16",
    NUMERIC_I32: "i32",
    NUMERIC_F32: "f32",
}
_PACKED_DOT_ASM_FAMILY_PREFIXES = frozenset(
    (
        "avx10_2",
        "avx512_bf16",
        "avx512_vnni",
        "avx_vnni",
        "avx_vnni_int8",
        "avx_vnni_int16",
    )
)


def _packed_dot_asm_family_prefix(descriptor_key: str) -> str:
    key_parts = descriptor_key.split(".")
    if (
        len(key_parts) < 4
        or key_parts[0] != "x86"
        or key_parts[1] not in _PACKED_DOT_ASM_FAMILY_PREFIXES
    ):
        raise ValueError(
            f"x86 packed-dot descriptor key '{descriptor_key}' has no family prefix"
        )
    return key_parts[1]


def _packed_dot_asm_form(
    descriptor: PackedDotDescriptor,
    *,
    qualify_family: bool,
) -> tuple[AsmForm, ...]:
    mnemonic = _vector_asm_mnemonic(descriptor.mnemonic, descriptor.vector_bit_width)
    native_assembly_mnemonic: str | None = None
    if qualify_family:
        native_assembly_mnemonic = mnemonic
        mnemonic = f"{_packed_dot_asm_family_prefix(descriptor.key)}.{mnemonic}"
    return _asm(
        mnemonic=mnemonic,
        native_assembly_mnemonic=native_assembly_mnemonic,
        results=("dst",),
        operands=("acc", "lhs", "rhs"),
    )


def _qualify_packed_dot_descriptor_asm_forms(descriptor: Descriptor) -> Descriptor:
    key_parts = descriptor.key.split(".")
    if (
        len(key_parts) < 4
        or key_parts[0] != "x86"
        or key_parts[1] not in _PACKED_DOT_ASM_FAMILY_PREFIXES
    ):
        return descriptor
    qualified_forms = []
    for asm_form in descriptor.asm_forms:
        mnemonic = asm_form.mnemonic or descriptor.mnemonic
        if mnemonic is None:
            raise ValueError(
                f"x86 packed-dot descriptor '{descriptor.key}' has no asm mnemonic"
            )
        family_prefix = _packed_dot_asm_family_prefix(descriptor.key)
        if mnemonic.startswith(f"{family_prefix}."):
            qualified_forms.append(asm_form)
            continue
        qualified_forms.append(
            replace(
                asm_form,
                mnemonic=f"{family_prefix}.{mnemonic}",
                native_assembly_mnemonic=(
                    asm_form.native_assembly_mnemonic or mnemonic
                ),
            )
        )
    return replace(descriptor, asm_forms=tuple(qualified_forms))


def _packed_dot_semantic_tag(descriptor: PackedDotDescriptor) -> str:
    lhs_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.lhs_numeric_type]
    rhs_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.rhs_numeric_type]
    result_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.result_numeric_type]
    saturating_suffix = ".sat" if descriptor.flags & CONTRACT_FLAG_SATURATING else ""
    return (
        f"dot.{lhs_tag}{rhs_tag}.{result_tag}x"
        f"{descriptor.result_lane_count}{saturating_suffix}"
    )


def _packed_dot_descriptor(
    descriptor: PackedDotDescriptor,
    *,
    qualify_asm_mnemonic: bool = False,
) -> Descriptor:
    return Descriptor(
        key=descriptor.key,
        mnemonic=descriptor.mnemonic,
        semantic_tag=_packed_dot_semantic_tag(descriptor),
        operands=(
            _vector_result(descriptor.vector_bit_width),
            _vector_operand(descriptor.vector_bit_width, "acc"),
            _vector_operand(descriptor.vector_bit_width, "lhs"),
            _vector_operand(descriptor.vector_bit_width, "rhs"),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        asm_forms=_packed_dot_asm_form(
            descriptor,
            qualify_family=qualify_asm_mnemonic,
        ),
        schedule_class=_vector_dot_schedule_class(descriptor.vector_bit_width),
        feature_mask_words=(descriptor.required_feature_bits,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_i32_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_i32_binary_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_i32_binary_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_i32_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _vector_f32_binary_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_f32_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _scalar_f32_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _xmm_result(),
            _xmm_operand("lhs"),
            _xmm_operand("rhs"),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.xmm",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_splat_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operand: Operand,
    schedule_class: str,
) -> Descriptor:
    return _vector_splat_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operand=operand,
        schedule_class=schedule_class,
    )


def _vector_splat_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operand: Operand,
    schedule_class: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(_vector_result(vector_bit_width), operand),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("value",),
        ),
        schedule_class=schedule_class,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_i32_compare_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_i32_compare_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_i32_compare_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _k_result(),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_compare_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_f32_compare_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_f32_compare_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_f32_compare_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _k_result(),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_compare_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_mask_select_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str,
) -> Descriptor:
    return _vector_mask_select_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
    )


def _vector_mask_select_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _k_operand("mask"),
            _vector_operand(vector_bit_width, "true_value"),
            _vector_operand(vector_bit_width, "false_value"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("mask", "true_value", "false_value"),
        ),
        schedule_class=schedule_class,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
