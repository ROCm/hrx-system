# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU vector, float scalar, and index madd source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_VGPR_MATERIALIZER,
    F32_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueMaterializer,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_add_f32",
    "amdgpu.v_add_f32.lit",
    "amdgpu.v_add_f32.src0_inline",
    "amdgpu.v_sub_f32",
    "amdgpu.v_sub_f32.lit",
    "amdgpu.v_sub_f32.src0_inline",
    "amdgpu.v_mul_f32",
    "amdgpu.v_mul_f32.lit",
    "amdgpu.v_mul_f32.src0_inline",
    "amdgpu.v_min_f32",
    "amdgpu.v_min_f32.lit",
    "amdgpu.v_min_f32.src0_inline",
    "amdgpu.v_max_f32",
    "amdgpu.v_max_f32.lit",
    "amdgpu.v_max_f32.src0_inline",
    "amdgpu.v_fma_f32",
    "amdgpu.v_fmaak_f32",
    "amdgpu.v_fmamk_f32",
    "amdgpu.v_pk_fma_f32",
    "amdgpu.v_exp_f32",
    "amdgpu.v_log_f32",
    "amdgpu.v_sin_f32",
    "amdgpu.v_cos_f32",
    "amdgpu.v_floor_f32",
    "amdgpu.v_ceil_f32",
    "amdgpu.v_rndne_f32",
    "amdgpu.v_trunc_f32",
    "amdgpu.v_sqrt_f32",
    "amdgpu.v_rsq_f32",
    "amdgpu.v_rcp_f32",
    "amdgpu.v_div_scale_f32",
    "amdgpu.v_div_fmas_f32",
    "amdgpu.v_div_fixup_f32",
    "amdgpu.v_cvt_f32_f16",
    "amdgpu.v_cvt_f16_f32",
    "amdgpu.v_pk_fmac_f16",
    "amdgpu.v_pk_fma_f16",
    "amdgpu.v_pk_mad_i16",
    "amdgpu.v_pk_mad_u16",
    "amdgpu.v_cvt_f32_i32",
    "amdgpu.v_cvt_f32_u32",
    "amdgpu.v_cvt_i32_f32",
    "amdgpu.v_cvt_u32_f32",
    "amdgpu.v_add_u32",
    "amdgpu.v_add_u32.lit",
    "amdgpu.v_sub_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_mad_u32_u24",
    "amdgpu.v_mad_u32_u24.src0_lit",
    "amdgpu.v_mad_u32_u24.src1_lit",
    "amdgpu.v_mad_u32_u24.src2_lit",
    "amdgpu.v_min_i32",
    "amdgpu.v_max_i32",
    "amdgpu.v_min_u32",
    "amdgpu.v_max_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_and_b32.src0_inline",
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_or_b32",
    "amdgpu.v_or_b32.lit",
    "amdgpu.v_xor_b32",
    "amdgpu.v_xor_b32.lit",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.src0_inline",
    "amdgpu.v_lshlrev_b32.lit",
    "amdgpu.v_lshlrev_b32.src0_16_low16",
    "amdgpu.v_lshlrev_b32.vop3_imm",
    "amdgpu.v_lshl_add_u32.shift_imm",
    "amdgpu.v_bfe_u32.offset_0_width_16_low16",
    "amdgpu.v_bfe_i32.offset_width_inline",
    "amdgpu.v_bfe_u32.offset_width_inline",
    "amdgpu.v_bfi_b32.src0_lit",
    "amdgpu.v_ashrrev_i32",
    "amdgpu.v_ashrrev_i32.src0_inline",
    "amdgpu.v_ashrrev_i32.lit",
    "amdgpu.v_lshrrev_b32",
    "amdgpu.v_lshrrev_b32.src0_inline",
    "amdgpu.v_lshrrev_b32.lit",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.arithmetic",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_VEC_I32 = Vector(
    "i32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_F32 = Vector(
    "f32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_I32_STATIC = Vector(
    "i32",
    minimum_static_elements=1,
    maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_F32_STATIC = Vector(
    "f32",
    minimum_static_elements=1,
    maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_I64_STATIC = Vector(
    "i64",
    minimum_static_elements=1,
    maximum_static_elements="(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)",
)
_VEC_F64_STATIC = Vector(
    "f64",
    minimum_static_elements=1,
    maximum_static_elements="(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)",
)
_VEC_F16_PACKED_STORAGE = Vector(
    "f16",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
)
_VEC_F16_PACKED = Vector(
    "f16",
    minimum_lanes=2,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
)
_VEC_BF16_PACKED_STORAGE = Vector(
    "bf16",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
)
_VEC_I16_PACKED_STORAGE = Vector(
    "i16",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_I16_LANES",
)
_VEC_I16_PACKED = Vector(
    "i16",
    minimum_lanes=2,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_I16_LANES",
)
_VEC_I8_PACKED = Vector(
    "i8",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_I8_LANES",
)
_VCC_MASK = Vector("i1", lanes=1)
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_F32_ABS_MASK = 0x7FFFFFFF
_F32_ONE_BITS = 0x3F800000
_F32_SIGN_MASK = 0x80000000
_BF16_ROUND_BIAS = 0x7FFF
_SOURCE_INLINE_F32_VALUES = (
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    4.0,
    -4.0,
    0.15915494,
)

_VEC_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<i32>",
    constraint_key="amdgpu.arithmetic.vector_i32",
)
_VEC_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<f32>",
    constraint_key="amdgpu.arithmetic.vector_f32",
)
_VEC_I64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<i64>",
    constraint_key="amdgpu.arithmetic.vector_i64",
)
_VEC_F64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<f64>",
    constraint_key="amdgpu.arithmetic.vector_f64",
)
_VEC_F16_PACKED_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<f16>",
    constraint_key="amdgpu.arithmetic.vector_f16_packed",
)
_VEC_BF16_PACKED_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<bf16>",
    constraint_key="amdgpu.arithmetic.vector_bf16_packed",
)
_VEC_I16_PACKED_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<i16>",
    constraint_key="amdgpu.arithmetic.vector_i16_packed",
)
_VEC_I8_PACKED_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="vector<i8>",
    constraint_key="amdgpu.arithmetic.vector_i8_packed",
)
_VEC_I16_PACKED_EVEN_LANES_DIAGNOSTIC = GuardDiagnostic(
    subject_role="lane-count",
    subject_name="vector<i16>",
    constraint_key="amdgpu.arithmetic.vector_i16_packed_even_lanes",
)
_VEC_F32_PACKED_EVEN_LANES_DIAGNOSTIC = GuardDiagnostic(
    subject_role="lane-count",
    subject_name="vector<f32>",
    constraint_key="amdgpu.arithmetic.vector_f32_packed_even_lanes",
)
_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i32",
    constraint_key="amdgpu.arithmetic.i32",
)
_I64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i64",
    constraint_key="amdgpu.arithmetic.i64",
)
_I8_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i8",
    constraint_key="amdgpu.arithmetic.i8",
)
_I16_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="i16",
    constraint_key="amdgpu.arithmetic.i16",
)
_F16_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="f16",
    constraint_key="amdgpu.arithmetic.f16",
)
_BF16_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="bf16",
    constraint_key="amdgpu.arithmetic.bf16",
)
_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="f32",
    constraint_key="amdgpu.arithmetic.f32",
)
_F64_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="f64",
    constraint_key="amdgpu.arithmetic.f64",
)
_INDEX_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="index",
    constraint_key="amdgpu.index.scalar",
)
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="u32",
    constraint_key="amdgpu.address.u32",
)
_ADDRESS_U24_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="u24",
    constraint_key="amdgpu.address.u24",
)
_ADDRESS_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-literal",
    subject_name="i64",
    constraint_key="amdgpu.address.exact_i64",
)
_ADDRESS_POWER_OF_TWO_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-scale",
    subject_name="i64",
    constraint_key="amdgpu.address.exact_power_of_two_i64",
)
_ADDRESS_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="materializer",
    subject_name="address-vgpr",
    constraint_key="amdgpu.address.vgpr_materializer",
)
_F32_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="materializer",
    subject_name="f32-vgpr",
    constraint_key="amdgpu.arithmetic.f32_vgpr_materializer",
)
_ADDRESS_SGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="sgpr",
    constraint_key="amdgpu.address.sgpr",
)
_RESULT_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="vgpr",
    constraint_key="amdgpu.arithmetic.result_vgpr",
)
_LITERAL_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="i64",
    constraint_key="amdgpu.literal.exact_i64",
)
_LITERAL_EXACT_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="f32",
    constraint_key="amdgpu.literal.exact_f32",
)
_LITERAL_I32_BITS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal-bits",
    subject_name="i32",
    constraint_key="amdgpu.literal.i32_bits",
)
_BITFIELD_OFFSET_DIAGNOSTIC = GuardDiagnostic(
    subject_role="bitfield-offset",
    subject_name="i32",
    constraint_key="amdgpu.bitfield.offset_i32",
)
_BITFIELD_WIDTH_DIAGNOSTIC = GuardDiagnostic(
    subject_role="bitfield-width",
    subject_name="i32",
    constraint_key="amdgpu.bitfield.width_i32",
)
_PACKED_INTEGER_WIDTH_DIAGNOSTIC = GuardDiagnostic(
    subject_role="field-width",
    subject_name="i32",
    constraint_key="amdgpu.packed_integer.width_i32",
)
_PACKED_INTEGER_PAYLOAD_FROM_LANES_DIAGNOSTIC = GuardDiagnostic(
    subject_role="packed-integer-storage",
    subject_name="vector.bitpack",
    constraint_key="amdgpu.packed_integer.payload_from_lanes",
)
_PACKED_INTEGER_LANES_FROM_PAYLOAD_DIAGNOSTIC = GuardDiagnostic(
    subject_role="packed-integer-storage",
    subject_name="vector.bitunpack",
    constraint_key="amdgpu.packed_integer.lanes_from_payload",
)
_VECTOR_EXTRACT_SHAPE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="shape",
    subject_name="vector.extract",
    constraint_key="amdgpu.arithmetic.vector_extract_shape",
)
_VECTOR_BF16_CONVERSION_SHAPE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="shape",
    subject_name="vector.bf16_conversion",
    constraint_key="amdgpu.arithmetic.vector_bf16_conversion_shape",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _type_diagnostic(type_pattern: TypePattern) -> GuardDiagnostic:
    if type_pattern in (_VEC_I32, _VEC_I32_STATIC):
        return _VEC_I32_DIAGNOSTIC
    if type_pattern in (_VEC_F32, _VEC_F32_STATIC):
        return _VEC_F32_DIAGNOSTIC
    if type_pattern == _VEC_I64_STATIC:
        return _VEC_I64_DIAGNOSTIC
    if type_pattern == _VEC_F64_STATIC:
        return _VEC_F64_DIAGNOSTIC
    if type_pattern in (_VEC_F16_PACKED, _VEC_F16_PACKED_STORAGE):
        return _VEC_F16_PACKED_DIAGNOSTIC
    if type_pattern == _VEC_BF16_PACKED_STORAGE:
        return _VEC_BF16_PACKED_DIAGNOSTIC
    if type_pattern in (_VEC_I16_PACKED, _VEC_I16_PACKED_STORAGE):
        return _VEC_I16_PACKED_DIAGNOSTIC
    if type_pattern == _VEC_I8_PACKED:
        return _VEC_I8_PACKED_DIAGNOSTIC
    if type_pattern == _I32:
        return _I32_DIAGNOSTIC
    if type_pattern == _I64:
        return _I64_DIAGNOSTIC
    if type_pattern == _I8:
        return _I8_DIAGNOSTIC
    if type_pattern == _I16:
        return _I16_DIAGNOSTIC
    if type_pattern == _F16:
        return _F16_DIAGNOSTIC
    if type_pattern == _BF16:
        return _BF16_DIAGNOSTIC
    if type_pattern == _F32:
        return _F32_DIAGNOSTIC
    if type_pattern == _F64:
        return _F64_DIAGNOSTIC
    if type_pattern == _INDEX:
        return _INDEX_DIAGNOSTIC
    raise ValueError(f"unknown AMDGPU arithmetic type pattern: {type_pattern!r}")


def _value_type(field: str, type_pattern: TypePattern) -> Guard:
    return Guard.value_type(
        field,
        type_pattern,
        diagnostic=_type_diagnostic(type_pattern),
    )


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(_value_type(field, type_pattern) for field in fields)


def _f32_vgpr_operand(field: str) -> ValueRef:
    return _materialized_operand(field, F32_VGPR_MATERIALIZER)


def _bitcast_alias_rule(
    input_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=scalar_conversion.scalar_bitcast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            _value_type("input", input_type),
            _value_type("result", result_type),
        ),
    )


def _emit_form(type_pattern: TypePattern) -> DescriptorEmitForm:
    if type_pattern.kind == "vector":
        return DescriptorEmitForm.PER_LANE
    return DescriptorEmitForm.OP


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    f32_operands: bool = False,
    f32_rhs: bool = False,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = {
        descriptor_lhs: _f32_vgpr_operand(source_lhs)
        if f32_operands
        else ValueRef.operand(source_lhs),
        descriptor_rhs: _f32_vgpr_operand(source_rhs)
        if f32_operands or f32_rhs
        else ValueRef.operand(source_rhs),
    }
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards((source_lhs, source_rhs, "result"), type_pattern),
            *extra_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _unary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_neg_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_xor_b32.lit")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _F32_SIGN_MASK},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_abs_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _F32_ABS_MASK},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_copysign_rule(
    source_op: Op,
    type_pattern: TypePattern,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_bfi_b32.src0_lit")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "insert": _f32_vgpr_operand("lhs"),
                    "base": _f32_vgpr_operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _F32_ABS_MASK},
                form=_emit_form(type_pattern),
            ),
        ),
    )


_BITFIELD_INLINE_MASK_WIDTHS = (1, 2, 3, 4, 5, 6)


def _bitfield_low_mask(width: int) -> int:
    return 0xFFFFFFFF if width == 32 else (1 << width) - 1


def _bitfield_attr_guards(
    *,
    offset_min: int,
    offset_max: int,
    width_min: int,
    width_max: int,
) -> tuple[Guard, ...]:
    return (
        Guard.i64_range(
            "offset",
            offset_min,
            offset_max,
            diagnostic=_BITFIELD_OFFSET_DIAGNOSTIC,
        ),
        Guard.i64_range(
            "width",
            width_min,
            width_max,
            diagnostic=_BITFIELD_WIDTH_DIAGNOSTIC,
        ),
    )


def _packed_integer_width_guard(maximum_width: int) -> Guard:
    return Guard.i64_range(
        "width",
        1,
        maximum_width,
        diagnostic=_PACKED_INTEGER_WIDTH_DIAGNOSTIC,
    )


def _vector_bitpack_recipe_rule() -> RecipeRule:
    return RecipeRule(
        source_op=vector.vector_bitpack,
        guards=(
            _value_type("source", _VEC_I32),
            _value_type("result", _VEC_I8_PACKED),
            _packed_integer_width_guard(8),
            Guard.value_packed_integer_payload_from_lanes(
                "source",
                "result",
                "width",
                storage_unit_bit_count=32,
                storage_payload_multiple=32,
                diagnostic=_PACKED_INTEGER_PAYLOAD_FROM_LANES_DIAGNOSTIC,
            ),
        ),
    )


def _vector_bitunpack_recipe_rule(
    source_op: Op,
    result_type: TypePattern,
    *,
    maximum_width: int,
) -> RecipeRule:
    return RecipeRule(
        source_op=source_op,
        guards=(
            _packed_integer_width_guard(maximum_width),
            Guard.value_packed_integer_lanes_from_payload(
                "source",
                "result",
                "width",
                storage_unit_bit_count=32,
                maximum_storage_unit_count=16,
                maximum_lane_count=32,
                diagnostic=_PACKED_INTEGER_LANES_FROM_PAYLOAD_DIAGNOSTIC,
            ),
            _value_type("result", result_type),
        ),
    )


def _vector_packed_integer_recipe_rules() -> tuple[RecipeRule, ...]:
    return (
        _vector_bitpack_recipe_rule(),
        _vector_bitunpack_recipe_rule(
            vector.vector_bitunpacku,
            _VEC_I32,
            maximum_width=32,
        ),
        _vector_bitunpack_recipe_rule(
            vector.vector_bitunpacku,
            _VEC_I8_PACKED,
            maximum_width=8,
        ),
        _vector_bitunpack_recipe_rule(
            vector.vector_bitunpacks,
            _VEC_I32,
            maximum_width=32,
        ),
        _vector_bitunpack_recipe_rule(
            vector.vector_bitunpacks,
            _VEC_I8_PACKED,
            maximum_width=8,
        ),
    )


def _vector_extract_recipe_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> RecipeRule:
    return RecipeRule(
        source_op=vector.vector_extract,
        guards=(
            _value_type("source", source_type),
            _value_type("result", result_type),
            Guard.vector_extract_shape(
                "source",
                "result",
                "static_indices",
                diagnostic=_VECTOR_EXTRACT_SHAPE_DIAGNOSTIC,
            ),
        ),
    )


def _vector_extract_recipe_rules() -> tuple[RecipeRule, ...]:
    full_width_pairs = (
        (_VEC_I32_STATIC, _I32),
        (_VEC_F32_STATIC, _F32),
        (_VEC_I64_STATIC, _I64),
        (_VEC_F64_STATIC, _F64),
        (_VEC_I32_STATIC, _VEC_I32_STATIC),
        (_VEC_F32_STATIC, _VEC_F32_STATIC),
        (_VEC_I64_STATIC, _VEC_I64_STATIC),
        (_VEC_F64_STATIC, _VEC_F64_STATIC),
    )
    packed_scalar_pairs = (
        (_VEC_F16_PACKED_STORAGE, _F16),
        (_VEC_BF16_PACKED_STORAGE, _BF16),
        (_VEC_I16_PACKED_STORAGE, _I16),
        (_VEC_I8_PACKED, _I8),
    )
    return tuple(
        _vector_extract_recipe_rule(source_type, result_type)
        for source_type, result_type in (*full_width_pairs, *packed_scalar_pairs)
    )


def _vector_bf16_conversion_recipe_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
) -> RecipeRule:
    return RecipeRule(
        source_op=source_op,
        guards=(
            _value_type("input", input_type),
            _value_type("result", result_type),
            Guard.value_static_element_count_eq(
                "input",
                "result",
                diagnostic=_VECTOR_BF16_CONVERSION_SHAPE_DIAGNOSTIC,
            ),
        ),
    )


def _vector_bf16_conversion_recipe_rules() -> tuple[RecipeRule, ...]:
    return (
        _vector_bf16_conversion_recipe_rule(
            vector.vector_extf,
            _VEC_BF16_PACKED_STORAGE,
            _VEC_F32_STATIC,
        ),
        _vector_bf16_conversion_recipe_rule(
            vector.vector_fptrunc,
            _VEC_F32_STATIC,
            _VEC_BF16_PACKED_STORAGE,
        ),
    )


def _vector_bitfield_extract_alias_rule(source_op: Op) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=source_op,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=0,
                width_min=32,
                width_max=32,
            ),
        ),
    )


def _vector_bitfield_extract_bfe_rule(
    source_op: Op,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=31,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "offset": AttrProject.direct("offset"),
                    "width": AttrProject.direct("width"),
                },
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_bitfield_extractu_offset0_inline_rule(width: int) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_and_b32.src0_inline")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extractu,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=0,
                width_min=width,
                width_max=width,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"rhs": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _bitfield_low_mask(width)},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_bitfield_extractu_offset0_literal_rule() -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extractu,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=0,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"rhs": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.i64_low_bit_mask("width")},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_bitfield_extractu_shift_inline_mask_rule(width: int) -> DescriptorRule:
    shift = _descriptor("amdgpu.v_lshrrev_b32.src0_inline")
    mask = _descriptor("amdgpu.v_and_b32.src0_inline")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extractu,
        descriptor=shift,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=1,
                offset_max=31,
                width_min=width,
                width_max=width,
            ),
            Guard.descriptor_available(shift),
            Guard.descriptor_available(mask),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": ValueRef.operand("source")},
                results={"dst": ValueRef.temporary("shifted")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.direct("offset")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=mask,
                operands={"rhs": ValueRef.temporary("shifted")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _bitfield_low_mask(width)},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _vector_bitfield_extractu_shift_literal_mask_rule() -> DescriptorRule:
    shift = _descriptor("amdgpu.v_lshrrev_b32.src0_inline")
    mask = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extractu,
        descriptor=shift,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=1,
                offset_max=31,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(shift),
            Guard.descriptor_available(mask),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": ValueRef.operand("source")},
                results={"dst": ValueRef.temporary("shifted")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.direct("offset")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=mask,
                operands={"rhs": ValueRef.temporary("shifted")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.i64_low_bit_mask("width")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _vector_bitfield_extracts_top_aligned_rule(width: int) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_ashrrev_i32.src0_inline")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extracts,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=32 - width,
                offset_max=32 - width,
                width_min=width,
                width_max=width,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": 32 - width},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_bitfield_extracts_shift_rule() -> DescriptorRule:
    shift_left = _descriptor("amdgpu.v_lshlrev_b32.src0_inline")
    shift_right = _descriptor("amdgpu.v_ashrrev_i32.src0_inline")
    return DescriptorRule(
        source_op=vector.vector_bitfield_extracts,
        descriptor=shift_left,
        guards=(
            *_typed_guards(("source", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=31,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(shift_left),
            Guard.descriptor_available(shift_right),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift_left,
                operands={"value": ValueRef.operand("source")},
                results={"dst": ValueRef.temporary("shifted_left")},
                result_types={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": AttrProject.i64_literal_minus_attrs(
                        "offset",
                        other_source_attr="width",
                        literal=32,
                    )
                },
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=shift_right,
                operands={"value": ValueRef.temporary("shifted_left")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": AttrProject.i64_literal_minus_attr(
                        "width",
                        literal=32,
                    )
                },
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _vector_bitfield_insert_alias_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_bitfield_insert,
        source=ValueRef.operand("field"),
        result=ValueRef.result("result"),
        guards=(
            *_typed_guards(("field", "base", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=0,
                width_min=32,
                width_max=32,
            ),
        ),
    )


def _vector_bitfield_insert_bfi_offset0_rule() -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_bfi_b32.src0_lit")
    return DescriptorRule(
        source_op=vector.vector_bitfield_insert,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("field", "base", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=0,
                offset_max=0,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "insert": ValueRef.operand("field"),
                    "base": ValueRef.operand("base"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.i64_low_bit_mask("width")},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_bitfield_insert_bfi_shift_rule() -> DescriptorRule:
    shift = _descriptor("amdgpu.v_lshlrev_b32.src0_inline")
    insert = _descriptor("amdgpu.v_bfi_b32.src0_lit")
    return DescriptorRule(
        source_op=vector.vector_bitfield_insert,
        descriptor=shift,
        guards=(
            *_typed_guards(("field", "base", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=1,
                offset_max=31,
                width_min=1,
                width_max=31,
            ),
            Guard.descriptor_available(shift),
            Guard.descriptor_available(insert),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": ValueRef.operand("field")},
                results={"dst": ValueRef.temporary("shifted_field")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.direct("offset")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=insert,
                operands={
                    "insert": ValueRef.temporary("shifted_field"),
                    "base": ValueRef.operand("base"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": AttrProject.i64_shifted_low_bit_mask(
                        "width",
                        offset_attr="offset",
                    )
                },
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _vector_bitfield_insert_fallback_rule(
    *,
    offset_min: int,
    offset_max: int,
    inline_field_mask_width: int | None,
) -> DescriptorRule:
    field_mask = _descriptor(
        "amdgpu.v_and_b32.src0_inline"
        if inline_field_mask_width is not None
        else "amdgpu.v_and_b32.lit"
    )
    shift = _descriptor("amdgpu.v_lshlrev_b32.src0_inline")
    clear_base = _descriptor("amdgpu.v_and_b32.lit")
    merge = _descriptor("amdgpu.v_or_b32")
    has_shift = offset_max != 0
    width_min = inline_field_mask_width if inline_field_mask_width is not None else 1
    width_max = inline_field_mask_width if inline_field_mask_width is not None else 31
    shifted_field = (
        ValueRef.temporary("shifted_field")
        if has_shift
        else ValueRef.temporary("field_low_bits")
    )
    shift_emit = (
        (
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": ValueRef.temporary("field_low_bits")},
                results={"dst": shifted_field},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": AttrProject.direct("offset")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        )
        if has_shift
        else ()
    )
    return DescriptorRule(
        source_op=vector.vector_bitfield_insert,
        descriptor=field_mask,
        guards=(
            *_typed_guards(("field", "base", "result"), _VEC_I32),
            *_bitfield_attr_guards(
                offset_min=offset_min,
                offset_max=offset_max,
                width_min=width_min,
                width_max=width_max,
            ),
            Guard.descriptor_available(field_mask),
            *((Guard.descriptor_available(shift),) if has_shift else ()),
            Guard.descriptor_available(clear_base),
            Guard.descriptor_available(merge),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=field_mask,
                operands={"rhs": ValueRef.operand("field")},
                results={"dst": ValueRef.temporary("field_low_bits")},
                result_types={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": _bitfield_low_mask(inline_field_mask_width)
                    if inline_field_mask_width is not None
                    else AttrProject.i64_low_bit_mask("width")
                },
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            *shift_emit,
            EmitDescriptorOp(
                descriptor=clear_base,
                operands={"rhs": ValueRef.operand("base")},
                results={"dst": ValueRef.temporary("cleared_base")},
                result_types={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": AttrProject.i64_shifted_low_bit_clear_mask(
                        "width",
                        offset_attr="offset",
                    )
                },
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=merge,
                operands={
                    "lhs": ValueRef.temporary("cleared_base"),
                    "rhs": shifted_field,
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _vector_bitfield_rules() -> tuple[ContractCase, ...]:
    rules: list[ContractCase] = [
        _vector_bitfield_extract_alias_rule(vector.vector_bitfield_extractu),
        *(
            _vector_bitfield_extractu_offset0_inline_rule(width)
            for width in _BITFIELD_INLINE_MASK_WIDTHS
        ),
        _vector_bitfield_extractu_offset0_literal_rule(),
        _vector_bitfield_extract_bfe_rule(
            vector.vector_bitfield_extractu,
            "amdgpu.v_bfe_u32.offset_width_inline",
        ),
        *(
            _vector_bitfield_extractu_shift_inline_mask_rule(width)
            for width in _BITFIELD_INLINE_MASK_WIDTHS
        ),
        _vector_bitfield_extractu_shift_literal_mask_rule(),
        _vector_bitfield_extract_alias_rule(vector.vector_bitfield_extracts),
        *(_vector_bitfield_extracts_top_aligned_rule(width) for width in range(1, 32)),
        _vector_bitfield_extract_bfe_rule(
            vector.vector_bitfield_extracts,
            "amdgpu.v_bfe_i32.offset_width_inline",
        ),
        _vector_bitfield_extracts_shift_rule(),
        _vector_bitfield_insert_alias_rule(),
        _vector_bitfield_insert_bfi_offset0_rule(),
        _vector_bitfield_insert_bfi_shift_rule(),
    ]
    rules.extend(
        _vector_bitfield_insert_fallback_rule(
            offset_min=0,
            offset_max=0,
            inline_field_mask_width=width,
        )
        for width in _BITFIELD_INLINE_MASK_WIDTHS
    )
    rules.append(
        _vector_bitfield_insert_fallback_rule(
            offset_min=0,
            offset_max=0,
            inline_field_mask_width=None,
        )
    )
    rules.extend(
        _vector_bitfield_insert_fallback_rule(
            offset_min=1,
            offset_max=31,
            inline_field_mask_width=width,
        )
        for width in _BITFIELD_INLINE_MASK_WIDTHS
    )
    rules.append(
        _vector_bitfield_insert_fallback_rule(
            offset_min=1,
            offset_max=31,
            inline_field_mask_width=None,
        )
    )
    return tuple(rules)


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    f32_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": _f32_vgpr_operand("a")
                    if f32_operands
                    else ValueRef.operand("a"),
                    "b": _f32_vgpr_operand("b")
                    if f32_operands
                    else ValueRef.operand("b"),
                    "c": _f32_vgpr_operand("c")
                    if f32_operands
                    else ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _divf_arcp_one_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    reciprocal = _descriptor("amdgpu.v_rcp_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=reciprocal,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.instance_flags_has_all("fastmath", "arcp"),
            Guard.value_f64_equals("lhs", 1.0),
            Guard.descriptor_available(reciprocal),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=reciprocal,
                operands={
                    "input": _f32_vgpr_operand("rhs")
                    if f32_operand
                    else ValueRef.operand("rhs")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _divf_arcp_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operands: bool = False,
) -> DescriptorRule:
    reciprocal = _descriptor("amdgpu.v_rcp_f32")
    multiply = _descriptor("amdgpu.v_mul_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=reciprocal,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.instance_flags_has_all("fastmath", "arcp"),
            Guard.descriptor_available(reciprocal),
            Guard.descriptor_available(multiply),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=reciprocal,
                operands={
                    "input": _f32_vgpr_operand("rhs")
                    if f32_operands
                    else ValueRef.operand("rhs")
                },
                results={"dst": ValueRef.temporary("reciprocal")},
                result_types={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": _f32_vgpr_operand("lhs")
                    if f32_operands
                    else ValueRef.operand("lhs"),
                    "rhs": ValueRef.temporary("reciprocal"),
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _divf_exact_rule(source_op: Op, type_pattern: TypePattern) -> DescriptorRule:
    scale = _descriptor("amdgpu.v_div_scale_f32")
    reciprocal = _descriptor("amdgpu.v_rcp_f32")
    negate = _descriptor("amdgpu.v_xor_b32.lit")
    multiply = _descriptor("amdgpu.v_mul_f32")
    fma = _descriptor("amdgpu.v_fma_f32")
    fmaak = _descriptor("amdgpu.v_fmaak_f32")
    fmas = _descriptor("amdgpu.v_div_fmas_f32")
    fixup = _descriptor("amdgpu.v_div_fixup_f32")
    result_type = _F32
    numerator = _f32_vgpr_operand("lhs")
    denominator = _f32_vgpr_operand("rhs")
    return DescriptorRule(
        source_op=source_op,
        descriptor=scale,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_materializable(
                "lhs",
                F32_VGPR_MATERIALIZER.name,
                diagnostic=_F32_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "rhs",
                F32_VGPR_MATERIALIZER.name,
                diagnostic=_F32_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(scale),
            Guard.descriptor_available(reciprocal),
            Guard.descriptor_available(negate),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(fma),
            Guard.descriptor_available(fmaak),
            Guard.descriptor_available(fmas),
            Guard.descriptor_available(fixup),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=scale,
                operands={
                    "value": denominator,
                    "denominator": denominator,
                    "numerator": numerator,
                },
                results={
                    "dst": ValueRef.temporary("scaled_denominator"),
                    "mask": ValueRef.temporary("denominator_scale_mask"),
                },
                result_types={
                    "dst": result_type,
                    "mask": _VCC_MASK,
                },
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=reciprocal,
                operands={"input": ValueRef.temporary("scaled_denominator")},
                results={"dst": ValueRef.temporary("reciprocal")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=negate,
                operands={"rhs": ValueRef.temporary("scaled_denominator")},
                results={"dst": ValueRef.temporary("neg_scaled_denominator")},
                result_types={"dst": result_type},
                immediates={"imm32": _F32_SIGN_MASK},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fmaak,
                operands={
                    "a": ValueRef.temporary("neg_scaled_denominator"),
                    "b": ValueRef.temporary("reciprocal"),
                },
                results={"dst": ValueRef.temporary("reciprocal_error")},
                result_types={"dst": result_type},
                immediates={"imm32": _F32_ONE_BITS},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("reciprocal_error"),
                    "b": ValueRef.temporary("reciprocal"),
                    "c": ValueRef.temporary("reciprocal"),
                },
                results={"dst": ValueRef.temporary("refined_reciprocal")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=scale,
                operands={
                    "value": numerator,
                    "denominator": denominator,
                    "numerator": numerator,
                },
                results={
                    "dst": ValueRef.temporary("scaled_numerator"),
                    "mask": ValueRef.temporary("scale_mask"),
                },
                result_types={
                    "dst": result_type,
                    "mask": _VCC_MASK,
                },
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.temporary("scaled_numerator"),
                    "rhs": ValueRef.temporary("refined_reciprocal"),
                },
                results={"dst": ValueRef.temporary("quotient0")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("neg_scaled_denominator"),
                    "b": ValueRef.temporary("quotient0"),
                    "c": ValueRef.temporary("scaled_numerator"),
                },
                results={"dst": ValueRef.temporary("remainder0")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("remainder0"),
                    "b": ValueRef.temporary("refined_reciprocal"),
                    "c": ValueRef.temporary("quotient0"),
                },
                results={"dst": ValueRef.temporary("quotient1")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("neg_scaled_denominator"),
                    "b": ValueRef.temporary("quotient1"),
                    "c": ValueRef.temporary("scaled_numerator"),
                },
                results={"dst": ValueRef.temporary("remainder1")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fmas,
                operands={
                    "a": ValueRef.temporary("remainder1"),
                    "b": ValueRef.temporary("refined_reciprocal"),
                    "c": ValueRef.temporary("quotient1"),
                    "scale_mask": ValueRef.temporary("scale_mask"),
                },
                results={"dst": ValueRef.temporary("quotient2")},
                result_types={"dst": result_type},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=fixup,
                operands={
                    "quotient": ValueRef.temporary("quotient2"),
                    "denominator": denominator,
                    "numerator": numerator,
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _cast_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    f32_input: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            _value_type("input", input_type),
            _value_type("result", result_type),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": _f32_vgpr_operand("input")
                    if f32_input
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(result_type),
            ),
        ),
    )


def _bf16_extf_rule() -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_lshlrev_b32.src0_16_low16")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=descriptor,
        guards=(
            _value_type("input", _BF16),
            _value_type("result", _F32),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": 16},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _bf16_fptrunc_rule() -> DescriptorRule:
    shift_down = _descriptor("amdgpu.v_lshrrev_b32.lit")
    and_bits = _descriptor("amdgpu.v_and_b32.lit")
    add_literal = _descriptor("amdgpu.v_add_u32.lit")
    add = _descriptor("amdgpu.v_add_u32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=shift_down,
        guards=(
            _value_type("input", _F32),
            _value_type("result", _BF16),
            Guard.descriptor_available(shift_down),
            Guard.descriptor_available(and_bits),
            Guard.descriptor_available(add_literal),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift_down,
                operands={"value": _f32_vgpr_operand("input")},
                results={"dst": ValueRef.temporary("upper")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": 16},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=and_bits,
                operands={"rhs": ValueRef.temporary("upper")},
                results={"dst": ValueRef.temporary("lsb")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": 1},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=add_literal,
                operands={"rhs": ValueRef.temporary("lsb")},
                results={"dst": ValueRef.temporary("bias")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": _BF16_ROUND_BIAS},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": _f32_vgpr_operand("input"),
                    "rhs": ValueRef.temporary("bias"),
                },
                results={"dst": ValueRef.temporary("rounded")},
                result_types={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=shift_down,
                operands={"value": ValueRef.temporary("rounded")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": 16},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _literal_binary_rule(
    source_op: Op,
    descriptor_key: str,
    *,
    literal_source: str,
    nonliteral_source: str,
    descriptor_operand: str = "rhs",
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _VEC_I32),
            Guard.value_exact_i64(
                literal_source,
                diagnostic=_LITERAL_EXACT_DIAGNOSTIC,
            ),
            Guard.value_signed_bit_count(
                literal_source,
                32,
                diagnostic=_LITERAL_I32_BITS_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={descriptor_operand: ValueRef.operand(nonliteral_source)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.i32_as_u32_bits(literal_source),
                },
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _f32_literal_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    literal_source: str,
    nonliteral_source: str,
    f32_operand: bool = False,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_exact_f64(
                literal_source,
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            *extra_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand(nonliteral_source)
                    if f32_operand
                    else ValueRef.operand(nonliteral_source)
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.f64_as_f32_bits(literal_source),
                },
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_inline_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    literal_source: str,
    nonliteral_source: str,
    literal_value: float,
    f32_operand: bool = False,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_f64_equals(
                literal_source,
                literal_value,
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            *extra_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand(nonliteral_source)
                    if f32_operand
                    else ValueRef.operand(nonliteral_source)
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.f64_as_f32_bits(literal_source),
                },
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _index_madd_power_of_two_rule(
    *,
    scale_source: str,
    value_source: str,
    literal_addend: bool,
    preserve_value_register: bool,
) -> DescriptorRule:
    shift = _descriptor(
        "amdgpu.v_lshlrev_b32.vop3_imm"
        if preserve_value_register
        else "amdgpu.v_lshlrev_b32.lit"
    )
    add = _descriptor("amdgpu.v_add_u32.lit" if literal_addend else "amdgpu.v_add_u32")
    value_guard = (
        Guard.low_value_register_class(
            value_source,
            "amdgpu.sgpr",
            diagnostic=_ADDRESS_SGPR_DIAGNOSTIC,
        )
        if preserve_value_register
        else Guard.value_materializable(
            value_source,
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )
    )
    value_operand = (
        ValueRef.operand(value_source)
        if preserve_value_register
        else _materialized_operand(value_source, ADDRESS_VGPR_MATERIALIZER)
    )
    addend_guards = (
        (
            Guard.value_exact_i64("c", diagnostic=_ADDRESS_EXACT_DIAGNOSTIC),
            Guard.value_unsigned_bit_count(
                "c",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
        )
        if literal_addend
        else (
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
        )
    )
    add_emit = (
        EmitDescriptorOp(
            descriptor=add,
            operands={"rhs": ValueRef.temporary("scaled")},
            results={"dst": ValueRef.result("result")},
            immediates={"imm32": ValueProject.exact_i64("c")},
            form=DescriptorEmitForm.OP,
        )
        if literal_addend
        else EmitDescriptorOp(
            descriptor=add,
            operands={
                "lhs": ValueRef.temporary("scaled"),
                "rhs": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
            },
            results={"dst": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=shift,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                scale_source,
                diagnostic=_ADDRESS_POWER_OF_TWO_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                scale_source,
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            value_guard,
            *addend_guards,
            Guard.descriptor_available(shift),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": value_operand},
                results={"dst": ValueRef.temporary("scaled")},
                result_types={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.exact_i64_log2(scale_source),
                },
                form=DescriptorEmitForm.OP,
            ),
            add_emit,
        ),
    )


def _index_madd_power_of_two_lshl_add_rule(
    *,
    scale_source: str,
    value_source: str,
    preserve_value_register: bool,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_lshl_add_u32.shift_imm")
    value_guard = (
        Guard.low_value_register_class(
            value_source,
            "amdgpu.sgpr",
            diagnostic=_ADDRESS_SGPR_DIAGNOSTIC,
        )
        if preserve_value_register
        else Guard.value_materializable(
            value_source,
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )
    )
    value_operand = (
        ValueRef.operand(value_source)
        if preserve_value_register
        else _materialized_operand(value_source, ADDRESS_VGPR_MATERIALIZER)
    )
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                scale_source,
                diagnostic=_ADDRESS_POWER_OF_TWO_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                scale_source,
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            value_guard,
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "value": value_operand,
                    "addend": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"shift": ValueProject.exact_i64_log2(scale_source)},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_literal_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.v_mul_lo_u32")
    add = _descriptor("amdgpu.v_add_u32.lit")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "a",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "b",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_exact_i64("c", diagnostic=_ADDRESS_EXACT_DIAGNOSTIC),
            Guard.value_unsigned_bit_count(
                "c",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            _index_madd_product_emit(multiply),
            EmitDescriptorOp(
                descriptor=add,
                operands={"rhs": ValueRef.temporary("product")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64("c")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_u24_mad_rule(
    *,
    preserved_source: str | None,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_mad_u32_u24")

    def multiply_guard(field: str) -> Guard:
        if preserved_source == field:
            return Guard.low_value_register_class(
                field,
                "amdgpu.sgpr",
                diagnostic=_ADDRESS_SGPR_DIAGNOSTIC,
            )
        return Guard.value_materializable(
            field,
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )

    def multiply_operand(field: str) -> ValueRef:
        if preserved_source == field:
            return ValueRef.operand(field)
        return _materialized_operand(field, ADDRESS_VGPR_MATERIALIZER)

    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                "a",
                24,
                diagnostic=_ADDRESS_U24_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                "b",
                24,
                diagnostic=_ADDRESS_U24_DIAGNOSTIC,
            ),
            multiply_guard("a"),
            multiply_guard("b"),
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": multiply_operand("a"),
                    "b": multiply_operand("b"),
                    "addend": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_u24_mad_literal_rule(
    *,
    literal_source: str,
    preserved_source: str | None,
) -> DescriptorRule:
    descriptor_by_source = {
        "a": _descriptor("amdgpu.v_mad_u32_u24.src0_lit"),
        "b": _descriptor("amdgpu.v_mad_u32_u24.src1_lit"),
        "c": _descriptor("amdgpu.v_mad_u32_u24.src2_lit"),
    }
    descriptor = descriptor_by_source[literal_source]

    def multiply_guard(field: str) -> Guard:
        if literal_source == field:
            return Guard.value_exact_i64(
                field,
                diagnostic=_ADDRESS_EXACT_DIAGNOSTIC,
            )
        if preserved_source == field:
            return Guard.low_value_register_class(
                field,
                "amdgpu.sgpr",
                diagnostic=_ADDRESS_SGPR_DIAGNOSTIC,
            )
        return Guard.value_materializable(
            field,
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )

    def multiply_operand(field: str) -> ValueRef | None:
        if literal_source == field:
            return None
        if preserved_source == field:
            return ValueRef.operand(field)
        return _materialized_operand(field, ADDRESS_VGPR_MATERIALIZER)

    def addend_guard() -> Guard:
        if literal_source == "c":
            return Guard.value_exact_i64("c", diagnostic=_ADDRESS_EXACT_DIAGNOSTIC)
        return Guard.value_materializable(
            "c",
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )

    operands: dict[str, ValueRef] = {}
    for field in ("a", "b"):
        operand = multiply_operand(field)
        if operand is not None:
            operands[field] = operand
    if literal_source != "c":
        operands["addend"] = _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER)

    extra_literal_guards: tuple[Guard, ...] = ()
    if literal_source == "c":
        extra_literal_guards = (
            Guard.value_unsigned_bit_count(
                "c",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
        )

    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                "a",
                24,
                diagnostic=_ADDRESS_U24_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                "b",
                24,
                diagnostic=_ADDRESS_U24_DIAGNOSTIC,
            ),
            multiply_guard("a"),
            multiply_guard("b"),
            addend_guard(),
            *extra_literal_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64(literal_source)},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.v_mul_lo_u32")
    add = _descriptor("amdgpu.v_add_u32")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "a",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "b",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            _index_madd_product_emit(multiply),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_product_emit(descriptor: Descriptor) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={
            "lhs": _materialized_operand("a", ADDRESS_VGPR_MATERIALIZER),
            "rhs": _materialized_operand("b", ADDRESS_VGPR_MATERIALIZER),
        },
        results={"dst": ValueRef.temporary("product")},
        result_types={"dst": ValueRef.result("result")},
        form=DescriptorEmitForm.OP,
    )


def _materialized_operand(field: str, materializer: ValueMaterializer) -> ValueRef:
    return ValueRef.operand(field, materializer=materializer.name)


def _register_class(field: str, register_class: str) -> Guard:
    return Guard.low_value_register_class(field, register_class)


def _commutative_f32_binary_rules(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> tuple[DescriptorRule, DescriptorRule]:
    return (
        _binary_rule(
            source_op,
            type_pattern,
            descriptor_key,
            descriptor_lhs="lhs",
            descriptor_rhs="rhs",
            source_lhs="rhs",
            source_rhs="lhs",
            extra_guards=(
                _register_class("rhs", "amdgpu.sgpr"),
                _register_class("lhs", "amdgpu.vgpr"),
            ),
        ),
        _binary_rule(
            source_op,
            type_pattern,
            descriptor_key,
            f32_rhs=True,
        ),
    )


def _f32_fma_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    a_register_class: str,
    b_register_class: str,
    c_register_class: str,
    materialize_c: bool,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_fma_f32")
    materializer_guards: tuple[Guard, ...] = ()
    c_operand = ValueRef.operand("c")
    if materialize_c:
        materializer_guards = (
            Guard.value_materializable(
                "c",
                F32_VGPR_MATERIALIZER.name,
                diagnostic=_F32_VGPR_DIAGNOSTIC,
            ),
        )
        c_operand = _f32_vgpr_operand("c")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            _register_class("a", a_register_class),
            _register_class("b", b_register_class),
            _register_class("c", c_register_class),
            *materializer_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": c_operand,
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_fmaak_literal_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    a_register_class: str,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_fmaak_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            _register_class("a", a_register_class),
            Guard.value_exact_f64(
                "c",
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "b",
                F32_VGPR_MATERIALIZER.name,
                diagnostic=_F32_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": _f32_vgpr_operand("b"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.f64_as_f32_bits("c"),
                },
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_fmamk_literal_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    literal_source: str,
    multiply_source: str,
    multiply_register_class: str,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_fmamk_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            _register_class(multiply_source, multiply_register_class),
            Guard.value_exact_f64(
                literal_source,
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "c",
                F32_VGPR_MATERIALIZER.name,
                diagnostic=_F32_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand(multiply_source),
                    "c": _f32_vgpr_operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.f64_as_f32_bits(literal_source),
                },
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_fma_rules(
    source_op: Op,
    type_pattern: TypePattern,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    register_classes = ("amdgpu.sgpr", "amdgpu.vgpr")
    for register_class in register_classes:
        rules.append(
            _f32_fmaak_literal_rule(
                source_op,
                type_pattern,
                a_register_class=register_class,
            )
        )
        rules.append(
            _f32_fmamk_literal_rule(
                source_op,
                type_pattern,
                literal_source="a",
                multiply_source="b",
                multiply_register_class=register_class,
            )
        )
        rules.append(
            _f32_fmamk_literal_rule(
                source_op,
                type_pattern,
                literal_source="b",
                multiply_source="a",
                multiply_register_class=register_class,
            )
        )
    for a_register_class in register_classes:
        for b_register_class in register_classes:
            for c_register_class in register_classes:
                all_sources_sgpr = (
                    a_register_class == "amdgpu.sgpr"
                    and b_register_class == "amdgpu.sgpr"
                    and c_register_class == "amdgpu.sgpr"
                )
                rules.append(
                    _f32_fma_rule(
                        source_op,
                        type_pattern,
                        a_register_class=a_register_class,
                        b_register_class=b_register_class,
                        c_register_class=c_register_class,
                        materialize_c=all_sources_sgpr,
                    )
                )
    return tuple(rules)


def _packed_f16_vector_fma_rule(
    descriptor_key: str,
    operands: dict[str, ValueRef],
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_fmaf,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _VEC_F16_PACKED),
            Guard.value_static_dim0_multiple(
                "result",
                2,
                diagnostic=_VEC_F16_PACKED_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _packed_f16_vector_fma_rules() -> tuple[DescriptorRule, ...]:
    return (
        _packed_f16_vector_fma_rule(
            "amdgpu.v_pk_fmac_f16",
            operands={
                "acc": ValueRef.operand("c"),
                "a": ValueRef.operand("a"),
                "b": ValueRef.operand("b"),
            },
        ),
        _packed_f16_vector_fma_rule(
            "amdgpu.v_pk_fma_f16",
            operands={
                "a": ValueRef.operand("a"),
                "b": ValueRef.operand("b"),
                "c": ValueRef.operand("c"),
            },
        ),
    )


def _packed_i16_vector_fmai_rule(descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_fmai,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _VEC_I16_PACKED),
            Guard.value_static_dim0_multiple(
                "result",
                2,
                diagnostic=_VEC_I16_PACKED_EVEN_LANES_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _packed_i16_vector_fmai_rules() -> tuple[DescriptorRule, ...]:
    return (
        _packed_i16_vector_fmai_rule("amdgpu.v_pk_mad_i16"),
        _packed_i16_vector_fmai_rule("amdgpu.v_pk_mad_u16"),
    )


def _packed_f32_vector_fma_rule() -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_pk_fma_f32")
    return DescriptorRule(
        source_op=vector.vector_fmaf,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _VEC_F32),
            Guard.value_static_dim0_multiple(
                "result",
                2,
                diagnostic=_VEC_F32_PACKED_EVEN_LANES_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _commutative_f32_vector_literal_rules(
    source_op: Op,
    descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    inline_descriptor_key = descriptor_key.removesuffix(".lit") + ".src0_inline"
    for literal_value in _SOURCE_INLINE_F32_VALUES:
        rules.extend(
            (
                _f32_inline_binary_rule(
                    source_op,
                    _VEC_F32,
                    inline_descriptor_key,
                    literal_source="lhs",
                    nonliteral_source="rhs",
                    literal_value=literal_value,
                    f32_operand=True,
                ),
                _f32_inline_binary_rule(
                    source_op,
                    _VEC_F32,
                    inline_descriptor_key,
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    literal_value=literal_value,
                    f32_operand=True,
                ),
            )
        )
    rules.extend(
        (
            _f32_literal_binary_rule(
                source_op,
                _VEC_F32,
                descriptor_key,
                literal_source="lhs",
                nonliteral_source="rhs",
                f32_operand=True,
            ),
            _f32_literal_binary_rule(
                source_op,
                _VEC_F32,
                descriptor_key,
                literal_source="rhs",
                nonliteral_source="lhs",
                f32_operand=True,
            ),
        )
    )
    return tuple(rules)


def _f32_vector_sub_literal_rules() -> tuple[DescriptorRule, ...]:
    return (
        *(
            _f32_inline_binary_rule(
                vector.vector_subf,
                _VEC_F32,
                "amdgpu.v_sub_f32.src0_inline",
                literal_source="lhs",
                nonliteral_source="rhs",
                literal_value=literal_value,
                f32_operand=True,
            )
            for literal_value in _SOURCE_INLINE_F32_VALUES
        ),
        _f32_literal_binary_rule(
            vector.vector_subf,
            _VEC_F32,
            "amdgpu.v_sub_f32.lit",
            literal_source="lhs",
            nonliteral_source="rhs",
            f32_operand=True,
        ),
    )


def _rules() -> tuple[ContractCase, ...]:
    rules: list[ContractCase] = []
    for source_op, descriptor_key in (
        (vector.vector_addf, "amdgpu.v_add_f32.lit"),
        (vector.vector_mulf, "amdgpu.v_mul_f32.lit"),
        (vector.vector_minnumf, "amdgpu.v_min_f32.lit"),
        (vector.vector_maxnumf, "amdgpu.v_max_f32.lit"),
    ):
        rules.extend(_commutative_f32_vector_literal_rules(source_op, descriptor_key))
    rules.extend(_f32_vector_sub_literal_rules())
    rules.extend(
        (
            *_commutative_f32_binary_rules(
                vector.vector_addf,
                _VEC_F32,
                "amdgpu.v_add_f32",
            ),
            _binary_rule(
                vector.vector_subf,
                _VEC_F32,
                "amdgpu.v_sub_f32",
                f32_rhs=True,
            ),
            *_commutative_f32_binary_rules(
                vector.vector_mulf,
                _VEC_F32,
                "amdgpu.v_mul_f32",
            ),
            _f32_neg_rule(vector.vector_negf, _VEC_F32, f32_operand=True),
            _f32_abs_rule(vector.vector_absf, _VEC_F32, f32_operand=True),
            _f32_copysign_rule(vector.vector_copysignf, _VEC_F32),
            _divf_arcp_one_rule(vector.vector_divf, _VEC_F32),
            _divf_arcp_rule(vector.vector_divf, _VEC_F32),
            _divf_exact_rule(vector.vector_divf, _VEC_F32),
            *_commutative_f32_binary_rules(
                vector.vector_minnumf,
                _VEC_F32,
                "amdgpu.v_min_f32",
            ),
            *_commutative_f32_binary_rules(
                vector.vector_maxnumf,
                _VEC_F32,
                "amdgpu.v_max_f32",
            ),
            _packed_f32_vector_fma_rule(),
            *_packed_f16_vector_fma_rules(),
            *_packed_i16_vector_fmai_rules(),
            *_vector_extract_recipe_rules(),
            *_vector_bf16_conversion_recipe_rules(),
            *_f32_fma_rules(vector.vector_fmaf, _VEC_F32),
            _unary_rule(vector.vector_exp2f, _VEC_F32, "amdgpu.v_exp_f32"),
            _unary_rule(vector.vector_log2f, _VEC_F32, "amdgpu.v_log_f32"),
            _unary_rule(vector.vector_sinturnsf, _VEC_F32, "amdgpu.v_sin_f32"),
            _unary_rule(vector.vector_costurnsf, _VEC_F32, "amdgpu.v_cos_f32"),
            _unary_rule(vector.vector_floorf, _VEC_F32, "amdgpu.v_floor_f32"),
            _unary_rule(vector.vector_ceilf, _VEC_F32, "amdgpu.v_ceil_f32"),
            _unary_rule(vector.vector_roundevenf, _VEC_F32, "amdgpu.v_rndne_f32"),
            _unary_rule(vector.vector_truncf, _VEC_F32, "amdgpu.v_trunc_f32"),
            _unary_rule(vector.vector_sqrtf, _VEC_F32, "amdgpu.v_sqrt_f32"),
            _unary_rule(vector.vector_rsqrtf, _VEC_F32, "amdgpu.v_rsq_f32"),
        )
    )
    rules.extend(
        (
            _literal_binary_rule(
                vector.vector_addi,
                "amdgpu.v_add_u32.lit",
                literal_source="lhs",
                nonliteral_source="rhs",
            ),
            _literal_binary_rule(
                vector.vector_addi,
                "amdgpu.v_add_u32.lit",
                literal_source="rhs",
                nonliteral_source="lhs",
            ),
            _binary_rule(vector.vector_addi, _VEC_I32, "amdgpu.v_add_u32"),
        )
    )
    rules.extend(
        (
            _binary_rule(vector.vector_subi, _VEC_I32, "amdgpu.v_sub_u32"),
            _binary_rule(vector.vector_muli, _VEC_I32, "amdgpu.v_mul_lo_u32"),
            _binary_rule(vector.vector_minsi, _VEC_I32, "amdgpu.v_min_i32"),
            _binary_rule(vector.vector_maxsi, _VEC_I32, "amdgpu.v_max_i32"),
            _binary_rule(vector.vector_minui, _VEC_I32, "amdgpu.v_min_u32"),
            _binary_rule(vector.vector_maxui, _VEC_I32, "amdgpu.v_max_u32"),
        )
    )
    for source_op, descriptor_key in (
        (vector.vector_andi, "amdgpu.v_and_b32"),
        (vector.vector_ori, "amdgpu.v_or_b32"),
        (vector.vector_xori, "amdgpu.v_xor_b32"),
    ):
        rules.extend(
            (
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="lhs",
                    nonliteral_source="rhs",
                ),
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="rhs",
                    nonliteral_source="lhs",
                ),
                _binary_rule(source_op, _VEC_I32, descriptor_key),
            )
        )
    for source_op, descriptor_key in (
        (vector.vector_shli, "amdgpu.v_lshlrev_b32"),
        (vector.vector_shrsi, "amdgpu.v_ashrrev_i32"),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b32"),
    ):
        rules.extend(
            (
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    descriptor_operand="value",
                ),
                _binary_rule(
                    source_op,
                    _VEC_I32,
                    descriptor_key,
                    descriptor_lhs="shift",
                    descriptor_rhs="value",
                    source_lhs="rhs",
                    source_rhs="lhs",
                ),
            )
        )
    rules.extend(_vector_bitfield_rules())
    rules.extend(_vector_packed_integer_recipe_rules())
    for source_op, descriptor_key in (
        (scalar_arithmetic.scalar_addf, "amdgpu.v_add_f32.lit"),
        (scalar_arithmetic.scalar_mulf, "amdgpu.v_mul_f32.lit"),
        (scalar_arithmetic.scalar_minnumf, "amdgpu.v_min_f32.lit"),
        (scalar_arithmetic.scalar_maxnumf, "amdgpu.v_max_f32.lit"),
    ):
        inline_descriptor_key = descriptor_key.removesuffix(".lit") + ".src0_inline"
        for literal_value in _SOURCE_INLINE_F32_VALUES:
            rules.extend(
                (
                    _f32_inline_binary_rule(
                        source_op,
                        _F32,
                        inline_descriptor_key,
                        literal_source="lhs",
                        nonliteral_source="rhs",
                        literal_value=literal_value,
                        f32_operand=True,
                    ),
                    _f32_inline_binary_rule(
                        source_op,
                        _F32,
                        inline_descriptor_key,
                        literal_source="rhs",
                        nonliteral_source="lhs",
                        literal_value=literal_value,
                        f32_operand=True,
                    ),
                )
            )
        rules.extend(
            (
                _f32_literal_binary_rule(
                    source_op,
                    _F32,
                    descriptor_key,
                    literal_source="lhs",
                    nonliteral_source="rhs",
                    f32_operand=True,
                ),
                _f32_literal_binary_rule(
                    source_op,
                    _F32,
                    descriptor_key,
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    f32_operand=True,
                ),
            )
        )
    rules.extend(
        _f32_inline_binary_rule(
            scalar_arithmetic.scalar_subf,
            _F32,
            "amdgpu.v_sub_f32.src0_inline",
            literal_source="lhs",
            nonliteral_source="rhs",
            literal_value=literal_value,
            f32_operand=True,
        )
        for literal_value in _SOURCE_INLINE_F32_VALUES
    )
    rules.append(
        _f32_literal_binary_rule(
            scalar_arithmetic.scalar_subf,
            _F32,
            "amdgpu.v_sub_f32.lit",
            literal_source="lhs",
            nonliteral_source="rhs",
            f32_operand=True,
        )
    )
    rules.extend(
        (
            _cast_rule(
                vector.vector_sitofp,
                _VEC_I32,
                _VEC_F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                vector.vector_uitofp,
                _VEC_I32,
                _VEC_F32,
                "amdgpu.v_cvt_f32_u32",
            ),
            _cast_rule(
                vector.vector_fptosi,
                _VEC_F32,
                _VEC_I32,
                "amdgpu.v_cvt_i32_f32",
                f32_input=True,
            ),
            _cast_rule(
                vector.vector_fptoui,
                _VEC_F32,
                _VEC_I32,
                "amdgpu.v_cvt_u32_f32",
                f32_input=True,
            ),
            *_commutative_f32_binary_rules(
                scalar_arithmetic.scalar_addf,
                _F32,
                "amdgpu.v_add_f32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_subf,
                _F32,
                "amdgpu.v_sub_f32",
                f32_rhs=True,
            ),
            *_commutative_f32_binary_rules(
                scalar_arithmetic.scalar_mulf,
                _F32,
                "amdgpu.v_mul_f32",
            ),
            _f32_neg_rule(scalar_arithmetic.scalar_negf, _F32, f32_operand=True),
            _f32_abs_rule(scalar_arithmetic.scalar_absf, _F32, f32_operand=True),
            _f32_copysign_rule(scalar_arithmetic.scalar_copysignf, _F32),
            _divf_arcp_one_rule(scalar_arithmetic.scalar_divf, _F32),
            _divf_arcp_rule(scalar_arithmetic.scalar_divf, _F32),
            _divf_exact_rule(scalar_arithmetic.scalar_divf, _F32),
            *_commutative_f32_binary_rules(
                scalar_arithmetic.scalar_minnumf,
                _F32,
                "amdgpu.v_min_f32",
            ),
            *_commutative_f32_binary_rules(
                scalar_arithmetic.scalar_maxnumf,
                _F32,
                "amdgpu.v_max_f32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minsi,
                _I32,
                "amdgpu.v_min_i32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxsi,
                _I32,
                "amdgpu.v_max_i32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minui,
                _I32,
                "amdgpu.v_min_u32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxui,
                _I32,
                "amdgpu.v_max_u32",
            ),
            *_f32_fma_rules(scalar_math.scalar_fmaf, _F32),
            _unary_rule(scalar_math.scalar_exp2f, _F32, "amdgpu.v_exp_f32"),
            _unary_rule(scalar_math.scalar_log2f, _F32, "amdgpu.v_log_f32"),
            _unary_rule(scalar_math.scalar_sinturnsf, _F32, "amdgpu.v_sin_f32"),
            _unary_rule(scalar_math.scalar_costurnsf, _F32, "amdgpu.v_cos_f32"),
            _unary_rule(scalar_math.scalar_floorf, _F32, "amdgpu.v_floor_f32"),
            _unary_rule(scalar_math.scalar_ceilf, _F32, "amdgpu.v_ceil_f32"),
            _unary_rule(scalar_math.scalar_roundevenf, _F32, "amdgpu.v_rndne_f32"),
            _unary_rule(scalar_math.scalar_truncf, _F32, "amdgpu.v_trunc_f32"),
            _unary_rule(scalar_math.scalar_sqrtf, _F32, "amdgpu.v_sqrt_f32"),
            _unary_rule(scalar_math.scalar_rsqrtf, _F32, "amdgpu.v_rsq_f32"),
            _cast_rule(
                scalar_conversion.scalar_extf,
                _F16,
                _F32,
                "amdgpu.v_cvt_f32_f16",
            ),
            _bf16_extf_rule(),
            _cast_rule(
                scalar_conversion.scalar_fptrunc,
                _F32,
                _F16,
                "amdgpu.v_cvt_f16_f32",
                f32_input=True,
            ),
            _bf16_fptrunc_rule(),
            _cast_rule(
                scalar_conversion.scalar_sitofp,
                _I32,
                _F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                scalar_conversion.scalar_sitofp,
                _I8,
                _F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                scalar_conversion.scalar_sitofp,
                _I16,
                _F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                scalar_conversion.scalar_uitofp,
                _I32,
                _F32,
                "amdgpu.v_cvt_f32_u32",
            ),
            _bitcast_alias_rule(_I32, _F32),
            _bitcast_alias_rule(_F32, _I32),
            _bitcast_alias_rule(_I64, _F64),
            _bitcast_alias_rule(_F64, _I64),
            _bitcast_alias_rule(_I16, _F16),
            _bitcast_alias_rule(_F16, _I16),
            _bitcast_alias_rule(_I16, _BF16),
            _bitcast_alias_rule(_BF16, _I16),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=True,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=True,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_lshl_add_rule(
                scale_source="a",
                value_source="b",
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_lshl_add_rule(
                scale_source="b",
                value_source="a",
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=False,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=False,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=True,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=True,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_lshl_add_rule(
                scale_source="a",
                value_source="b",
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_lshl_add_rule(
                scale_source="b",
                value_source="a",
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=False,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=False,
                preserve_value_register=False,
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="b",
                preserved_source="a",
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="a",
                preserved_source="b",
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="c",
                preserved_source="a",
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="c",
                preserved_source="b",
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="b",
                preserved_source=None,
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="a",
                preserved_source=None,
            ),
            _index_madd_u24_mad_literal_rule(
                literal_source="c",
                preserved_source=None,
            ),
            _index_madd_u24_mad_rule(preserved_source="a"),
            _index_madd_u24_mad_rule(preserved_source="b"),
            _index_madd_u24_mad_rule(preserved_source=None),
            _index_madd_literal_rule(),
            _index_madd_rule(),
        )
    )
    return tuple(rules)


AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.arithmetic",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/arithmetic.h",
    c_source_includes=("loom/target/arch/amdgpu/lower/kinds.h",),
    materializers=(ADDRESS_VGPR_MATERIALIZER, F32_VGPR_MATERIALIZER),
    cases=_rules(),
)
