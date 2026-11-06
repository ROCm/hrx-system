# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU integer arithmetic and predicate mask source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.index_bitwise import index_bitwise_rules
from loom.target.arch.amdgpu.contracts.integer_division import integer_division_rules
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_SGPR_MATERIALIZER,
    ADDRESS_VGPR_MATERIALIZER,
    I1_NATIVE_MASK_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
    I64_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.contracts.memory import BYTE_ADDRESSABLE_SCALAR_ELEMENTS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    AttrProject,
    Buffer,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterSlice,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    Scalar,
    TypePattern,
    ValueMaterializer,
    ValueProject,
    ValueRef,
    Vector,
    View,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_cselect_b32",
    "amdgpu.s_cmp_lg_i32",
    "amdgpu.s_cmp_lg_i32.src1_inline",
    "amdgpu.s_cmp_ge_u32",
    "amdgpu.s_add_u32",
    "amdgpu.s_add_co_u32",
    "amdgpu.s_sub_u32",
    "amdgpu.s_sub_co_u32",
    "amdgpu.s_mul_i32",
    "amdgpu.s_mul_hi_i32",
    "amdgpu.s_mul_hi_u32",
    "amdgpu.s_min_i32",
    "amdgpu.s_min_i32.rhs_inline",
    "amdgpu.s_min_i32.lit",
    "amdgpu.s_max_i32",
    "amdgpu.s_max_i32.rhs_inline",
    "amdgpu.s_max_i32.lit",
    "amdgpu.s_min_u32",
    "amdgpu.s_min_u32.rhs_inline",
    "amdgpu.s_min_u32.lit",
    "amdgpu.s_max_u32",
    "amdgpu.s_max_u32.rhs_inline",
    "amdgpu.s_max_u32.lit",
    "amdgpu.s_and_b32",
    "amdgpu.s_and_b32.rhs_inline",
    "amdgpu.s_and_b32.lit",
    "amdgpu.s_or_b32",
    "amdgpu.s_or_b32.rhs_inline",
    "amdgpu.s_or_b32.lit",
    "amdgpu.s_xor_b32",
    "amdgpu.s_xor_b32.rhs_inline",
    "amdgpu.s_xor_b32.lit",
    "amdgpu.s_and_b64",
    "amdgpu.s_mov_b64_exec_read",
    "amdgpu.s_or_b64",
    "amdgpu.s_xor_b64",
    "amdgpu.s_lshl_b64",
    "amdgpu.s_lshr_b64",
    "amdgpu.s_lshl_b32",
    "amdgpu.s_lshr_b32",
    "amdgpu.s_lshr_b32.rhs_inline",
    "amdgpu.s_ashr_i32",
    "amdgpu.s_ashr_i32.rhs_inline",
    "amdgpu.s_bcnt1_i32_b32",
    "amdgpu.s_bcnt1_i32_b64",
    "amdgpu.s_clz_i32_u32",
    "amdgpu.s_ctz_i32_b32",
    "amdgpu.s_bfe_i32.lit",
    "amdgpu.s_bfe_u32.lit",
    "amdgpu.v_mov_b32",
    "amdgpu.v_readfirstlane_b32",
    "amdgpu.v_cvt_f32_u32",
    "amdgpu.v_rcp_f32",
    "amdgpu.v_mul_f32.lit",
    "amdgpu.v_cvt_u32_f32",
    "amdgpu.v_cmp_uge_u32",
    "amdgpu.v_cndmask_b32",
    "amdgpu.v_add_u32",
    "amdgpu.v_add_co_u32",
    "amdgpu.v_sub_u32",
    "amdgpu.v_sub_co_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_mul_hi_i32",
    "amdgpu.v_mul_hi_u32",
    "amdgpu.v_mad_u32_u24.src0_inline",
    "amdgpu.v_mad_u32_u24.src1_inline",
    "amdgpu.v_lshl_add_u32.shift_imm",
    "amdgpu.v_min_i32",
    "amdgpu.v_min_i32.src0_inline",
    "amdgpu.v_min_i32.lit",
    "amdgpu.v_max_i32",
    "amdgpu.v_max_i32.src0_inline",
    "amdgpu.v_max_i32.lit",
    "amdgpu.v_min_u32",
    "amdgpu.v_min_u32.src0_inline",
    "amdgpu.v_min_u32.lit",
    "amdgpu.v_max_u32",
    "amdgpu.v_max_u32.src0_inline",
    "amdgpu.v_max_u32.lit",
    "amdgpu.v_and_b32",
    "amdgpu.v_and_b32.src0_inline",
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_or_b32",
    "amdgpu.v_or_b32.src0_inline",
    "amdgpu.v_or_b32.lit",
    "amdgpu.v_xor_b32",
    "amdgpu.v_xor_b32.src0_inline",
    "amdgpu.v_xor_b32.lit",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.src0_inline",
    "amdgpu.v_lshlrev_b32.vop3_imm",
    "amdgpu.v_lshrrev_b32",
    "amdgpu.v_lshrrev_b32.src0_inline",
    "amdgpu.v_bcnt_u32_b32",
    "amdgpu.v_bcnt_u32_b32.src1_zero",
    "amdgpu.v_clz_i32_u32",
    "amdgpu.v_ctz_i32_b32",
    "amdgpu.v_alignbit_b32",
    "amdgpu.v_ashrrev_i32",
    "amdgpu.v_ashrrev_i32.src0_inline",
    "amdgpu.v_bfe_i32.offset_width_inline",
    "amdgpu.v_bfe_u32.offset_width_inline",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.integer",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I1 = Scalar("i1")
_VECTOR_I1 = Vector(
    "i1",
    minimum_static_elements=1,
    maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_DIRECT_LHS = ValueRef.operand("lhs")
_DIRECT_RHS = ValueRef.operand("rhs")
_RESULT = ValueRef.result("result")
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="u32",
    constraint_key="amdgpu.address.u32",
)
_ADDRESS_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="i32",
    constraint_key="amdgpu.address.i32",
)
_ADDRESS_32BIT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="32-bit",
    constraint_key="amdgpu.address.bits32",
)
_BYTE_OFFSET_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="byte-offset-width",
    subject_name="u32",
    constraint_key="amdgpu.byte_offset.u32",
)
_I32_LITERAL_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="i32",
    constraint_key="amdgpu.literal.exact_i64",
)
_I32_LITERAL_BITS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal-bits",
    subject_name="i32",
    constraint_key="amdgpu.literal.i32_bits",
)
_SHIFT_AMOUNT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="shift-amount",
    subject_name="u5",
    constraint_key="amdgpu.shift_amount.u5",
)
_POSITIVE_U32_DIVISOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="divisor",
    subject_name="u32",
    constraint_key="amdgpu.divisor.positive_u32",
)
_UINT32_MAX = (2**32) - 1


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _descriptor_available_guards(*descriptors: Descriptor) -> tuple[Guard, ...]:
    seen_keys: set[str] = set()
    guards: list[Guard] = []
    for descriptor in descriptors:
        if descriptor.key in seen_keys:
            continue
        seen_keys.add(descriptor.key)
        guards.append(Guard.descriptor_available(descriptor))
    return tuple(guards)


def _typed_binary_guards(
    type_pattern: TypePattern,
    *,
    signed_bit_count: int | None = None,
    signed_diagnostic: GuardDiagnostic | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> tuple[Guard, ...]:
    guards = [
        Guard.value_type("lhs", type_pattern),
        Guard.value_type("rhs", type_pattern),
        Guard.value_type("result", type_pattern),
    ]
    if signed_bit_count is not None:
        guards.append(
            Guard.value_signed_bit_count(
                "result",
                signed_bit_count,
                diagnostic=signed_diagnostic,
            )
        )
    if unsigned_bit_count is not None:
        guards.append(
            Guard.value_unsigned_bit_count(
                "result",
                unsigned_bit_count,
                diagnostic=unsigned_diagnostic,
            )
        )
    return tuple(guards)


def _sgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    *,
    signed_bit_count: int | None = None,
    signed_diagnostic: GuardDiagnostic | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
    register_unit_count: int | None = None,
) -> DescriptorRule:
    register_shape_guards = (
        (
            Guard.low_value_register_unit_count("result", register_unit_count),
            Guard.low_value_register_unit_count("lhs", register_unit_count),
            Guard.low_value_register_unit_count("rhs", register_unit_count),
        )
        if register_unit_count is not None
        else (
            Guard.low_value_register_unit_count_eq("result", "lhs"),
            Guard.low_value_register_unit_count_eq("result", "rhs"),
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                signed_bit_count=signed_bit_count,
                signed_diagnostic=signed_diagnostic,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            *register_shape_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": _DIRECT_LHS, "rhs": _DIRECT_RHS},
                results={"dst": _RESULT},
            ),
        ),
    )


def _materialized_operand(field: str, materializer: ValueMaterializer) -> ValueRef:
    return ValueRef.operand(field, materializer=materializer.name)


def _scalar_fmai_typed_guards() -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, _I32) for field in ("a", "b", "c", "result"))


def _scalar_fmai_input_guard(field: str, sgpr_source: str | None) -> Guard:
    register_class = "amdgpu.sgpr" if field == sgpr_source else "amdgpu.vgpr"
    return Guard.low_value_register_class(field, register_class)


def _scalar_fmai_power_of_two_rule(
    *, scale_source: str, value_source: str, sgpr_source: str | None
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_lshl_add_u32.shift_imm")
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_fmai,
        descriptor=descriptor,
        guards=(
            *_scalar_fmai_typed_guards(),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_exact_power_of_two_i64(scale_source),
            Guard.value_unsigned_bit_count(scale_source, 32),
            _scalar_fmai_input_guard(value_source, sgpr_source),
            _scalar_fmai_input_guard("c", sgpr_source),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "value": ValueRef.operand(value_source),
                    "addend": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"shift": ValueProject.exact_i64_log2(scale_source)},
            ),
        ),
    )


def _scalar_fmai_inline_u24_rule(
    *, scale_source: str, value_source: str, sgpr_source: str | None
) -> DescriptorRule:
    descriptor_source = "src0" if scale_source == "a" else "src1"
    descriptor = _descriptor(f"amdgpu.v_mad_u32_u24.{descriptor_source}_inline")
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_fmai,
        descriptor=descriptor,
        guards=(
            *_scalar_fmai_typed_guards(),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_unsigned_bit_count("a", 24),
            Guard.value_unsigned_bit_count("b", 24),
            Guard.value_exact_i64(scale_source),
            Guard.value_i64_range(scale_source, 0, 64),
            _scalar_fmai_input_guard(value_source, sgpr_source),
            _scalar_fmai_input_guard("c", sgpr_source),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    value_source: ValueRef.operand(value_source),
                    "addend": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64(scale_source)},
            ),
        ),
    )


def _scalar_fmai_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _scalar_fmai_power_of_two_rule(
            scale_source=scale_source,
            value_source=value_source,
            sgpr_source=sgpr_source,
        )
        for scale_source, value_source in (("a", "b"), ("b", "a"))
        for sgpr_source in ("c", value_source, None)
    ]
    rules.extend(
        _scalar_fmai_inline_u24_rule(
            scale_source=scale_source,
            value_source=value_source,
            sgpr_source=sgpr_source,
        )
        for scale_source, value_source in (("a", "b"), ("b", "a"))
        for sgpr_source in ("c", value_source, None)
    )
    return tuple(rules)


def _vgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    materializer: ValueMaterializer,
    *,
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    signed_bit_count: int | None = None,
    signed_diagnostic: GuardDiagnostic | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
    register_unit_count: int | None = None,
    source_value_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    register_shape_guards = (
        (Guard.low_value_register_unit_count("result", register_unit_count),)
        if register_unit_count is not None
        else ()
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                signed_bit_count=signed_bit_count,
                signed_diagnostic=signed_diagnostic,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_materializable("lhs", materializer.name),
            Guard.value_materializable("rhs", materializer.name),
            *source_value_guards,
            *register_shape_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    descriptor_lhs: _materialized_operand(source_lhs, materializer),
                    descriptor_rhs: _materialized_operand(source_rhs, materializer),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _narrow_address_sgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    *,
    signed_bit_count: int | None = None,
    signed_diagnostic: GuardDiagnostic | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
    source_value_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                signed_bit_count=signed_bit_count,
                signed_diagnostic=signed_diagnostic,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 1),
            Guard.value_materializable("lhs", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.value_materializable("rhs", ADDRESS_SGPR_MATERIALIZER.name),
            *source_value_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand("lhs", ADDRESS_SGPR_MATERIALIZER),
                    "rhs": _materialized_operand("rhs", ADDRESS_SGPR_MATERIALIZER),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _narrow_address_vgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    *,
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    lhs_materializer: ValueMaterializer = ADDRESS_VGPR_MATERIALIZER,
    source_lhs_register_class: str | None = None,
    signed_bit_count: int | None = None,
    signed_diagnostic: GuardDiagnostic | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                signed_bit_count=signed_bit_count,
                signed_diagnostic=signed_diagnostic,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.low_value_register_unit_count("result", 1),
            *(
                (Guard.low_value_register_class(source_lhs, source_lhs_register_class),)
                if source_lhs_register_class is not None
                else ()
            ),
            Guard.value_materializable(source_lhs, lhs_materializer.name),
            Guard.value_materializable(source_rhs, ADDRESS_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand(source_lhs, lhs_materializer),
                    "rhs": _materialized_operand(source_rhs, ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _i64_vgpr_per_lane_binary_rule(
    source_op: Op,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_I64),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.low_value_register_unit_count("result", 2),
            Guard.value_materializable("lhs", I64_VGPR_MATERIALIZER.name),
            Guard.value_materializable("rhs", I64_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand("lhs", I64_VGPR_MATERIALIZER),
                    "rhs": _materialized_operand("rhs", I64_VGPR_MATERIALIZER),
                },
                results={"dst": _RESULT},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vgpr_constant_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    materializer: ValueMaterializer,
    *,
    preserve_source_register: bool = False,
    preserved_source_materializer: ValueMaterializer | None = None,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
    register_unit_count: int | None = None,
    source_value_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    register_shape_guards = (
        (Guard.low_value_register_unit_count("result", register_unit_count),)
        if register_unit_count is not None
        else ()
    )
    if preserve_source_register:
        value_guards = (
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            *(
                (Guard.value_materializable("lhs", preserved_source_materializer.name),)
                if preserved_source_materializer is not None
                else ()
            ),
        )
        value_operand = (
            _materialized_operand("lhs", preserved_source_materializer)
            if preserved_source_materializer is not None
            else ValueRef.operand("lhs")
        )
    else:
        value_guards = (Guard.value_materializable("lhs", materializer.name),)
        value_operand = _materialized_operand("lhs", materializer)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            *value_guards,
            *source_value_guards,
            *register_shape_guards,
            Guard.value_exact_i64("rhs", diagnostic=_SHIFT_AMOUNT_DIAGNOSTIC),
            Guard.value_i64_range(
                "rhs",
                0,
                31,
                diagnostic=_SHIFT_AMOUNT_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": value_operand},
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64("rhs")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _i32_sgpr_vgpr_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    return (
        _sgpr_binary_rule(source_op, _I32, sgpr_descriptor),
        _vgpr_binary_rule(
            source_op,
            _I32,
            vgpr_descriptor,
            I32_VGPR_MATERIALIZER,
        ),
    )


def _carry_borrow_typed_guards(
    value_type: TypePattern,
    value_result: str,
    predicate_result: str,
) -> tuple[Guard, ...]:
    return (
        Guard.value_type("lhs", value_type),
        Guard.value_type("rhs", value_type),
        Guard.value_type(value_result, value_type),
        Guard.value_type(predicate_result, _I1),
    )


def _i32_carry_borrow_sgpr_rule(
    source_op: Op,
    descriptor_key: str,
    *,
    value_result: str,
    predicate_result: str,
    descriptor_predicate_result: str,
    predicate_register_class: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    guards = (
        *_carry_borrow_typed_guards(_I32, value_result, predicate_result),
        Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
        Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
        Guard.low_value_register_class(value_result, "amdgpu.sgpr"),
        Guard.low_value_register_unit_count("lhs", 1),
        Guard.low_value_register_unit_count("rhs", 1),
        Guard.low_value_register_unit_count(value_result, 1),
        Guard.low_value_register_class(predicate_result, predicate_register_class),
        Guard.low_value_register_unit_count(predicate_result, 1),
        Guard.descriptor_available(descriptor),
    )
    if predicate_register_class == "amdgpu.scc":
        emit = (
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": _DIRECT_LHS, "rhs": _DIRECT_RHS},
                results={
                    "dst": ValueRef.result(value_result),
                    descriptor_predicate_result: ValueRef.result(predicate_result),
                },
            ),
        )
    else:
        move = _descriptor("amdgpu.s_mov_b32")
        select = _descriptor("amdgpu.s_cselect_b32")
        guards += (
            Guard.descriptor_available(move),
            Guard.descriptor_available(select),
        )
        emit = (
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": _DIRECT_LHS, "rhs": _DIRECT_RHS},
                results={
                    "dst": ValueRef.result(value_result),
                    descriptor_predicate_result: ValueRef.temporary("predicate"),
                },
                result_types={"dst": _I32, descriptor_predicate_result: _I1},
            ),
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("false_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 0},
            ),
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("true_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 1},
            ),
            EmitDescriptorOp(
                descriptor=select,
                operands={
                    "true_value": ValueRef.temporary("true_value"),
                    "false_value": ValueRef.temporary("false_value"),
                    "condition": ValueRef.temporary("predicate"),
                },
                results={"dst": ValueRef.result(predicate_result)},
            ),
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=guards,
        emit=emit,
    )


def _i32_carry_borrow_vgpr_rule(
    source_op: Op,
    descriptor_key: str,
    *,
    value_result: str,
    predicate_result: str,
    descriptor_value_result: str,
    descriptor_predicate_result: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_carry_borrow_typed_guards(_I32, value_result, predicate_result),
            Guard.low_value_register_class(value_result, "amdgpu.vgpr"),
            Guard.low_value_register_unit_count(value_result, 1),
            Guard.low_value_register_class(predicate_result, "amdgpu.sgpr"),
            Guard.low_value_register_unit_count(predicate_result, 2),
            Guard.value_materializable("lhs", I32_VGPR_MATERIALIZER.name),
            Guard.value_materializable("rhs", I32_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand("lhs", I32_VGPR_MATERIALIZER),
                    "rhs": _materialized_operand("rhs", I32_VGPR_MATERIALIZER),
                },
                results={
                    descriptor_value_result: ValueRef.result(value_result),
                    descriptor_predicate_result: ValueRef.result(predicate_result),
                },
            ),
        ),
    )


def _unsigned_carry_borrow_rules(
    source_op: Op,
    *,
    value_result: str,
    predicate_result: str,
    descriptor_predicate_result: str,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    vgpr_value_result: str,
) -> tuple[DescriptorRule | RecipeRule, ...]:
    return (
        *(
            _i32_carry_borrow_sgpr_rule(
                source_op,
                sgpr_descriptor_key,
                value_result=value_result,
                predicate_result=predicate_result,
                descriptor_predicate_result=descriptor_predicate_result,
                predicate_register_class=predicate_register_class,
            )
            for predicate_register_class in ("amdgpu.scc", "amdgpu.sgpr")
        ),
        _i32_carry_borrow_vgpr_rule(
            source_op,
            vgpr_descriptor_key,
            value_result=value_result,
            predicate_result=predicate_result,
            descriptor_value_result=vgpr_value_result,
            descriptor_predicate_result=descriptor_predicate_result,
        ),
        RecipeRule(
            source_op=source_op,
            guards=_carry_borrow_typed_guards(_I64, value_result, predicate_result),
        ),
    )


def _i32_immediate_binary_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    literal_source: str,
    nonliteral_source: str,
    register_class: str,
    literal_range: tuple[int, int] | None = None,
) -> DescriptorRule:
    scalar_register = register_class == "amdgpu.sgpr"
    register_guard = (
        Guard.low_value_register_class(nonliteral_source, register_class)
        if scalar_register
        else Guard.value_materializable(nonliteral_source, I32_VGPR_MATERIALIZER.name)
    )
    operand = (
        ValueRef.operand(nonliteral_source)
        if scalar_register
        else _materialized_operand(nonliteral_source, I32_VGPR_MATERIALIZER)
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_I32),
            Guard.low_value_register_class("result", register_class),
            register_guard,
            Guard.value_exact_i64(
                literal_source,
                diagnostic=_I32_LITERAL_EXACT_DIAGNOSTIC,
            ),
            Guard.value_signed_bit_count(
                literal_source,
                32,
                diagnostic=_I32_LITERAL_BITS_DIAGNOSTIC,
            ),
            *(
                (Guard.value_i64_range(literal_source, *literal_range),)
                if literal_range is not None
                else ()
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs" if scalar_register else "rhs": operand},
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.i32_as_u32_bits(literal_source)},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _i32_encoded_binary_rules(source_op: Op, suffix: str) -> tuple[DescriptorRule, ...]:
    immediate_rules = tuple(
        _i32_immediate_binary_rule(
            source_op,
            _descriptor(descriptor_key),
            literal_source=literal_source,
            nonliteral_source=nonliteral_source,
            register_class=register_class,
            literal_range=literal_range,
        )
        for descriptor_key, register_class, literal_range in (
            (f"amdgpu.s_{suffix}.rhs_inline", "amdgpu.sgpr", (0, 64)),
            (f"amdgpu.s_{suffix}.lit", "amdgpu.sgpr", None),
            (f"amdgpu.v_{suffix}.src0_inline", "amdgpu.vgpr", (0, 64)),
            (f"amdgpu.v_{suffix}.lit", "amdgpu.vgpr", None),
        )
        for literal_source, nonliteral_source in (("lhs", "rhs"), ("rhs", "lhs"))
    )
    return (
        *immediate_rules,
        _sgpr_binary_rule(source_op, _I32, _descriptor(f"amdgpu.s_{suffix}")),
        _vgpr_binary_rule(
            source_op,
            _I32,
            _descriptor(f"amdgpu.v_{suffix}"),
            I32_VGPR_MATERIALIZER,
        ),
    )


def _i1_sgpr_mask_rule(
    source_op: Op,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.value_materializable("lhs", I1_NATIVE_MASK_MATERIALIZER.name),
            Guard.value_materializable("rhs", I1_NATIVE_MASK_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand("lhs", I1_NATIVE_MASK_MATERIALIZER),
                    "rhs": _materialized_operand("rhs", I1_NATIVE_MASK_MATERIALIZER),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


_I1_SCALAR_BOOL_CLASSES = ("amdgpu.scc", "amdgpu.sgpr")


def _i1_mixed_bool_mask_rule(
    source_op: Op,
    operation: str,
    bool_field: str,
    bool_class: str,
) -> DescriptorRule:
    mask_field = "rhs" if bool_field == "lhs" else "lhs"
    move = _descriptor("amdgpu.s_mov_b32")
    read_exec = _descriptor("amdgpu.s_mov_b64_exec_read")
    bit_xor = _descriptor("amdgpu.s_xor_b64")
    compare = _descriptor("amdgpu.s_cmp_lg_i32.src1_inline")
    select = _descriptor("amdgpu.s_cselect_b32")

    setup: tuple[EmitDescriptorOp, ...]
    if operation == "and":
        setup = (
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("false_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 0},
            ),
        )
        true_value = ValueRef.operand(mask_field)
        false_value = ValueRef.temporary("false_value")
        setup_descriptors = (move,)
    elif operation == "or":
        setup = (
            EmitDescriptorOp(
                descriptor=read_exec,
                results={"dst": ValueRef.temporary("true_value")},
                result_types={"dst": _RESULT},
            ),
        )
        true_value = ValueRef.temporary("true_value")
        false_value = ValueRef.operand(mask_field)
        setup_descriptors = (read_exec,)
    elif operation == "xor":
        setup = (
            EmitDescriptorOp(
                descriptor=read_exec,
                results={"dst": ValueRef.temporary("exec")},
                result_types={"dst": _RESULT},
            ),
            EmitDescriptorOp(
                descriptor=bit_xor,
                operands={
                    "lhs": ValueRef.operand(mask_field),
                    "rhs": ValueRef.temporary("exec"),
                },
                results={"dst": ValueRef.temporary("true_value")},
                result_types={"dst": _RESULT},
            ),
        )
        true_value = ValueRef.temporary("true_value")
        false_value = ValueRef.operand(mask_field)
        setup_descriptors = (read_exec, bit_xor)
    else:
        raise ValueError(f"unsupported mixed Boolean/mask operation: {operation}")

    condition = ValueRef.operand(bool_field)
    if bool_class == "amdgpu.sgpr":
        setup += (
            EmitDescriptorOp(
                descriptor=compare,
                operands={"lhs": condition},
                results={"scc": ValueRef.temporary("condition")},
                result_types={"scc": _I1},
                immediates={"rhs": 0},
            ),
        )
        condition = ValueRef.temporary("condition")
        setup_descriptors += (compare,)

    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=(
            *_typed_binary_guards(_I1),
            Guard.low_value_register_class(mask_field, "amdgpu.sgpr"),
            Guard.low_value_register_unit_count(mask_field, 2),
            Guard.low_value_register_class(bool_field, bool_class),
            Guard.low_value_register_unit_count(bool_field, 1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 2),
            *_descriptor_available_guards(*setup_descriptors, select),
        ),
        emit=(
            *setup,
            EmitDescriptorOp(
                descriptor=select,
                operands={
                    "true_value": true_value,
                    "false_value": false_value,
                    "condition": condition,
                },
                results={"dst": _RESULT},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _i1_mixed_bool_mask_rules(
    source_op: Op,
    operation: str,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _i1_mixed_bool_mask_rule(
            source_op,
            operation,
            bool_field,
            bool_class,
        )
        for bool_class in _I1_SCALAR_BOOL_CLASSES
        if operation != "xor" or bool_class != "amdgpu.scc"
        for bool_field in ("lhs", "rhs")
    )


def _vector_predicate_bitwise_rule(
    source_op: Op, descriptor_key: str
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_VECTOR_I1),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": _DIRECT_LHS, "rhs": _DIRECT_RHS},
                results={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _vector_predicate_select_rule() -> DescriptorRule:
    bit_and = _descriptor("amdgpu.s_and_b64")
    bit_xor = _descriptor("amdgpu.s_xor_b64")
    read_exec = _descriptor("amdgpu.s_mov_b64_exec_read")
    bit_or = _descriptor("amdgpu.s_or_b64")
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=bit_or,
        guards=(
            *(
                Guard.value_type(field, _VECTOR_I1)
                for field in ("condition", "true_value", "false_value", "result")
            ),
            *_descriptor_available_guards(bit_and, read_exec, bit_xor, bit_or),
        ),
        # Match scalar selection's mask lifetimes by completing each two-SGPR
        # element before starting the next. Low CSE shares the EXEC snapshot;
        # true-side masks can die before false-side comparisons are available.
        emit=(
            EmitDescriptorOp(
                descriptor=bit_and,
                operands={
                    "lhs": ValueRef.operand("condition"),
                    "rhs": ValueRef.operand("true_value"),
                },
                results={"dst": ValueRef.temporary("selected_true")},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=read_exec,
                results={"dst": ValueRef.temporary("exec")},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=bit_xor,
                operands={
                    "lhs": ValueRef.operand("condition"),
                    "rhs": ValueRef.temporary("exec"),
                },
                results={"dst": ValueRef.temporary("inverse_condition")},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=bit_and,
                operands={
                    "lhs": ValueRef.temporary("inverse_condition"),
                    "rhs": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.temporary("selected_false")},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=bit_or,
                operands={
                    "lhs": ValueRef.temporary("selected_true"),
                    "rhs": ValueRef.temporary("selected_false"),
                },
                results={"dst": _RESULT},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def _whole_value_select_recipe_rules() -> tuple[RecipeRule, ...]:
    """Selects one supported scalar, reference, or vector value."""

    payload_types = (
        Scalar(("i1", *BYTE_ADDRESSABLE_SCALAR_ELEMENTS)),
        Buffer(),
        View(BYTE_ADDRESSABLE_SCALAR_ELEMENTS),
        _VECTOR_I1,
        Vector(
            ("i8", "f8E4M3", "f8E5M2"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_I8_LANES",
        ),
        Vector(
            "i16",
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_I16_LANES",
        ),
        Vector(
            ("f16", "bf16"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
        ),
        Vector(
            ("i32", "f32"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
        ),
        Vector(
            ("i64", "f64"),
            minimum_static_elements=1,
            maximum_static_elements=("(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)"),
        ),
    )
    return tuple(
        RecipeRule(
            source_op=scf.scf_select,
            guards=(
                Guard.value_type("condition", _I1),
                *(
                    Guard.value_type(field, payload_type)
                    for field in ("true_value", "false_value", "result")
                ),
            ),
        )
        for payload_type in payload_types
    )


def _i1_scalar_bool_operand(field: str, register_class: str) -> ValueRef:
    if register_class == "amdgpu.scc":
        return ValueRef.temporary(f"{field}_i32")
    return ValueRef.operand(field)


def _i1_scalar_bool_project_emit(
    field: str,
    register_class: str,
    select: Descriptor,
) -> tuple[EmitDescriptorOp, ...]:
    if register_class != "amdgpu.scc":
        return ()
    return (
        EmitDescriptorOp(
            descriptor=select,
            operands={
                "true_value": ValueRef.temporary("one"),
                "false_value": ValueRef.temporary("zero"),
                "condition": ValueRef.operand(field),
            },
            results={"dst": ValueRef.temporary(f"{field}_i32")},
            result_types={"dst": _I32},
        ),
    )


def _i1_scalar_bool_bitwise_rule(
    source_op: Op,
    descriptor_key: str,
    lhs_class: str,
    rhs_class: str,
    result_class: str,
) -> DescriptorRule:
    move = _descriptor("amdgpu.s_mov_b32")
    select = _descriptor("amdgpu.s_cselect_b32")
    bitwise = _descriptor(descriptor_key)
    compare = _descriptor("amdgpu.s_cmp_lg_i32")
    result_is_scc = result_class == "amdgpu.scc"
    projects_scc_operand = lhs_class == "amdgpu.scc" or rhs_class == "amdgpu.scc"
    bitwise_result = (
        ValueRef.temporary("result_i32") if result_is_scc else ValueRef.result("result")
    )
    prefix: tuple[EmitDescriptorOp, ...] = ()
    if result_is_scc or projects_scc_operand:
        prefix += (
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("zero")},
                result_types={"dst": _I32},
                immediates={"imm32": 0},
            ),
        )
    if projects_scc_operand:
        prefix += (
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("one")},
                result_types={"dst": _I32},
                immediates={"imm32": 1},
            ),
        )
    suffix: tuple[EmitDescriptorOp, ...] = ()
    if result_is_scc:
        suffix = (
            EmitDescriptorOp(
                descriptor=compare,
                operands={
                    "lhs": ValueRef.temporary("result_i32"),
                    "rhs": ValueRef.temporary("zero"),
                },
                results={"scc": ValueRef.result("result")},
            ),
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=bitwise,
        guards=(
            *_typed_binary_guards(_I1),
            Guard.low_value_register_class("lhs", lhs_class),
            Guard.low_value_register_unit_count("lhs", 1),
            Guard.low_value_register_class("rhs", rhs_class),
            Guard.low_value_register_unit_count("rhs", 1),
            Guard.low_value_register_class("result", result_class),
            Guard.low_value_register_unit_count("result", 1),
            *_descriptor_available_guards(move, select, bitwise, compare),
        ),
        emit=(
            *prefix,
            *_i1_scalar_bool_project_emit("lhs", lhs_class, select),
            *_i1_scalar_bool_project_emit("rhs", rhs_class, select),
            EmitDescriptorOp(
                descriptor=bitwise,
                operands={
                    "lhs": _i1_scalar_bool_operand("lhs", lhs_class),
                    "rhs": _i1_scalar_bool_operand("rhs", rhs_class),
                },
                results={"dst": bitwise_result},
                result_types={"dst": _I32},
            ),
            *suffix,
        ),
    )


def _i1_scalar_bool_bitwise_rules(
    source_op: Op,
    descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _i1_scalar_bool_bitwise_rule(
            source_op,
            descriptor_key,
            lhs_class,
            rhs_class,
            result_class,
        )
        for result_class in _I1_SCALAR_BOOL_CLASSES
        for lhs_class in _I1_SCALAR_BOOL_CLASSES
        for rhs_class in _I1_SCALAR_BOOL_CLASSES
    )


def _i32_shift_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    inline_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    inline_descriptor = _descriptor(inline_descriptor_key)
    preserve_rules = (
        (
            _vgpr_constant_shift_rule(
                source_op,
                _I32,
                _descriptor(preserve_descriptor_key),
                I32_VGPR_MATERIALIZER,
                preserve_source_register=True,
            ),
        )
        if preserve_descriptor_key is not None
        else ()
    )
    return (
        _sgpr_binary_rule(source_op, _I32, sgpr_descriptor),
        *preserve_rules,
        _vgpr_constant_shift_rule(
            source_op,
            _I32,
            inline_descriptor,
            I32_VGPR_MATERIALIZER,
        ),
        _vgpr_binary_rule(
            source_op,
            _I32,
            vgpr_descriptor,
            I32_VGPR_MATERIALIZER,
            descriptor_lhs="shift",
            descriptor_rhs="value",
            source_lhs="rhs",
            source_rhs="lhs",
        ),
    )


def _i64_word_extract_rules(source_op: Op) -> tuple[DescriptorRule, ...]:
    # A complete native word already occupies one register unit. Signed and
    # unsigned extraction have the same bits when the result is exactly i32.
    return tuple(
        DescriptorRule(
            source_op=source_op,
            guards=(
                Guard.value_type("source", _I64),
                Guard.value_type("result", _I32),
                Guard.i64_range("offset", offset, offset),
                Guard.i64_range("width", 32, 32),
                Guard.low_value_register_class("source", register_class),
                Guard.low_value_register_class("result", register_class),
            ),
            emit=(
                EmitRegisterSlice(
                    source=ValueRef.operand("source"),
                    result=_RESULT,
                    unit_offset=offset // 32,
                ),
            ),
        )
        for register_class in ("amdgpu.sgpr", "amdgpu.vgpr")
        for offset in (0, 32)
    )


def _i32_bitfield_extract_rules(
    source_op: Op,
    *,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    top_aligned_vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    top_aligned_vgpr_descriptor = _descriptor(top_aligned_vgpr_descriptor_key)
    return (
        DescriptorRule(
            source_op=source_op,
            descriptor=sgpr_descriptor,
            guards=(
                Guard.value_type("source", _I32),
                Guard.value_type("result", _I32),
                Guard.low_value_register_class("source", "amdgpu.sgpr"),
                Guard.low_value_register_class("result", "amdgpu.sgpr"),
                Guard.descriptor_available(sgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=sgpr_descriptor,
                    operands={"value": ValueRef.operand("source")},
                    results={"dst": _RESULT},
                    immediates={
                        # S_BFE places the offset in bits 4:0 and width in 22:16.
                        "imm32": AttrProject.attrs_pack_consecutive(
                            "offset",
                            count=2,
                            bit_width=16,
                        )
                    },
                    form=DescriptorEmitForm.OP,
                ),
            ),
        ),
        DescriptorRule(
            source_op=source_op,
            descriptor=top_aligned_vgpr_descriptor,
            guards=(
                Guard.value_type("source", _I32),
                Guard.value_type("result", _I32),
                Guard.i64_attrs_sum_equals("offset", "width", 32),
                Guard.i64_range("offset", 1, 31),
                Guard.low_value_register_class("result", "amdgpu.vgpr"),
                Guard.value_materializable("source", I32_VGPR_MATERIALIZER.name),
                Guard.descriptor_available(top_aligned_vgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=top_aligned_vgpr_descriptor,
                    operands={
                        "value": _materialized_operand(
                            "source",
                            I32_VGPR_MATERIALIZER,
                        )
                    },
                    results={"dst": _RESULT},
                    immediates={"imm32": AttrProject.direct("offset")},
                    form=DescriptorEmitForm.OP,
                ),
            ),
        ),
        DescriptorRule(
            source_op=source_op,
            descriptor=vgpr_descriptor,
            guards=(
                Guard.value_type("source", _I32),
                Guard.value_type("result", _I32),
                Guard.low_value_register_class("result", "amdgpu.vgpr"),
                Guard.value_materializable("source", I32_VGPR_MATERIALIZER.name),
                Guard.descriptor_available(vgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=vgpr_descriptor,
                    operands={
                        "value": _materialized_operand(
                            "source",
                            I32_VGPR_MATERIALIZER,
                        )
                    },
                    results={"dst": _RESULT},
                    immediates={
                        "offset": AttrProject.direct("offset"),
                        "width": AttrProject.direct("width"),
                    },
                    form=DescriptorEmitForm.OP,
                ),
            ),
        ),
    )


def _narrow_address_binary_rules(
    source_op: Op,
    type_pattern: TypePattern,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    *,
    sgpr_operand_orders: tuple[tuple[str, str], ...] = (),
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    return (
        _narrow_address_sgpr_binary_rule(
            source_op,
            type_pattern,
            sgpr_descriptor,
            unsigned_bit_count=unsigned_bit_count,
            unsigned_diagnostic=unsigned_diagnostic,
        ),
        *(
            _narrow_address_vgpr_binary_rule(
                source_op,
                type_pattern,
                vgpr_descriptor,
                source_lhs=source_lhs,
                source_rhs=source_rhs,
                lhs_materializer=ADDRESS_SGPR_MATERIALIZER,
                source_lhs_register_class="amdgpu.sgpr",
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            )
            for source_lhs, source_rhs in sgpr_operand_orders
        ),
        _narrow_address_vgpr_binary_rule(
            source_op,
            type_pattern,
            vgpr_descriptor,
            unsigned_bit_count=unsigned_bit_count,
            unsigned_diagnostic=unsigned_diagnostic,
        ),
    )


def _address_binary_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    *,
    sgpr_operand_orders: tuple[tuple[str, str], ...] = (),
) -> tuple[DescriptorRule, ...]:
    return (
        *_narrow_address_binary_rules(
            source_op,
            _INDEX,
            sgpr_descriptor_key,
            vgpr_descriptor_key,
            sgpr_operand_orders=sgpr_operand_orders,
        ),
        *_narrow_address_binary_rules(
            source_op,
            _OFFSET,
            sgpr_descriptor_key,
            vgpr_descriptor_key,
            sgpr_operand_orders=sgpr_operand_orders,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        ),
    )


def _commutative_index_binary_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    return _narrow_address_binary_rules(
        source_op,
        _INDEX,
        sgpr_descriptor_key,
        vgpr_descriptor_key,
        sgpr_operand_orders=(("lhs", "rhs"), ("rhs", "lhs")),
    )


def _signed_index_minmax_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    source_value_guards = tuple(
        Guard.value_signed_bit_count(
            field,
            32,
            diagnostic=_ADDRESS_I32_DIAGNOSTIC,
        )
        for field in ("lhs", "rhs")
    )
    return (
        _narrow_address_sgpr_binary_rule(
            source_op,
            _INDEX,
            sgpr_descriptor,
            signed_bit_count=32,
            signed_diagnostic=_ADDRESS_I32_DIAGNOSTIC,
            source_value_guards=source_value_guards,
        ),
        _vgpr_binary_rule(
            source_op,
            _INDEX,
            vgpr_descriptor,
            ADDRESS_VGPR_MATERIALIZER,
            signed_bit_count=32,
            signed_diagnostic=_ADDRESS_I32_DIAGNOSTIC,
            register_unit_count=1,
            source_value_guards=source_value_guards,
        ),
    )


def _address_scale_rules() -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor("amdgpu.s_mul_i32")
    vgpr_descriptor = _descriptor("amdgpu.v_mul_lo_u32")
    return (
        DescriptorRule(
            source_op=index.index_scale,
            descriptor=sgpr_descriptor,
            guards=(
                Guard.value_type("index", _INDEX),
                Guard.value_type("stride", _OFFSET),
                Guard.value_type("result", _OFFSET),
                Guard.value_unsigned_bit_count(
                    "result",
                    32,
                    diagnostic=_BYTE_OFFSET_U32_DIAGNOSTIC,
                ),
                Guard.low_value_register_class("result", "amdgpu.sgpr"),
                Guard.low_value_register_unit_count("result", 1),
                Guard.value_materializable("index", ADDRESS_SGPR_MATERIALIZER.name),
                Guard.value_materializable("stride", ADDRESS_SGPR_MATERIALIZER.name),
                Guard.descriptor_available(sgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=sgpr_descriptor,
                    operands={
                        "lhs": _materialized_operand(
                            "index", ADDRESS_SGPR_MATERIALIZER
                        ),
                        "rhs": _materialized_operand(
                            "stride", ADDRESS_SGPR_MATERIALIZER
                        ),
                    },
                    results={"dst": _RESULT},
                ),
            ),
        ),
        DescriptorRule(
            source_op=index.index_scale,
            descriptor=vgpr_descriptor,
            guards=(
                Guard.value_type("index", _INDEX),
                Guard.value_type("stride", _OFFSET),
                Guard.value_type("result", _OFFSET),
                Guard.value_unsigned_bit_count(
                    "result",
                    32,
                    diagnostic=_BYTE_OFFSET_U32_DIAGNOSTIC,
                ),
                Guard.low_value_register_class("result", "amdgpu.vgpr"),
                Guard.value_materializable("index", ADDRESS_VGPR_MATERIALIZER.name),
                Guard.value_materializable("stride", ADDRESS_VGPR_MATERIALIZER.name),
                Guard.descriptor_available(vgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=vgpr_descriptor,
                    operands={
                        "lhs": _materialized_operand(
                            "index", ADDRESS_VGPR_MATERIALIZER
                        ),
                        "rhs": _materialized_operand(
                            "stride", ADDRESS_VGPR_MATERIALIZER
                        ),
                    },
                    results={"dst": _RESULT},
                ),
            ),
        ),
    )


def _index_shift_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    inline_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
    source_value_guards: tuple[Guard, ...] = (),
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    inline_descriptor = _descriptor(inline_descriptor_key)
    preserve_descriptor = (
        _descriptor(preserve_descriptor_key)
        if preserve_descriptor_key is not None
        else None
    )
    rules: list[DescriptorRule] = [
        _narrow_address_sgpr_binary_rule(
            source_op,
            _INDEX,
            sgpr_descriptor,
            source_value_guards=source_value_guards,
        )
    ]
    if preserve_descriptor is not None:
        rules.append(
            _vgpr_constant_shift_rule(
                source_op,
                _INDEX,
                preserve_descriptor,
                ADDRESS_VGPR_MATERIALIZER,
                preserve_source_register=True,
                preserved_source_materializer=ADDRESS_SGPR_MATERIALIZER,
                register_unit_count=1,
                source_value_guards=source_value_guards,
            )
        )
    rules.append(
        _vgpr_constant_shift_rule(
            source_op,
            _INDEX,
            inline_descriptor,
            ADDRESS_VGPR_MATERIALIZER,
            register_unit_count=1,
            source_value_guards=source_value_guards,
        )
    )
    rules.append(
        _vgpr_binary_rule(
            source_op,
            _INDEX,
            vgpr_descriptor,
            ADDRESS_VGPR_MATERIALIZER,
            descriptor_lhs="shift",
            descriptor_rhs="value",
            source_lhs="rhs",
            source_rhs="lhs",
            register_unit_count=1,
            source_value_guards=source_value_guards,
        )
    )
    return tuple(rules)


def _index_div_power_of_two_sgpr_rule() -> DescriptorRule:
    move = _descriptor("amdgpu.s_mov_b32")
    shift = _descriptor("amdgpu.s_lshr_b32")
    return DescriptorRule(
        source_op=index.index_div,
        descriptor=shift,
        guards=(
            *_typed_binary_guards(
                _INDEX,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 1),
            Guard.value_materializable("lhs", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.value_unsigned_bit_count(
                "lhs",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                "rhs",
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                _UINT32_MAX,
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(move),
            Guard.descriptor_available(shift),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("shift")},
                result_types={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64_log2("rhs")},
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "lhs": _materialized_operand("lhs", ADDRESS_SGPR_MATERIALIZER),
                    "rhs": ValueRef.temporary("shift"),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _index_div_power_of_two_vgpr_rule() -> DescriptorRule:
    shift = _descriptor("amdgpu.v_lshrrev_b32.src0_inline")
    return DescriptorRule(
        source_op=index.index_div,
        descriptor=shift,
        guards=(
            *_typed_binary_guards(
                _INDEX,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_materializable("lhs", ADDRESS_VGPR_MATERIALIZER.name),
            Guard.value_unsigned_bit_count(
                "lhs",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                "rhs",
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                _UINT32_MAX,
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(shift),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "value": _materialized_operand(
                        "lhs",
                        ADDRESS_VGPR_MATERIALIZER,
                    )
                },
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64_log2("rhs")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_rem_power_of_two_sgpr_rule() -> DescriptorRule:
    move = _descriptor("amdgpu.s_mov_b32")
    bitwise_and = _descriptor("amdgpu.s_and_b32")
    return DescriptorRule(
        source_op=index.index_rem,
        descriptor=bitwise_and,
        guards=(
            *_typed_binary_guards(
                _INDEX,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 1),
            Guard.value_materializable("lhs", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.value_unsigned_bit_count(
                "lhs",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                "rhs",
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                _UINT32_MAX,
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(move),
            Guard.descriptor_available(bitwise_and),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("mask")},
                result_types={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64_minus_one("rhs")},
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "lhs": _materialized_operand("lhs", ADDRESS_SGPR_MATERIALIZER),
                    "rhs": ValueRef.temporary("mask"),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _index_rem_power_of_two_vgpr_rule() -> DescriptorRule:
    bitwise_and = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=index.index_rem,
        descriptor=bitwise_and,
        guards=(
            *_typed_binary_guards(
                _INDEX,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_materializable("lhs", ADDRESS_VGPR_MATERIALIZER.name),
            Guard.value_unsigned_bit_count(
                "lhs",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                "rhs",
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                _UINT32_MAX,
                diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(bitwise_and),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "rhs": _materialized_operand(
                        "lhs",
                        ADDRESS_VGPR_MATERIALIZER,
                    )
                },
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64_minus_one("rhs")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_sgpr_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.s_mul_i32")
    add = _descriptor("amdgpu.s_add_u32")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            Guard.value_type("a", _INDEX),
            Guard.value_type("b", _INDEX),
            Guard.value_type("c", _INDEX),
            Guard.value_type("result", _INDEX),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 1),
            Guard.value_materializable("a", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.value_materializable("b", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.value_materializable("c", ADDRESS_SGPR_MATERIALIZER.name),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": _materialized_operand("a", ADDRESS_SGPR_MATERIALIZER),
                    "rhs": _materialized_operand("b", ADDRESS_SGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": ValueRef.result("result")},
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": _materialized_operand("c", ADDRESS_SGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_ctpopi_i32_rules() -> tuple[DescriptorRule, ...]:
    scalar_descriptor = _descriptor("amdgpu.s_bcnt1_i32_b32")
    vector_descriptor = _descriptor("amdgpu.v_bcnt_u32_b32.src1_zero")
    typed_guards = (
        Guard.value_type("input", _I32),
        Guard.value_type("result", _I32),
    )
    return (
        DescriptorRule(
            source_op=scalar_bitwise.scalar_ctpopi,
            descriptor=scalar_descriptor,
            guards=(
                *typed_guards,
                Guard.low_value_register_class("input", "amdgpu.sgpr"),
                Guard.low_value_register_class("result", "amdgpu.sgpr"),
                Guard.descriptor_available(scalar_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=scalar_descriptor,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        ),
        DescriptorRule(
            source_op=scalar_bitwise.scalar_ctpopi,
            descriptor=vector_descriptor,
            guards=(
                *typed_guards,
                Guard.low_value_register_class("result", "amdgpu.vgpr"),
                Guard.value_materializable("input", I32_VGPR_MATERIALIZER.name),
                Guard.descriptor_available(vector_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=vector_descriptor,
                    operands={
                        "input": _materialized_operand("input", I32_VGPR_MATERIALIZER)
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        ),
    )


def _rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for source_op, suffix in (
        (scalar_arithmetic.scalar_minsi, "min_i32"),
        (scalar_arithmetic.scalar_maxsi, "max_i32"),
        (scalar_arithmetic.scalar_minui, "min_u32"),
        (scalar_arithmetic.scalar_maxui, "max_u32"),
    ):
        rules.extend(_i32_encoded_binary_rules(source_op, suffix))
    rules.extend(
        _unsigned_carry_borrow_rules(
            scalar_arithmetic.scalar_addcarryui,
            value_result="sum",
            predicate_result="carry",
            descriptor_predicate_result="carry",
            sgpr_descriptor_key="amdgpu.s_add_co_u32",
            vgpr_descriptor_key="amdgpu.v_add_co_u32",
            vgpr_value_result="sum",
        )
    )
    rules.extend(
        _unsigned_carry_borrow_rules(
            scalar_arithmetic.scalar_subborrowui,
            value_result="difference",
            predicate_result="borrow",
            descriptor_predicate_result="borrow",
            sgpr_descriptor_key="amdgpu.s_sub_co_u32",
            vgpr_descriptor_key="amdgpu.v_sub_co_u32",
            vgpr_value_result="difference",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_addi,
            "amdgpu.s_add_u32",
            "amdgpu.v_add_u32",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_subi,
            "amdgpu.s_sub_u32",
            "amdgpu.v_sub_u32",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_muli,
            "amdgpu.s_mul_i32",
            "amdgpu.v_mul_lo_u32",
        )
    )
    rules.extend(_scalar_fmai_rules())
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b32",
        )
    )
    rules.extend(
        _i1_mixed_bool_mask_rules(
            scalar_bitwise.scalar_andi,
            "and",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b64",
        )
    )
    rules.append(
        _sgpr_binary_rule(
            scalar_bitwise.scalar_andi,
            _I64,
            _descriptor("amdgpu.s_and_b64"),
        )
    )
    rules.append(
        _i64_vgpr_per_lane_binary_rule(
            scalar_bitwise.scalar_andi,
            "amdgpu.v_and_b32",
        )
    )
    rules.extend(_i32_encoded_binary_rules(scalar_bitwise.scalar_andi, "and_b32"))
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b32",
        )
    )
    rules.extend(
        _i1_mixed_bool_mask_rules(
            scalar_bitwise.scalar_ori,
            "or",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b64",
        )
    )
    rules.append(
        _sgpr_binary_rule(
            scalar_bitwise.scalar_ori,
            _I64,
            _descriptor("amdgpu.s_or_b64"),
        )
    )
    rules.append(
        _i64_vgpr_per_lane_binary_rule(
            scalar_bitwise.scalar_ori,
            "amdgpu.v_or_b32",
        )
    )
    rules.extend(_i32_encoded_binary_rules(scalar_bitwise.scalar_ori, "or_b32"))
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b32",
        )
    )
    rules.extend(
        _i1_mixed_bool_mask_rules(
            scalar_bitwise.scalar_xori,
            "xor",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b64",
        )
    )
    rules.append(
        _sgpr_binary_rule(
            scalar_bitwise.scalar_xori,
            _I64,
            _descriptor("amdgpu.s_xor_b64"),
        )
    )
    rules.append(
        _i64_vgpr_per_lane_binary_rule(
            scalar_bitwise.scalar_xori,
            "amdgpu.v_xor_b32",
        )
    )
    rules.extend(_i32_encoded_binary_rules(scalar_bitwise.scalar_xori, "xor_b32"))
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.src0_inline",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.src0_inline",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.src0_inline",
        )
    )
    rules.extend(_i64_word_extract_rules(scalar_bitwise.scalar_bitfield_extracts))
    rules.extend(_i64_word_extract_rules(scalar_bitwise.scalar_bitfield_extractu))
    rules.extend(
        _i32_bitfield_extract_rules(
            scalar_bitwise.scalar_bitfield_extracts,
            sgpr_descriptor_key="amdgpu.s_bfe_i32.lit",
            vgpr_descriptor_key="amdgpu.v_bfe_i32.offset_width_inline",
            top_aligned_vgpr_descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        )
    )
    rules.extend(
        _i32_bitfield_extract_rules(
            scalar_bitwise.scalar_bitfield_extractu,
            sgpr_descriptor_key="amdgpu.s_bfe_u32.lit",
            vgpr_descriptor_key="amdgpu.v_bfe_u32.offset_width_inline",
            top_aligned_vgpr_descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        )
    )
    rules.extend(
        _address_binary_rules(
            index.index_add,
            "amdgpu.s_add_u32",
            "amdgpu.v_add_u32",
            # VOP2 addition admits a uniform SRC0 and commutes its sources.
            # Keeping that operand scalar also exposes its immediate forms.
            sgpr_operand_orders=(("lhs", "rhs"), ("rhs", "lhs")),
        )
    )
    rules.extend(
        _address_binary_rules(
            index.index_sub,
            "amdgpu.s_sub_u32",
            "amdgpu.v_sub_u32",
            sgpr_operand_orders=(("lhs", "rhs"),),
        )
    )
    rules.extend(
        _commutative_index_binary_rules(
            index.index_mul,
            "amdgpu.s_mul_i32",
            "amdgpu.v_mul_lo_u32",
        )
    )
    rules.extend(_address_scale_rules())
    rules.extend(
        (
            _index_div_power_of_two_sgpr_rule(),
            _index_div_power_of_two_vgpr_rule(),
            _index_rem_power_of_two_sgpr_rule(),
            _index_rem_power_of_two_vgpr_rule(),
        )
    )
    rules.extend(integer_division_rules(_DESCRIPTOR_SET))
    rules.extend(
        _signed_index_minmax_rules(
            index.index_min, "amdgpu.s_min_i32", "amdgpu.v_min_i32"
        )
    )
    rules.extend(
        _signed_index_minmax_rules(
            index.index_max, "amdgpu.s_max_i32", "amdgpu.v_max_i32"
        )
    )
    rules.extend(
        _commutative_index_binary_rules(
            index.index_andi, "amdgpu.s_and_b32", "amdgpu.v_and_b32"
        )
    )
    rules.extend(
        _commutative_index_binary_rules(
            index.index_ori, "amdgpu.s_or_b32", "amdgpu.v_or_b32"
        )
    )
    rules.extend(
        _commutative_index_binary_rules(
            index.index_xori, "amdgpu.s_xor_b32", "amdgpu.v_xor_b32"
        )
    )
    rules.extend(
        _index_shift_rules(
            index.index_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.src0_inline",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _index_shift_rules(
            index.index_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.src0_inline",
            source_value_guards=(
                Guard.value_signed_bit_count(
                    "lhs",
                    32,
                    diagnostic=_ADDRESS_I32_DIAGNOSTIC,
                ),
            ),
        )
    )
    rules.extend(
        _index_shift_rules(
            index.index_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.src0_inline",
            source_value_guards=(
                Guard.value_i64_range(
                    "lhs",
                    -(2**31),
                    _UINT32_MAX,
                    diagnostic=_ADDRESS_32BIT_DIAGNOSTIC,
                ),
            ),
        )
    )
    rules.append(_index_madd_sgpr_rule())
    rules.extend(_scalar_ctpopi_i32_rules())
    rules.extend(index_bitwise_rules(_DESCRIPTOR_SET))
    rules.extend(
        _vector_predicate_bitwise_rule(source_op, descriptor_key)
        for source_op, descriptor_key in (
            (vector.vector_andi, "amdgpu.s_and_b64"),
            (vector.vector_ori, "amdgpu.s_or_b64"),
            (vector.vector_xori, "amdgpu.s_xor_b64"),
        )
    )
    rules.append(_vector_predicate_select_rule())
    rules.extend(_whole_value_select_recipe_rules())
    return tuple(rules)


AMDGPU_INTEGER_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scf": ALL_SCF_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_INTEGER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.integer",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/integer.h",
    materializers=(
        ADDRESS_SGPR_MATERIALIZER,
        I32_VGPR_MATERIALIZER,
        I64_VGPR_MATERIALIZER,
        ADDRESS_VGPR_MATERIALIZER,
        I1_NATIVE_MASK_MATERIALIZER,
    ),
    cases=_rules(),
)
