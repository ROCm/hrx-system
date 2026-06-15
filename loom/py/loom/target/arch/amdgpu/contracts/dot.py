# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU vector dot-product source-to-low contracts."""

from __future__ import annotations

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_fma_f32",
    "amdgpu.v_lshlrev_b32.lit",
    "amdgpu.v_mul_f32",
    "amdgpu.v_dot2_f32_f16",
    "amdgpu.v_dot2_f32_bf16",
    "amdgpu.v_dot4_i32_i8",
    "amdgpu.v_dot4_u32_u8",
    "amdgpu.v_dot4_i32_iu8.u8s8",
    "amdgpu.v_dot4_i32_iu8.s8u8",
    "amdgpu.v_dot8_i32_i4",
    "amdgpu.v_dot8_i32_iu4.s4u4",
    "amdgpu.v_dot8_i32_iu4.u4s4",
    "amdgpu.v_dot8_u32_u4",
    "amdgpu.v_dot4_f32_fp8_bf8",
    "amdgpu.v_dot4_f32_bf8_fp8",
    "amdgpu.v_dot4_f32_fp8_fp8",
    "amdgpu.v_dot4_f32_bf8_bf8",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.dot",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_VEC_F32 = Vector(
    "f32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_F32 = Scalar("f32")
_VEC_I8 = Vector(
    "i8",
    minimum_lanes=4,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_I8_LANES",
)
_VEC_I32_PACKED = Vector(
    "i32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS",
)
_VEC_F16_PACKED = Vector(
    "f16",
    minimum_lanes=2,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
)
_VEC_BF16_PACKED = Vector(
    "bf16",
    minimum_lanes=2,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
)

_VGPR = "amdgpu.vgpr"
_BF16_HIGH_MASK = 0xFFFF0000

_KIND_DIAGNOSTIC = GuardDiagnostic(
    subject_role="kind",
    subject_name="vector.dot",
    constraint_key="amdgpu.dot.kind",
)
_PACKED_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.packed_dot",
    constraint_key="amdgpu.dot.packed_descriptor",
)
_DOT4I_MIXED_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_dot4_i32_iu8",
    constraint_key="amdgpu.dot4i.mixed_descriptor",
)
_DOT8I4_MIXED_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_dot8_i32_iu4",
    constraint_key="amdgpu.dot8i4.mixed_descriptor",
)
_DOT2F_F16_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_dot2_f32_f16",
    constraint_key="amdgpu.dot2f.f16_descriptor",
)
_DOT2F_BF16_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_dot2_f32_bf16",
    constraint_key="amdgpu.dot2f.bf16_descriptor",
)
_DOT2F_BF16_FALLBACK_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.bf16_dot2f_fallback",
    constraint_key="amdgpu.dot2f.bf16_fallback_descriptors",
)
_DOT4F8_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_dot4_f32",
    constraint_key="amdgpu.dot4f8.descriptor",
)
_DOTF_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="amdgpu.v_fma_f32",
    constraint_key="amdgpu.dotf.descriptor",
)

_DOTF_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dotf.lhs_type",
)
_DOTF_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dotf.rhs_type",
)
_DOTF_ACC_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="accumulator",
    constraint_key="amdgpu.dotf.accumulator_type",
)
_DOTF_RESULT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="result",
    constraint_key="amdgpu.dotf.result_type",
)
_RESULT_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="result",
    constraint_key="amdgpu.dotf.result_vgpr",
)
_PACKED_ACC_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="accumulator",
    constraint_key="amdgpu.packed_dot.accumulator_vgpr",
)
_PACKED_RESULT_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="result",
    constraint_key="amdgpu.packed_dot.result_vgpr",
)
_LHS_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="lhs",
    constraint_key="amdgpu.packed_dot.lhs_vgpr",
)
_RHS_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="register-class",
    subject_name="rhs",
    constraint_key="amdgpu.packed_dot.rhs_vgpr",
)

_DOT4I_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dot4i.lhs_type",
)
_DOT4I_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dot4i.rhs_type",
)
_PACKED_I32_ACC_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="accumulator",
    constraint_key="amdgpu.packed_dot.i32_accumulator_type",
)
_PACKED_I32_RESULT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="result",
    constraint_key="amdgpu.packed_dot.i32_result_type",
)
_DOT8I4_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dot8i4.lhs_type",
)
_DOT8I4_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dot8i4.rhs_type",
)
_DOT2F_F16_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dot2f.f16_lhs_type",
)
_DOT2F_F16_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dot2f.f16_rhs_type",
)
_DOT2F_BF16_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dot2f.bf16_lhs_type",
)
_DOT2F_BF16_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dot2f.bf16_rhs_type",
)
_DOT2F_ACC_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="accumulator",
    constraint_key="amdgpu.dot2f.accumulator_type",
)
_DOT2F_RESULT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="result",
    constraint_key="amdgpu.dot2f.result_type",
)
_DOT4F8_LHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="lhs",
    constraint_key="amdgpu.dot4f8.lhs_type",
)
_DOT4F8_RHS_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="rhs",
    constraint_key="amdgpu.dot4f8.rhs_type",
)
_DOT4F8_ACC_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="accumulator",
    constraint_key="amdgpu.dot4f8.accumulator_type",
)
_DOT4F8_RESULT_DIAGNOSTIC = GuardDiagnostic(
    subject_role="type",
    subject_name="result",
    constraint_key="amdgpu.dot4f8.result_type",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _value_type(
    field: str,
    type_pattern: TypePattern,
    diagnostic: GuardDiagnostic,
) -> Guard:
    return Guard.value_type(field, type_pattern, diagnostic=diagnostic)


def _unit_count_eq(
    field: str,
    other_field: str,
    diagnostic: GuardDiagnostic,
) -> Guard:
    return Guard.low_value_register_unit_count_eq(
        field,
        other_field,
        diagnostic=diagnostic,
    )


def _vgpr(field: str, diagnostic: GuardDiagnostic) -> Guard:
    return Guard.low_value_register_class(field, _VGPR, diagnostic=diagnostic)


def _dotf_rule() -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_fma_f32")
    return DescriptorRule(
        source_op=vector.vector_dotf,
        descriptor=descriptor,
        guards=(
            _value_type("lhs", _VEC_F32, _DOTF_LHS_DIAGNOSTIC),
            _value_type("rhs", _VEC_F32, _DOTF_RHS_DIAGNOSTIC),
            _unit_count_eq("lhs", "rhs", _DOTF_RHS_DIAGNOSTIC),
            _value_type("init", _F32, _DOTF_ACC_DIAGNOSTIC),
            _value_type("result", _F32, _DOTF_RESULT_DIAGNOSTIC),
            _vgpr("result", _RESULT_VGPR_DIAGNOSTIC),
            Guard.descriptor_available(
                descriptor,
                diagnostic=_DOTF_DESCRIPTOR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("lhs"),
                    "b": ValueRef.operand("rhs"),
                    "c": ValueRef.operand("init"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="c",
            ),
        ),
    )


def _dotf_zero_init_rule() -> DescriptorRule:
    seed_descriptor = _descriptor("amdgpu.v_mul_f32")
    accumulate_descriptor = _descriptor("amdgpu.v_fma_f32")
    return DescriptorRule(
        source_op=vector.vector_dotf,
        descriptor=accumulate_descriptor,
        guards=(
            _value_type("lhs", _VEC_F32, _DOTF_LHS_DIAGNOSTIC),
            _value_type("rhs", _VEC_F32, _DOTF_RHS_DIAGNOSTIC),
            _unit_count_eq("lhs", "rhs", _DOTF_RHS_DIAGNOSTIC),
            _value_type("init", _F32, _DOTF_ACC_DIAGNOSTIC),
            Guard.value_f64_equals("init", 0.0, diagnostic=_DOTF_ACC_DIAGNOSTIC),
            _value_type("result", _F32, _DOTF_RESULT_DIAGNOSTIC),
            _vgpr("result", _RESULT_VGPR_DIAGNOSTIC),
            Guard.descriptor_available(
                seed_descriptor,
                diagnostic=_DOTF_DESCRIPTOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(
                accumulate_descriptor,
                diagnostic=_DOTF_DESCRIPTOR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=seed_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("seed")},
                result_types={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.FIRST_LANE,
            ),
            EmitDescriptorOp(
                descriptor=accumulate_descriptor,
                operands={
                    "a": ValueRef.operand("lhs"),
                    "b": ValueRef.operand("rhs"),
                    "c": ValueRef.temporary("seed"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="c",
                skip_first_lane=True,
            ),
        ),
    )


def _dot2f_rule(
    *,
    source_type: TypePattern,
    lhs_diagnostic: GuardDiagnostic,
    rhs_diagnostic: GuardDiagnostic,
    descriptor_key: str,
    descriptor_diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    return _packed_dot_rule(
        source_op=vector.vector_dot2f,
        descriptor_key=descriptor_key,
        guards=(
            _value_type("lhs", source_type, lhs_diagnostic),
            Guard.value_static_dim0_multiple(
                "lhs",
                2,
                diagnostic=lhs_diagnostic,
            ),
            _value_type("rhs", source_type, rhs_diagnostic),
            Guard.value_static_dim0_multiple(
                "rhs",
                2,
                diagnostic=rhs_diagnostic,
            ),
            _unit_count_eq("lhs", "rhs", rhs_diagnostic),
            _value_type("acc", _VEC_F32, _DOT2F_ACC_DIAGNOSTIC),
            _unit_count_eq("lhs", "acc", _DOT2F_ACC_DIAGNOSTIC),
            _value_type("result", _VEC_F32, _DOT2F_RESULT_DIAGNOSTIC),
            _unit_count_eq("lhs", "result", _DOT2F_RESULT_DIAGNOSTIC),
            _vgpr("lhs", _LHS_VGPR_DIAGNOSTIC),
            _vgpr("rhs", _RHS_VGPR_DIAGNOSTIC),
            _vgpr("acc", _PACKED_ACC_VGPR_DIAGNOSTIC),
            _vgpr("result", _PACKED_RESULT_VGPR_DIAGNOSTIC),
        ),
        descriptor_diagnostic=descriptor_diagnostic,
    )


def _dot2f_bf16_fallback_rule() -> DescriptorRule:
    fma = _descriptor("amdgpu.v_fma_f32")
    shift_up = _descriptor("amdgpu.v_lshlrev_b32.lit")
    high_bits = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=fma,
        guards=(
            _value_type("lhs", _VEC_BF16_PACKED, _DOT2F_BF16_LHS_DIAGNOSTIC),
            Guard.value_static_dim0_multiple(
                "lhs",
                2,
                diagnostic=_DOT2F_BF16_LHS_DIAGNOSTIC,
            ),
            _value_type("rhs", _VEC_BF16_PACKED, _DOT2F_BF16_RHS_DIAGNOSTIC),
            Guard.value_static_dim0_multiple(
                "rhs",
                2,
                diagnostic=_DOT2F_BF16_RHS_DIAGNOSTIC,
            ),
            _unit_count_eq("lhs", "rhs", _DOT2F_BF16_RHS_DIAGNOSTIC),
            _value_type("acc", _VEC_F32, _DOT2F_ACC_DIAGNOSTIC),
            _unit_count_eq("lhs", "acc", _DOT2F_ACC_DIAGNOSTIC),
            _value_type("result", _VEC_F32, _DOT2F_RESULT_DIAGNOSTIC),
            _unit_count_eq("lhs", "result", _DOT2F_RESULT_DIAGNOSTIC),
            _vgpr("lhs", _LHS_VGPR_DIAGNOSTIC),
            _vgpr("rhs", _RHS_VGPR_DIAGNOSTIC),
            _vgpr("acc", _PACKED_ACC_VGPR_DIAGNOSTIC),
            _vgpr("result", _PACKED_RESULT_VGPR_DIAGNOSTIC),
            Guard.descriptor_available(
                fma,
                diagnostic=_DOT2F_BF16_FALLBACK_DESCRIPTOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(
                shift_up,
                diagnostic=_DOT2F_BF16_FALLBACK_DESCRIPTOR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(
                high_bits,
                diagnostic=_DOT2F_BF16_FALLBACK_DESCRIPTOR_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift_up,
                operands={"value": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_low")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": 16},
                form=DescriptorEmitForm.PER_LANE,
            ),
            EmitDescriptorOp(
                descriptor=shift_up,
                operands={"value": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_low")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": 16},
                form=DescriptorEmitForm.PER_LANE,
            ),
            EmitDescriptorOp(
                descriptor=high_bits,
                operands={"rhs": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_high")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": _BF16_HIGH_MASK},
                form=DescriptorEmitForm.PER_LANE,
            ),
            EmitDescriptorOp(
                descriptor=high_bits,
                operands={"rhs": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_high")},
                result_types={"dst": ValueRef.result("result")},
                immediates={"imm32": _BF16_HIGH_MASK},
                form=DescriptorEmitForm.PER_LANE,
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("lhs_low"),
                    "b": ValueRef.temporary("rhs_low"),
                    "c": ValueRef.operand("acc"),
                },
                results={"dst": ValueRef.temporary("partial")},
                result_types={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE,
            ),
            EmitDescriptorOp(
                descriptor=fma,
                operands={
                    "a": ValueRef.temporary("lhs_high"),
                    "b": ValueRef.temporary("rhs_high"),
                    "c": ValueRef.temporary("partial"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _dot4i_rule(
    kind: str,
    descriptor_key: str,
    *,
    descriptor_diagnostic: GuardDiagnostic = _PACKED_DESCRIPTOR_DIAGNOSTIC,
) -> DescriptorRule:
    return _packed_dot_rule(
        source_op=vector.vector_dot4i,
        descriptor_key=descriptor_key,
        guards=(
            Guard.enum_attr_equals("kind", kind, diagnostic=_KIND_DIAGNOSTIC),
            _value_type("lhs", _VEC_I8, _DOT4I_LHS_DIAGNOSTIC),
            Guard.value_static_dim0_multiple(
                "lhs",
                4,
                diagnostic=_DOT4I_LHS_DIAGNOSTIC,
            ),
            _value_type("rhs", _VEC_I8, _DOT4I_RHS_DIAGNOSTIC),
            Guard.value_static_dim0_multiple(
                "rhs",
                4,
                diagnostic=_DOT4I_RHS_DIAGNOSTIC,
            ),
            _unit_count_eq("lhs", "rhs", _DOT4I_RHS_DIAGNOSTIC),
            _value_type("acc", _VEC_I32_PACKED, _PACKED_I32_ACC_DIAGNOSTIC),
            _unit_count_eq("lhs", "acc", _PACKED_I32_ACC_DIAGNOSTIC),
            _value_type("result", _VEC_I32_PACKED, _PACKED_I32_RESULT_DIAGNOSTIC),
            _unit_count_eq("lhs", "result", _PACKED_I32_RESULT_DIAGNOSTIC),
            _vgpr("lhs", _LHS_VGPR_DIAGNOSTIC),
            _vgpr("rhs", _RHS_VGPR_DIAGNOSTIC),
            _vgpr("acc", _PACKED_ACC_VGPR_DIAGNOSTIC),
            _vgpr("result", _PACKED_RESULT_VGPR_DIAGNOSTIC),
        ),
        descriptor_diagnostic=descriptor_diagnostic,
    )


def _dot8i4_rule(
    kind: str,
    descriptor_key: str,
    *,
    descriptor_diagnostic: GuardDiagnostic = _PACKED_DESCRIPTOR_DIAGNOSTIC,
) -> DescriptorRule:
    return _packed_dot_rule(
        source_op=vector.vector_dot8i4,
        descriptor_key=descriptor_key,
        guards=(
            Guard.enum_attr_equals("kind", kind, diagnostic=_KIND_DIAGNOSTIC),
            _value_type("lhs", _VEC_I32_PACKED, _DOT8I4_LHS_DIAGNOSTIC),
            _value_type("rhs", _VEC_I32_PACKED, _DOT8I4_RHS_DIAGNOSTIC),
            _unit_count_eq("lhs", "rhs", _DOT8I4_RHS_DIAGNOSTIC),
            _value_type("acc", _VEC_I32_PACKED, _PACKED_I32_ACC_DIAGNOSTIC),
            _unit_count_eq("lhs", "acc", _PACKED_I32_ACC_DIAGNOSTIC),
            _value_type("result", _VEC_I32_PACKED, _PACKED_I32_RESULT_DIAGNOSTIC),
            _unit_count_eq("lhs", "result", _PACKED_I32_RESULT_DIAGNOSTIC),
            _vgpr("lhs", _LHS_VGPR_DIAGNOSTIC),
            _vgpr("rhs", _RHS_VGPR_DIAGNOSTIC),
            _vgpr("acc", _PACKED_ACC_VGPR_DIAGNOSTIC),
            _vgpr("result", _PACKED_RESULT_VGPR_DIAGNOSTIC),
        ),
        descriptor_diagnostic=descriptor_diagnostic,
    )


def _dot4f8_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _packed_dot_rule(
        source_op=vector.vector_dot4f8,
        descriptor_key=descriptor_key,
        guards=(
            Guard.enum_attr_equals("kind", kind, diagnostic=_KIND_DIAGNOSTIC),
            _value_type("lhs", _VEC_I32_PACKED, _DOT4F8_LHS_DIAGNOSTIC),
            _value_type("rhs", _VEC_I32_PACKED, _DOT4F8_RHS_DIAGNOSTIC),
            _unit_count_eq("lhs", "rhs", _DOT4F8_RHS_DIAGNOSTIC),
            _value_type("acc", _VEC_F32, _DOT4F8_ACC_DIAGNOSTIC),
            _unit_count_eq("lhs", "acc", _DOT4F8_ACC_DIAGNOSTIC),
            _value_type("result", _VEC_F32, _DOT4F8_RESULT_DIAGNOSTIC),
            _unit_count_eq("lhs", "result", _DOT4F8_RESULT_DIAGNOSTIC),
            _vgpr("lhs", _LHS_VGPR_DIAGNOSTIC),
            _vgpr("rhs", _RHS_VGPR_DIAGNOSTIC),
            _vgpr("acc", _PACKED_ACC_VGPR_DIAGNOSTIC),
            _vgpr("result", _PACKED_RESULT_VGPR_DIAGNOSTIC),
        ),
        descriptor_diagnostic=_DOT4F8_DESCRIPTOR_DIAGNOSTIC,
    )


def _packed_dot_rule(
    *,
    source_op: Op,
    descriptor_key: str,
    guards: tuple[Guard, ...],
    descriptor_diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *guards,
            Guard.descriptor_available(
                descriptor,
                diagnostic=descriptor_diagnostic,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                    "acc": ValueRef.operand("acc"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


AMDGPU_DOT_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_DOT_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.dot",
    descriptor_set=_DESCRIPTOR_SET,
    c_source_includes=("loom/target/arch/amdgpu/lower/kinds.h",),
    cases=(
        _dotf_zero_init_rule(),
        _dotf_rule(),
        _dot2f_rule(
            source_type=_VEC_F16_PACKED,
            lhs_diagnostic=_DOT2F_F16_LHS_DIAGNOSTIC,
            rhs_diagnostic=_DOT2F_F16_RHS_DIAGNOSTIC,
            descriptor_key="amdgpu.v_dot2_f32_f16",
            descriptor_diagnostic=_DOT2F_F16_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot2f_rule(
            source_type=_VEC_BF16_PACKED,
            lhs_diagnostic=_DOT2F_BF16_LHS_DIAGNOSTIC,
            rhs_diagnostic=_DOT2F_BF16_RHS_DIAGNOSTIC,
            descriptor_key="amdgpu.v_dot2_f32_bf16",
            descriptor_diagnostic=_DOT2F_BF16_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot2f_bf16_fallback_rule(),
        _dot4i_rule("s8s8", "amdgpu.v_dot4_i32_i8"),
        _dot4i_rule("u8u8", "amdgpu.v_dot4_u32_u8"),
        _dot4i_rule(
            "u8s8",
            "amdgpu.v_dot4_i32_iu8.u8s8",
            descriptor_diagnostic=_DOT4I_MIXED_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot4i_rule(
            "s8u8",
            "amdgpu.v_dot4_i32_iu8.s8u8",
            descriptor_diagnostic=_DOT4I_MIXED_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot8i4_rule("s4s4", "amdgpu.v_dot8_i32_i4"),
        _dot8i4_rule(
            "s4u4",
            "amdgpu.v_dot8_i32_iu4.s4u4",
            descriptor_diagnostic=_DOT8I4_MIXED_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot8i4_rule(
            "u4s4",
            "amdgpu.v_dot8_i32_iu4.u4s4",
            descriptor_diagnostic=_DOT8I4_MIXED_DESCRIPTOR_DIAGNOSTIC,
        ),
        _dot8i4_rule("u4u4", "amdgpu.v_dot8_u32_u4"),
        _dot4f8_rule("fp8bf8", "amdgpu.v_dot4_f32_fp8_bf8"),
        _dot4f8_rule("bf8fp8", "amdgpu.v_dot4_f32_bf8_fp8"),
        _dot4f8_rule("fp8fp8", "amdgpu.v_dot4_f32_fp8_fp8"),
        _dot4f8_rule("bf8bf8", "amdgpu.v_dot4_f32_bf8_bf8"),
    ),
)
