# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU scalar integer and index arithmetic source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_VGPR_MATERIALIZER,
    I1_NATIVE_MASK_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
    I64_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueMaterializer,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_cselect_b32",
    "amdgpu.s_cmp_lg_i32",
    "amdgpu.s_add_u32",
    "amdgpu.s_sub_u32",
    "amdgpu.s_mul_i32",
    "amdgpu.s_mul_hi_u32",
    "amdgpu.s_min_i32",
    "amdgpu.s_max_i32",
    "amdgpu.s_min_u32",
    "amdgpu.s_max_u32",
    "amdgpu.s_and_b32",
    "amdgpu.s_or_b32",
    "amdgpu.s_xor_b32",
    "amdgpu.s_and_b64",
    "amdgpu.s_or_b64",
    "amdgpu.s_xor_b64",
    "amdgpu.s_lshl_b64",
    "amdgpu.s_lshr_b64",
    "amdgpu.s_lshl_b32",
    "amdgpu.s_lshr_b32",
    "amdgpu.s_ashr_i32",
    "amdgpu.s_bfe_i32.lit",
    "amdgpu.s_bfe_u32.lit",
    "amdgpu.v_mov_b32",
    "amdgpu.v_add_u32",
    "amdgpu.v_sub_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_mul_hi_u32",
    "amdgpu.v_min_i32",
    "amdgpu.v_max_i32",
    "amdgpu.v_min_u32",
    "amdgpu.v_max_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_or_b32",
    "amdgpu.v_or_b32.lit",
    "amdgpu.v_xor_b32",
    "amdgpu.v_xor_b32.lit",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.lit",
    "amdgpu.v_lshlrev_b32.vop3_imm",
    "amdgpu.v_lshrrev_b32",
    "amdgpu.v_lshrrev_b32.lit",
    "amdgpu.v_ashrrev_i32",
    "amdgpu.v_ashrrev_i32.lit",
    "amdgpu.v_bfe_i32.offset_width_inline",
    "amdgpu.v_bfe_u32.offset_width_inline",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.integer",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I1 = Scalar("i1")
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
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> tuple[Guard, ...]:
    guards = [
        Guard.value_type("lhs", type_pattern),
        Guard.value_type("rhs", type_pattern),
        Guard.value_type("result", type_pattern),
    ]
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
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
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
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
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
            Guard.value_materializable("lhs", materializer.name),
            Guard.value_materializable("rhs", materializer.name),
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


def _vgpr_literal_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    materializer: ValueMaterializer,
    *,
    preserve_source_register: bool = False,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    value_guard = (
        Guard.low_value_register_class("lhs", "amdgpu.sgpr")
        if preserve_source_register
        else Guard.value_materializable("lhs", materializer.name)
    )
    value_operand = (
        ValueRef.operand("lhs")
        if preserve_source_register
        else _materialized_operand("lhs", materializer)
    )
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
            value_guard,
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


def _i32_vgpr_literal_binary_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    literal_source: str,
    nonliteral_source: str,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_I32),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_materializable(nonliteral_source, I32_VGPR_MATERIALIZER.name),
            Guard.value_exact_i64(
                literal_source,
                diagnostic=_I32_LITERAL_EXACT_DIAGNOSTIC,
            ),
            Guard.value_signed_bit_count(
                literal_source,
                32,
                diagnostic=_I32_LITERAL_BITS_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _materialized_operand(
                        nonliteral_source,
                        I32_VGPR_MATERIALIZER,
                    )
                },
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.i32_as_u32_bits(literal_source)},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _i32_sgpr_vgpr_literal_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    literal_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    literal_descriptor = _descriptor(literal_descriptor_key)
    return (
        _sgpr_binary_rule(source_op, _I32, sgpr_descriptor),
        _i32_vgpr_literal_binary_rule(
            source_op,
            literal_descriptor,
            literal_source="lhs",
            nonliteral_source="rhs",
        ),
        _i32_vgpr_literal_binary_rule(
            source_op,
            literal_descriptor,
            literal_source="rhs",
            nonliteral_source="lhs",
        ),
        _vgpr_binary_rule(
            source_op,
            _I32,
            vgpr_descriptor,
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
    literal_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    literal_descriptor = _descriptor(literal_descriptor_key)
    preserve_rules = (
        (
            _vgpr_literal_shift_rule(
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
        _vgpr_literal_shift_rule(
            source_op,
            _I32,
            literal_descriptor,
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


def _i32_bitfield_extract_rules(
    source_op: Op,
    *,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
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
                        "imm32": AttrProject.i64_attrs_pack_consecutive(
                            "offset",
                            count=2,
                            bit_width=8,
                        )
                    },
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


def _address_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    return tuple(
        _sgpr_binary_rule(
            source_op,
            type_pattern,
            sgpr_descriptor,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        )
        for type_pattern in (_INDEX, _OFFSET)
    ) + tuple(
        _vgpr_binary_rule(
            source_op,
            type_pattern,
            vgpr_descriptor,
            ADDRESS_VGPR_MATERIALIZER,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        )
        for type_pattern in (_INDEX, _OFFSET)
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
                Guard.low_value_register_class("index", "amdgpu.sgpr"),
                Guard.low_value_register_class("stride", "amdgpu.sgpr"),
                Guard.descriptor_available(sgpr_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=sgpr_descriptor,
                    operands={
                        "lhs": ValueRef.operand("index"),
                        "rhs": ValueRef.operand("stride"),
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


def _address_shift_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    literal_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    literal_descriptor = _descriptor(literal_descriptor_key)
    preserve_descriptor = (
        _descriptor(preserve_descriptor_key)
        if preserve_descriptor_key is not None
        else None
    )
    rules: list[DescriptorRule] = []
    for type_pattern in (_INDEX, _OFFSET):
        rules.append(
            _sgpr_binary_rule(
                source_op,
                type_pattern,
                sgpr_descriptor,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            )
        )
        if preserve_descriptor is not None:
            rules.append(
                _vgpr_literal_shift_rule(
                    source_op,
                    type_pattern,
                    preserve_descriptor,
                    ADDRESS_VGPR_MATERIALIZER,
                    preserve_source_register=True,
                    unsigned_bit_count=32,
                    unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
                )
            )
        rules.append(
            _vgpr_literal_shift_rule(
                source_op,
                type_pattern,
                literal_descriptor,
                ADDRESS_VGPR_MATERIALIZER,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            )
        )
        rules.append(
            _vgpr_binary_rule(
                source_op,
                type_pattern,
                vgpr_descriptor,
                ADDRESS_VGPR_MATERIALIZER,
                descriptor_lhs="shift",
                descriptor_rhs="value",
                source_lhs="rhs",
                source_rhs="lhs",
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
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
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
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
                operands={"lhs": _DIRECT_LHS, "rhs": ValueRef.temporary("shift")},
                results={"dst": _RESULT},
            ),
        ),
    )


def _index_div_power_of_two_vgpr_rule() -> DescriptorRule:
    shift = _descriptor("amdgpu.v_lshrrev_b32.lit")
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
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
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
                operands={"lhs": _DIRECT_LHS, "rhs": ValueRef.temporary("mask")},
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


def _index_div_magic_guards(
    *,
    register_class: str,
    is_add: bool,
) -> tuple[Guard, ...]:
    if register_class == "amdgpu.sgpr":
        value_guards = (
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
        )
    else:
        value_guards = (
            Guard.value_materializable("lhs", ADDRESS_VGPR_MATERIALIZER.name),
            Guard.value_materializable("rhs", ADDRESS_VGPR_MATERIALIZER.name),
        )
    return (
        *_typed_binary_guards(
            _INDEX,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        ),
        Guard.low_value_register_class("result", register_class),
        Guard.value_unsigned_bit_count(
            "lhs",
            32,
            diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        ),
        Guard.value_exact_i64("rhs", diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC),
        Guard.value_i64_range(
            "rhs",
            2,
            _UINT32_MAX,
            diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
        ),
        Guard.value_u32_divisor_magic_is_add("rhs", is_add),
        *value_guards,
    )


def _index_div_magic_sgpr_emits(
    *,
    is_add: bool,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    move = _descriptor("amdgpu.s_mov_b32")
    multiply_hi = _descriptor("amdgpu.s_mul_hi_u32")
    subtract = _descriptor("amdgpu.s_sub_u32")
    shift = _descriptor("amdgpu.s_lshr_b32")
    add = _descriptor("amdgpu.s_add_u32")
    quotient_value = (
        ValueRef.temporary("adjusted_quotient")
        if is_add
        else ValueRef.temporary("quotient")
    )
    emits = [
        EmitDescriptorOp(
            descriptor=move,
            results={"dst": ValueRef.temporary("magic")},
            result_types={"dst": _RESULT},
            immediates={"imm32": ValueProject.u32_divisor_magic_multiplier("rhs")},
        ),
        EmitDescriptorOp(
            descriptor=multiply_hi,
            operands={"lhs": _DIRECT_LHS, "rhs": ValueRef.temporary("magic")},
            results={"dst": ValueRef.temporary("quotient")},
            result_types={"dst": _RESULT},
        ),
    ]
    if is_add:
        emits.extend(
            [
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={
                        "lhs": _DIRECT_LHS,
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("npq")},
                    result_types={"dst": _RESULT},
                ),
                EmitDescriptorOp(
                    descriptor=move,
                    results={"dst": ValueRef.temporary("one")},
                    result_types={"dst": _RESULT},
                    immediates={"imm32": 1},
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={
                        "lhs": ValueRef.temporary("npq"),
                        "rhs": ValueRef.temporary("one"),
                    },
                    results={"dst": ValueRef.temporary("npq_half")},
                    result_types={"dst": _RESULT},
                ),
                EmitDescriptorOp(
                    descriptor=add,
                    operands={
                        "lhs": ValueRef.temporary("npq_half"),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("adjusted_quotient")},
                    result_types={"dst": _RESULT},
                ),
            ]
        )
    emits.extend(
        [
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("post_shift")},
                result_types={"dst": _RESULT},
                immediates={"imm32": ValueProject.u32_divisor_magic_shift("rhs")},
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "lhs": quotient_value,
                    "rhs": ValueRef.temporary("post_shift"),
                },
                results={"dst": result},
                result_types={"dst": _RESULT},
            ),
        ]
    )
    return tuple(emits)


def _index_div_magic_sgpr_descriptors(*, is_add: bool) -> tuple[Descriptor, ...]:
    move = _descriptor("amdgpu.s_mov_b32")
    multiply_hi = _descriptor("amdgpu.s_mul_hi_u32")
    subtract = _descriptor("amdgpu.s_sub_u32")
    shift = _descriptor("amdgpu.s_lshr_b32")
    add = _descriptor("amdgpu.s_add_u32")
    return (
        (move, multiply_hi, subtract, shift, add)
        if is_add
        else (move, multiply_hi, shift)
    )


def _index_div_magic_vgpr_emits(
    *,
    is_add: bool,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    move = _descriptor("amdgpu.v_mov_b32")
    multiply_hi = _descriptor("amdgpu.v_mul_hi_u32")
    subtract = _descriptor("amdgpu.v_sub_u32")
    shift = _descriptor("amdgpu.v_lshrrev_b32.lit")
    add = _descriptor("amdgpu.v_add_u32")
    quotient_value = (
        ValueRef.temporary("adjusted_quotient")
        if is_add
        else ValueRef.temporary("quotient")
    )
    emits = [
        EmitDescriptorOp(
            descriptor=move,
            results={"dst": ValueRef.temporary("magic")},
            result_types={"dst": _RESULT},
            immediates={"imm32": ValueProject.u32_divisor_magic_multiplier("rhs")},
        ),
        EmitDescriptorOp(
            descriptor=multiply_hi,
            operands={
                "lhs": _materialized_operand("lhs", ADDRESS_VGPR_MATERIALIZER),
                "rhs": ValueRef.temporary("magic"),
            },
            results={"dst": ValueRef.temporary("quotient")},
            result_types={"dst": _RESULT},
            form=DescriptorEmitForm.OP,
        ),
    ]
    if is_add:
        emits.extend(
            [
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={
                        "lhs": _materialized_operand(
                            "lhs",
                            ADDRESS_VGPR_MATERIALIZER,
                        ),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("npq")},
                    result_types={"dst": _RESULT},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={"value": ValueRef.temporary("npq")},
                    results={"dst": ValueRef.temporary("npq_half")},
                    result_types={"dst": _RESULT},
                    immediates={"imm32": 1},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=add,
                    operands={
                        "lhs": ValueRef.temporary("npq_half"),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("adjusted_quotient")},
                    result_types={"dst": _RESULT},
                    form=DescriptorEmitForm.OP,
                ),
            ]
        )
    emits.append(
        EmitDescriptorOp(
            descriptor=shift,
            operands={"value": quotient_value},
            results={"dst": result},
            result_types={"dst": _RESULT},
            immediates={"imm32": ValueProject.u32_divisor_magic_shift("rhs")},
            form=DescriptorEmitForm.OP,
        )
    )
    return tuple(emits)


def _index_div_magic_vgpr_descriptors(*, is_add: bool) -> tuple[Descriptor, ...]:
    move = _descriptor("amdgpu.v_mov_b32")
    multiply_hi = _descriptor("amdgpu.v_mul_hi_u32")
    subtract = _descriptor("amdgpu.v_sub_u32")
    shift = _descriptor("amdgpu.v_lshrrev_b32.lit")
    add = _descriptor("amdgpu.v_add_u32")
    return (
        (move, multiply_hi, subtract, shift, add)
        if is_add
        else (move, multiply_hi, shift)
    )


def _index_div_magic_sgpr_rule(*, is_add: bool) -> DescriptorRule:
    multiply_hi = _descriptor("amdgpu.s_mul_hi_u32")
    return DescriptorRule(
        source_op=index.index_div,
        descriptor=multiply_hi,
        guards=(
            *_index_div_magic_guards(
                register_class="amdgpu.sgpr",
                is_add=is_add,
            ),
            *_descriptor_available_guards(
                *_index_div_magic_sgpr_descriptors(is_add=is_add)
            ),
        ),
        emit=_index_div_magic_sgpr_emits(is_add=is_add, result=_RESULT),
    )


def _index_div_magic_vgpr_rule(*, is_add: bool) -> DescriptorRule:
    multiply_hi = _descriptor("amdgpu.v_mul_hi_u32")
    return DescriptorRule(
        source_op=index.index_div,
        descriptor=multiply_hi,
        guards=(
            *_index_div_magic_guards(
                register_class="amdgpu.vgpr",
                is_add=is_add,
            ),
            *_descriptor_available_guards(
                *_index_div_magic_vgpr_descriptors(is_add=is_add)
            ),
        ),
        emit=_index_div_magic_vgpr_emits(is_add=is_add, result=_RESULT),
    )


def _index_rem_magic_sgpr_rule(*, is_add: bool) -> DescriptorRule:
    multiply_lo = _descriptor("amdgpu.s_mul_i32")
    subtract = _descriptor("amdgpu.s_sub_u32")
    return DescriptorRule(
        source_op=index.index_rem,
        descriptor=multiply_lo,
        guards=(
            *_index_div_magic_guards(
                register_class="amdgpu.sgpr",
                is_add=is_add,
            ),
            *_descriptor_available_guards(
                *_index_div_magic_sgpr_descriptors(is_add=is_add),
                multiply_lo,
                subtract,
            ),
        ),
        emit=(
            *_index_div_magic_sgpr_emits(
                is_add=is_add,
                result=ValueRef.temporary("quotient_final"),
            ),
            EmitDescriptorOp(
                descriptor=multiply_lo,
                operands={
                    "lhs": ValueRef.temporary("quotient_final"),
                    "rhs": _DIRECT_RHS,
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _RESULT},
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={"lhs": _DIRECT_LHS, "rhs": ValueRef.temporary("product")},
                results={"dst": _RESULT},
            ),
        ),
    )


def _index_rem_magic_vgpr_rule(*, is_add: bool) -> DescriptorRule:
    multiply_lo = _descriptor("amdgpu.v_mul_lo_u32")
    subtract = _descriptor("amdgpu.v_sub_u32")
    return DescriptorRule(
        source_op=index.index_rem,
        descriptor=multiply_lo,
        guards=(
            *_index_div_magic_guards(
                register_class="amdgpu.vgpr",
                is_add=is_add,
            ),
            *_descriptor_available_guards(
                *_index_div_magic_vgpr_descriptors(is_add=is_add),
                multiply_lo,
                subtract,
            ),
        ),
        emit=(
            *_index_div_magic_vgpr_emits(
                is_add=is_add,
                result=ValueRef.temporary("quotient_final"),
            ),
            EmitDescriptorOp(
                descriptor=multiply_lo,
                operands={
                    "lhs": ValueRef.temporary("quotient_final"),
                    "rhs": _materialized_operand("rhs", ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _RESULT},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={
                    "lhs": _materialized_operand("lhs", ADDRESS_VGPR_MATERIALIZER),
                    "rhs": ValueRef.temporary("product"),
                },
                results={"dst": _RESULT},
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
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("a", "amdgpu.sgpr"),
            Guard.low_value_register_class("b", "amdgpu.sgpr"),
            Guard.low_value_register_class("c", "amdgpu.sgpr"),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={"lhs": ValueRef.operand("a"), "rhs": ValueRef.operand("b")},
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": ValueRef.result("result")},
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
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
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b32",
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
    rules.extend(
        _i32_sgpr_vgpr_literal_rules(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b32",
            "amdgpu.v_and_b32",
            "amdgpu.v_and_b32.lit",
        )
    )
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b32",
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
    rules.extend(
        _i32_sgpr_vgpr_literal_rules(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b32",
            "amdgpu.v_or_b32",
            "amdgpu.v_or_b32.lit",
        )
    )
    rules.extend(
        _i1_scalar_bool_bitwise_rules(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b32",
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
    rules.extend(
        _i32_sgpr_vgpr_literal_rules(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b32",
            "amdgpu.v_xor_b32",
            "amdgpu.v_xor_b32.lit",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.lit",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.lit",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.lit",
        )
    )
    rules.extend(
        _i32_bitfield_extract_rules(
            scalar_bitwise.scalar_bitfield_extracts,
            sgpr_descriptor_key="amdgpu.s_bfe_i32.lit",
            vgpr_descriptor_key="amdgpu.v_bfe_i32.offset_width_inline",
        )
    )
    rules.extend(
        _i32_bitfield_extract_rules(
            scalar_bitwise.scalar_bitfield_extractu,
            sgpr_descriptor_key="amdgpu.s_bfe_u32.lit",
            vgpr_descriptor_key="amdgpu.v_bfe_u32.offset_width_inline",
        )
    )
    rules.extend(
        _address_rules(index.index_add, "amdgpu.s_add_u32", "amdgpu.v_add_u32")
    )
    rules.extend(
        _address_rules(index.index_sub, "amdgpu.s_sub_u32", "amdgpu.v_sub_u32")
    )
    rules.extend(
        _address_rules(
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
    for is_add in (False, True):
        rules.extend(
            (
                _index_div_magic_sgpr_rule(is_add=is_add),
                _index_div_magic_vgpr_rule(is_add=is_add),
                _index_rem_magic_sgpr_rule(is_add=is_add),
                _index_rem_magic_vgpr_rule(is_add=is_add),
            )
        )
    rules.extend(
        _address_rules(index.index_min, "amdgpu.s_min_i32", "amdgpu.v_min_i32")
    )
    rules.extend(
        _address_rules(index.index_max, "amdgpu.s_max_i32", "amdgpu.v_max_i32")
    )
    rules.extend(
        _address_rules(index.index_andi, "amdgpu.s_and_b32", "amdgpu.v_and_b32")
    )
    rules.extend(_address_rules(index.index_ori, "amdgpu.s_or_b32", "amdgpu.v_or_b32"))
    rules.extend(
        _address_rules(index.index_xori, "amdgpu.s_xor_b32", "amdgpu.v_xor_b32")
    )
    rules.extend(
        _address_shift_rules(
            index.index_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.lit",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _address_shift_rules(
            index.index_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.lit",
        )
    )
    rules.extend(
        _address_shift_rules(
            index.index_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.lit",
        )
    )
    rules.append(_index_madd_sgpr_rule())
    return tuple(rules)


AMDGPU_INTEGER_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}

AMDGPU_INTEGER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.integer",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/integer.h",
    materializers=(
        I32_VGPR_MATERIALIZER,
        I64_VGPR_MATERIALIZER,
        ADDRESS_VGPR_MATERIALIZER,
        I1_NATIVE_MASK_MATERIALIZER,
    ),
    cases=_rules(),
)
