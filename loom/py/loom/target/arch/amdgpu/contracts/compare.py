# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU scalar integer and index comparison source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import comparison as scalar
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_VGPR_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_CMP_I32_CASES = (
    ("eq", "amdgpu.s_cmp_eq_i32", "amdgpu.v_cmp_eq_i32"),
    ("ne", "amdgpu.s_cmp_lg_i32", "amdgpu.v_cmp_ne_i32"),
    ("slt", "amdgpu.s_cmp_lt_i32", "amdgpu.v_cmp_slt_i32"),
    ("sle", "amdgpu.s_cmp_le_i32", "amdgpu.v_cmp_sle_i32"),
    ("sgt", "amdgpu.s_cmp_gt_i32", "amdgpu.v_cmp_sgt_i32"),
    ("sge", "amdgpu.s_cmp_ge_i32", "amdgpu.v_cmp_sge_i32"),
    ("ult", "amdgpu.s_cmp_lt_u32", "amdgpu.v_cmp_ult_u32"),
    ("ule", "amdgpu.s_cmp_le_u32", "amdgpu.v_cmp_ule_u32"),
    ("ugt", "amdgpu.s_cmp_gt_u32", "amdgpu.v_cmp_ugt_u32"),
    ("uge", "amdgpu.s_cmp_ge_u32", "amdgpu.v_cmp_uge_u32"),
)

_CMP_I64_SCALAR_CASES = (
    ("eq", "amdgpu.s_cmp_eq_u64"),
    ("ne", "amdgpu.s_cmp_lg_u64"),
)

_COMPARE_DESCRIPTOR_KEYS = tuple(
    descriptor_key
    for _, scalar_descriptor_key, mask_descriptor_key in _CMP_I32_CASES
    for descriptor_key in (
        scalar_descriptor_key,
        mask_descriptor_key,
        f"{mask_descriptor_key}.src0_inline",
        f"{mask_descriptor_key}.src1_inline",
    )
)
_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_cselect_b32",
    *_COMPARE_DESCRIPTOR_KEYS,
    *(descriptor_key for _, descriptor_key in _CMP_I64_SCALAR_CASES),
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.compare",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I64 = Scalar("i64")
_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_I1 = Scalar("i1")
_I32_VGPR_LHS = ValueRef.operand("lhs", materializer=I32_VGPR_MATERIALIZER.name)
_I32_VGPR_RHS = ValueRef.operand("rhs", materializer=I32_VGPR_MATERIALIZER.name)
_ADDRESS_VGPR_LHS = ValueRef.operand("lhs", materializer=ADDRESS_VGPR_MATERIALIZER.name)
_ADDRESS_VGPR_RHS = ValueRef.operand("rhs", materializer=ADDRESS_VGPR_MATERIALIZER.name)
_SOURCE_INLINE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="source-inline-u32",
    constraint_key="amdgpu.source_inline_u32",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _scc_rule(
    source_op: Op,
    type_pattern: TypePattern,
    predicate: str,
    descriptor: Descriptor,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", type_pattern),
            Guard.value_type("rhs", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("result", "amdgpu.scc"),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"scc": ValueRef.result("result")},
            ),
        ),
    )


def _sgpr_bool_rule(
    source_op: Op,
    type_pattern: TypePattern,
    predicate: str,
    compare_descriptor: Descriptor,
) -> DescriptorRule:
    move_descriptor = _descriptor("amdgpu.s_mov_b32")
    select_descriptor = _descriptor("amdgpu.s_cselect_b32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select_descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", type_pattern),
            Guard.value_type("rhs", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count_eq("result", "lhs"),
            Guard.low_value_register_unit_count_eq("result", "rhs"),
            Guard.descriptor_available(compare_descriptor),
            Guard.descriptor_available(move_descriptor),
            Guard.descriptor_available(select_descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"scc": ValueRef.temporary("condition")},
                result_types={"scc": _I1},
            ),
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("false_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 0},
            ),
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("true_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 1},
            ),
            EmitDescriptorOp(
                descriptor=select_descriptor,
                operands={
                    "true_value": ValueRef.temporary("true_value"),
                    "false_value": ValueRef.temporary("false_value"),
                    "condition": ValueRef.temporary("condition"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _i64_scc_rule(
    source_op: Op,
    predicate: str,
    descriptor: Descriptor,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I64),
            Guard.value_type("rhs", _I64),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("lhs", 2),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("rhs", 2),
            Guard.low_value_register_class("result", "amdgpu.scc"),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"scc": ValueRef.result("result")},
            ),
        ),
    )


def _i64_sgpr_bool_rule(
    source_op: Op,
    predicate: str,
    compare_descriptor: Descriptor,
) -> DescriptorRule:
    move_descriptor = _descriptor("amdgpu.s_mov_b32")
    select_descriptor = _descriptor("amdgpu.s_cselect_b32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select_descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I64),
            Guard.value_type("rhs", _I64),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("lhs", 2),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("rhs", 2),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 1),
            Guard.descriptor_available(compare_descriptor),
            Guard.descriptor_available(move_descriptor),
            Guard.descriptor_available(select_descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"scc": ValueRef.temporary("condition")},
                result_types={"scc": _I1},
            ),
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("false_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 0},
            ),
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("true_value")},
                result_types={"dst": _I32},
                immediates={"imm32": 1},
            ),
            EmitDescriptorOp(
                descriptor=select_descriptor,
                operands={
                    "true_value": ValueRef.temporary("true_value"),
                    "false_value": ValueRef.temporary("false_value"),
                    "condition": ValueRef.temporary("condition"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _i64_scalar_rules(source_op: Op) -> tuple[DescriptorRule, ...]:
    scalar_rules = tuple(
        _i64_scc_rule(source_op, predicate, _descriptor(descriptor_key))
        for predicate, descriptor_key in _CMP_I64_SCALAR_CASES
    )
    sgpr_bool_rules = tuple(
        _i64_sgpr_bool_rule(source_op, predicate, _descriptor(descriptor_key))
        for predicate, descriptor_key in _CMP_I64_SCALAR_CASES
    )
    return scalar_rules + sgpr_bool_rules


def _mask_rule(
    source_op: Op,
    type_pattern: TypePattern,
    lhs: ValueRef,
    rhs: ValueRef,
    materializer_name: str,
    predicate: str,
    descriptor: Descriptor,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", type_pattern),
            Guard.value_type("rhs", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.value_materializable("lhs", materializer_name),
            Guard.value_materializable("rhs", materializer_name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": lhs,
                    "rhs": rhs,
                },
                results={"mask": ValueRef.result("result")},
            ),
        ),
    )


def _mask_inline_rule(
    source_op: Op,
    type_pattern: TypePattern,
    source_refs: dict[str, ValueRef],
    materializer_name: str,
    literal_source: str,
    nonliteral_source: str,
    predicate: str,
    descriptor: Descriptor,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", type_pattern),
            Guard.value_type("rhs", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.value_materializable(
                nonliteral_source,
                materializer_name,
            ),
            Guard.value_exact_i64(
                literal_source,
                diagnostic=_SOURCE_INLINE_DIAGNOSTIC,
            ),
            Guard.value_i64_range(
                literal_source,
                0,
                64,
                diagnostic=_SOURCE_INLINE_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    nonliteral_source: source_refs[nonliteral_source],
                },
                results={"mask": ValueRef.result("result")},
                immediates={
                    literal_source: ValueProject.exact_i64(literal_source),
                },
            ),
        ),
    )


def _typed_rules(
    source_op: Op,
    type_pattern: TypePattern,
    lhs: ValueRef,
    rhs: ValueRef,
    materializer_name: str,
) -> tuple[DescriptorRule, ...]:
    source_refs = {
        "lhs": lhs,
        "rhs": rhs,
    }
    scalar_rules = tuple(
        _scc_rule(source_op, type_pattern, predicate, _descriptor(descriptor_key))
        for predicate, descriptor_key, _ in _CMP_I32_CASES
    )
    sgpr_bool_rules = tuple(
        _sgpr_bool_rule(
            source_op,
            type_pattern,
            predicate,
            _descriptor(descriptor_key),
        )
        for predicate, descriptor_key, _ in _CMP_I32_CASES
    )
    inline_rules = tuple(
        rule
        for predicate, _, descriptor_key in _CMP_I32_CASES
        for rule in (
            _mask_inline_rule(
                source_op,
                type_pattern,
                source_refs,
                materializer_name,
                literal_source="lhs",
                nonliteral_source="rhs",
                predicate=predicate,
                descriptor=_descriptor(f"{descriptor_key}.src0_inline"),
            ),
            _mask_inline_rule(
                source_op,
                type_pattern,
                source_refs,
                materializer_name,
                literal_source="rhs",
                nonliteral_source="lhs",
                predicate=predicate,
                descriptor=_descriptor(f"{descriptor_key}.src1_inline"),
            ),
        )
    )
    mask_rules = tuple(
        _mask_rule(
            source_op,
            type_pattern,
            lhs,
            rhs,
            materializer_name,
            predicate,
            _descriptor(descriptor_key),
        )
        for predicate, _, descriptor_key in _CMP_I32_CASES
    )
    return scalar_rules + sgpr_bool_rules + inline_rules + mask_rules


def _rules() -> tuple[DescriptorRule, ...]:
    return (
        *_i64_scalar_rules(scalar.scalar_cmpi),
        *_typed_rules(
            scalar.scalar_cmpi,
            _I32,
            _I32_VGPR_LHS,
            _I32_VGPR_RHS,
            I32_VGPR_MATERIALIZER.name,
        ),
        *_typed_rules(
            index.index_cmp,
            _INDEX,
            _ADDRESS_VGPR_LHS,
            _ADDRESS_VGPR_RHS,
            ADDRESS_VGPR_MATERIALIZER.name,
        ),
        *_typed_rules(
            index.index_cmp,
            _OFFSET,
            _ADDRESS_VGPR_LHS,
            _ADDRESS_VGPR_RHS,
            ADDRESS_VGPR_MATERIALIZER.name,
        ),
    )


AMDGPU_COMPARE_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}

AMDGPU_COMPARE_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.compare",
    descriptor_set=_DESCRIPTOR_SET,
    materializers=(I32_VGPR_MATERIALIZER, ADDRESS_VGPR_MATERIALIZER),
    cases=_rules(),
)
