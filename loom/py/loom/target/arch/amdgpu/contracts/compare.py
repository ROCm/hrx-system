# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU scalar and vector comparison source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import comparison as scalar
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_REGISTER_MATERIALIZER,
    F32_VGPR_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
    REGISTERS_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import (
    AMDGPU_SOURCE_INLINE_F32_VALUES,
    build_amdgpu_contract_descriptor_set,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorOperandMaterialization,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
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

_CMP_FLOAT_SCALAR_PREDICATES = (
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
_FLOAT_COMPARE_DESCRIPTOR_KEYS = tuple(
    f"amdgpu.s_cmp_{predicate}_f{bit_width}"
    for bit_width in (16, 32)
    for predicate in _CMP_FLOAT_SCALAR_PREDICATES
)
_FLOAT_COMPARE_MASK_DESCRIPTOR_KEYS = tuple(
    descriptor_key
    for predicate in _CMP_FLOAT_SCALAR_PREDICATES
    for descriptor_key in (
        f"amdgpu.v_cmp_{predicate}_f16",
        f"amdgpu.v_cmp_{predicate}_f32",
        f"amdgpu.v_cmp_{predicate}_f64",
        f"amdgpu.v_cmp_{predicate}_f32.src0_inline",
        f"amdgpu.v_cmp_{predicate}_f32.src1_inline",
    )
)
_FLOAT_CLASSIFICATION_CASES = (
    (scalar.scalar_isnanf, 0x003),
    (scalar.scalar_isinff, 0x204),
    (scalar.scalar_isfinitef, 0x1F8),
)
_FLOAT_CLASSIFICATION_DESCRIPTOR_KEYS = (
    *(
        descriptor_key
        for bit_width in (16, 32, 64)
        for descriptor_key in (
            f"amdgpu.v_cmp_class_f{bit_width}",
            f"amdgpu.v_cmp_class_f{bit_width}.classes_inline",
            f"amdgpu.v_cmp_class_f{bit_width}.classes_lit",
        )
    ),
    "amdgpu.v_cmp_class_f16.input_high",
    "amdgpu.v_cmp_class_f16.input_high.classes_inline",
    "amdgpu.v_cmp_class_f16.input_high.classes_lit",
)
_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_cselect_b32",
    *_COMPARE_DESCRIPTOR_KEYS,
    *_FLOAT_COMPARE_DESCRIPTOR_KEYS,
    *_FLOAT_COMPARE_MASK_DESCRIPTOR_KEYS,
    *_FLOAT_CLASSIFICATION_DESCRIPTOR_KEYS,
    *(descriptor_key for _, descriptor_key in _CMP_I64_SCALAR_CASES),
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.compare",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I64 = Scalar("i64")
_I32 = Scalar("i32")
_F16 = Scalar("f16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_I1 = Scalar("i1")
_REGISTER_LHS = ValueRef.operand("lhs")
_REGISTER_RHS = ValueRef.operand("rhs")
_ADDRESS_LHS = ValueRef.operand("lhs", materializer=ADDRESS_REGISTER_MATERIALIZER.name)
_ADDRESS_RHS = ValueRef.operand("rhs", materializer=ADDRESS_REGISTER_MATERIALIZER.name)
_SOURCE_INLINE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="source-inline-u32",
    constraint_key="amdgpu.source_inline_u32",
)
_SOURCE_INLINE_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="literal",
    subject_name="f32",
    constraint_key="amdgpu.literal.exact_f32",
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


def _float_scalar_rules(
    source_op: Op, type_pattern: TypePattern, bit_width: int
) -> tuple[DescriptorRule, ...]:
    scalar_rules = tuple(
        _scc_rule(
            source_op,
            type_pattern,
            predicate,
            _descriptor(f"amdgpu.s_cmp_{predicate}_f{bit_width}"),
        )
        for predicate in _CMP_FLOAT_SCALAR_PREDICATES
    )
    sgpr_bool_rules = tuple(
        _sgpr_bool_rule(
            source_op,
            type_pattern,
            predicate,
            _descriptor(f"amdgpu.s_cmp_{predicate}_f{bit_width}"),
        )
        for predicate in _CMP_FLOAT_SCALAR_PREDICATES
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
            Guard.low_value_register_unit_count("result", 2),
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
                operand_materialization=DescriptorOperandMaterialization.TARGET,
            ),
        ),
    )


def _float_mask_inline_rule(
    *,
    literal_source: str,
    nonliteral_source: str,
    literal_value: float,
    predicate: str,
    descriptor: Descriptor,
) -> DescriptorRule:
    source_refs = {
        "lhs": _REGISTER_LHS,
        "rhs": _REGISTER_RHS,
    }
    return DescriptorRule(
        source_op=scalar.scalar_cmpf,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _F32),
            Guard.value_type("rhs", _F32),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 2),
            Guard.value_materializable(
                nonliteral_source,
                F32_VGPR_MATERIALIZER.name,
            ),
            Guard.value_float_equals(
                literal_source,
                literal_value,
                diagnostic=_SOURCE_INLINE_F32_DIAGNOSTIC,
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
                    literal_source: ValueProject.float_bits(literal_source),
                },
            ),
        ),
    )


def _float_mask_rules() -> tuple[DescriptorRule, ...]:
    inline_rules = tuple(
        rule
        for predicate in _CMP_FLOAT_SCALAR_PREDICATES
        for literal_value in AMDGPU_SOURCE_INLINE_F32_VALUES
        for rule in (
            _float_mask_inline_rule(
                literal_source="lhs",
                nonliteral_source="rhs",
                literal_value=literal_value,
                predicate=predicate,
                descriptor=_descriptor(f"amdgpu.v_cmp_{predicate}_f32.src0_inline"),
            ),
            _float_mask_inline_rule(
                literal_source="rhs",
                nonliteral_source="lhs",
                literal_value=literal_value,
                predicate=predicate,
                descriptor=_descriptor(f"amdgpu.v_cmp_{predicate}_f32.src1_inline"),
            ),
        )
    )
    register_rules = tuple(
        _mask_rule(
            scalar.scalar_cmpf,
            _F32,
            _REGISTER_LHS,
            _REGISTER_RHS,
            F32_VGPR_MATERIALIZER.name,
            predicate,
            _descriptor(f"amdgpu.v_cmp_{predicate}_f32"),
        )
        for predicate in _CMP_FLOAT_SCALAR_PREDICATES
    )
    other_register_rules = tuple(
        _mask_rule(
            scalar.scalar_cmpf,
            type_pattern,
            _REGISTER_LHS,
            _REGISTER_RHS,
            REGISTERS_VGPR_MATERIALIZER.name,
            predicate,
            _descriptor(f"amdgpu.v_cmp_{predicate}_f{bit_width}"),
        )
        for type_pattern, bit_width in ((_F16, 16), (_F64, 64))
        for predicate in _CMP_FLOAT_SCALAR_PREDICATES
    )
    return inline_rules + register_rules + other_register_rules


def _float_classification_immediate_rule(
    source_op: Op,
    type_pattern: TypePattern,
    bit_width: int,
    class_mask: int,
    descriptor_suffix: str,
    immediate_name: str,
) -> DescriptorRule:
    descriptor = _descriptor(f"amdgpu.v_cmp_class_f{bit_width}.{descriptor_suffix}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 2),
            Guard.value_materializable("input", REGISTERS_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": ValueRef.operand(
                        "input", materializer=REGISTERS_VGPR_MATERIALIZER.name
                    ),
                },
                results={"mask": ValueRef.result("result")},
                immediates={immediate_name: class_mask},
            ),
        ),
    )


def _float_classification_register_rule(
    source_op: Op,
    type_pattern: TypePattern,
    bit_width: int,
    class_mask: int,
) -> DescriptorRule:
    move_descriptor = _descriptor("amdgpu.s_mov_b32")
    descriptor = _descriptor(f"amdgpu.v_cmp_class_f{bit_width}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 2),
            Guard.value_materializable("input", REGISTERS_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(move_descriptor),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("classes")},
                result_types={"dst": _I32},
                immediates={"imm32": class_mask},
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": ValueRef.operand(
                        "input", materializer=REGISTERS_VGPR_MATERIALIZER.name
                    ),
                    "classes": ValueRef.temporary("classes"),
                },
                results={"mask": ValueRef.result("result")},
            ),
        ),
    )


def _float_classification_scalar_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for source_op, class_mask in _FLOAT_CLASSIFICATION_CASES:
        for type_pattern, bit_width in ((_F16, 16), (_F32, 32), (_F64, 64)):
            if class_mask <= 64:
                rules.append(
                    _float_classification_immediate_rule(
                        source_op,
                        type_pattern,
                        bit_width,
                        class_mask,
                        "classes_inline",
                        "classes",
                    )
                )
            else:
                rules.append(
                    _float_classification_immediate_rule(
                        source_op,
                        type_pattern,
                        bit_width,
                        class_mask,
                        "classes_lit",
                        "imm32",
                    )
                )
                rules.append(
                    _float_classification_register_rule(
                        source_op, type_pattern, bit_width, class_mask
                    )
                )
    return tuple(rules)


def _float_classification_vector_rules() -> tuple[RecipeRule, ...]:
    type_rows = (
        ("f16", 16, "LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES"),
        ("f32", 32, "LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES"),
        ("f64", 64, "LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u"),
    )
    vector_ops = (
        vector.vector_isnanf,
        vector.vector_isinff,
        vector.vector_isfinitef,
    )
    return tuple(
        RecipeRule(
            source_op=source_op,
            guards=(
                Guard.value_type(
                    "input",
                    Vector(
                        element,
                        minimum_lanes=1,
                        maximum_lanes=maximum_lanes,
                    ),
                ),
                Guard.descriptor_available(
                    _descriptor(f"amdgpu.v_cmp_class_f{bit_width}")
                ),
            ),
        )
        for source_op in vector_ops
        for element, bit_width, maximum_lanes in type_rows
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


def _vector_rules() -> tuple[RecipeRule, ...]:
    # The C emitter keeps vector operands intact until Low emission. Publishing
    # its contract prevents reference legalization from unpacking native inputs.
    families = (
        (
            vector.vector_cmpi,
            "i32",
            tuple((predicate, key) for predicate, _, key in _CMP_I32_CASES),
        ),
        (
            vector.vector_cmpf,
            "f32",
            tuple(
                (predicate, f"amdgpu.v_cmp_{predicate}_f32")
                for predicate in _CMP_FLOAT_SCALAR_PREDICATES
            ),
        ),
    )
    return tuple(
        RecipeRule(
            source_op=source_op,
            guards=(
                Guard.value_type(
                    "lhs",
                    Vector(
                        element,
                        minimum_lanes=1,
                        maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
                    ),
                ),
                Guard.enum_attr_equals("predicate", predicate),
                Guard.descriptor_available(_descriptor(descriptor_key)),
            ),
        )
        for source_op, element, predicates in families
        for predicate, descriptor_key in predicates
    )


def _rules() -> tuple[DescriptorRule | RecipeRule, ...]:
    return (
        *_vector_rules(),
        *_float_classification_vector_rules(),
        *_float_classification_scalar_rules(),
        *_float_scalar_rules(scalar.scalar_cmpf, _F16, 16),
        *_float_scalar_rules(scalar.scalar_cmpf, _F32, 32),
        *_float_mask_rules(),
        *_i64_scalar_rules(scalar.scalar_cmpi),
        *_typed_rules(
            scalar.scalar_cmpi,
            _I32,
            _REGISTER_LHS,
            _REGISTER_RHS,
            I32_VGPR_MATERIALIZER.name,
        ),
        *_typed_rules(
            index.index_cmp,
            _INDEX,
            _ADDRESS_LHS,
            _ADDRESS_RHS,
            ADDRESS_REGISTER_MATERIALIZER.name,
        ),
        *_typed_rules(
            index.index_cmp,
            _OFFSET,
            _ADDRESS_LHS,
            _ADDRESS_RHS,
            ADDRESS_REGISTER_MATERIALIZER.name,
        ),
    )


AMDGPU_COMPARE_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_COMPARE_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.compare",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/compare.h",
    materializers=(
        I32_VGPR_MATERIALIZER,
        REGISTERS_VGPR_MATERIALIZER,
        F32_VGPR_MATERIALIZER,
        ADDRESS_REGISTER_MATERIALIZER,
    ),
    cases=_rules(),
)
