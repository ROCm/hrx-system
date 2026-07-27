# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from loom.dialect.combining import CombiningKind
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar.comparison import CmpIPredicate
from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorRule,
    DirectDescriptorCase,
    PredicateDescriptorCase,
    ReductionDescriptorCase,
    Scalar,
    SelectDescriptorCase,
    ValueRef,
    Vector,
    binary_descriptor_rules,
    compare_descriptor_rules,
    reduction_descriptor_rules,
    select_descriptor_rules,
    ternary_descriptor_rules,
    unary_descriptor_rules,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_ADD_V4I32_DESCRIPTOR,
    TEST_LOW_ALT_DESCRIPTOR_SET,
    TEST_LOW_ALT_NEG_I32_DESCRIPTOR,
    TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
    TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_SELECT_I32_DESCRIPTOR,
)


def test_binary_descriptor_template_generates_test_low_arithmetic_rules() -> None:
    rules = binary_descriptor_rules(
        (
            DirectDescriptorCase(
                scalar_arithmetic.scalar_addi,
                TEST_LOW_ADD_I32_DESCRIPTOR,
                Scalar("i32"),
                semantic_tag="integer.add.i32",
            ),
            DirectDescriptorCase(
                scalar_arithmetic.scalar_muli,
                TEST_LOW_MUL_I32_DESCRIPTOR,
                Scalar("i32"),
                semantic_tag="integer.mul.i32",
            ),
            DirectDescriptorCase(
                scalar_arithmetic.scalar_addf,
                TEST_LOW_ADD_F32_DESCRIPTOR,
                Scalar("f32"),
                semantic_tag="float.add.f32",
            ),
        )
    )

    table = _test_table(rules)

    assert [_descriptor_rule(case).descriptor.key for case in table.cases] == [
        "test.add.i32",
        "test.mul.i32",
        "test.add.f32",
    ]
    add_rule = _descriptor_rule(table.cases[0])
    assert [guard.field for guard in add_rule.guards] == ["lhs", "rhs", "result"]
    assert add_rule.emit[0].operands == {
        "lhs": ValueRef.operand("lhs"),
        "rhs": ValueRef.operand("rhs"),
    }
    assert add_rule.emit[0].results == {"dst": ValueRef.result("result")}


def test_compare_descriptor_template_generates_test_low_predicate_cases() -> None:
    rules = compare_descriptor_rules(
        vector.vector_cmpi,
        (
            PredicateDescriptorCase(
                CmpIPredicate.cases[0],
                TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
                semantic_tag="vector.cmp.eq.i32x4",
            ),
            PredicateDescriptorCase(
                "slt",
                TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
            ),
        ),
        operand_type=Vector("i32", lanes=4),
        result_type=Vector("i1", lanes=4),
    )

    table = _test_table(rules)

    assert [_descriptor_rule(case).descriptor.key for case in table.cases] == [
        "test.cmp.eq.v4i32",
        "test.cmp.eq.v4i32",
    ]
    assert [_descriptor_rule(case).guards[0].enum_keyword for case in table.cases] == [
        "eq",
        "slt",
    ]


def test_select_descriptor_template_generates_test_low_rule() -> None:
    rules = select_descriptor_rules(
        (
            SelectDescriptorCase(
                vector.vector_select,
                TEST_LOW_SELECT_I32_DESCRIPTOR,
                condition_type=Vector("i1", lanes=4),
                value_type=Vector("i32", lanes=4),
            ),
        )
    )

    rule = _descriptor_rule(_test_table(rules).cases[0])

    assert [guard.field for guard in rule.guards] == [
        "condition",
        "true_value",
        "false_value",
        "result",
    ]
    assert rule.emit[0].operands == {
        "true_value": ValueRef.operand("true_value"),
        "false_value": ValueRef.operand("false_value"),
        "condition": ValueRef.operand("condition"),
    }
    assert rule.emit[0].results == {"dst": ValueRef.result("result")}


def test_reduction_descriptor_template_generates_four_lane_test_low_chain() -> None:
    rules = reduction_descriptor_rules(
        vector.vector_reduce,
        (
            ReductionDescriptorCase(
                kind=CombiningKind.cases[0],
                input_type=Vector("i32", lanes=4),
                accumulator_type=Scalar("i32"),
                extract_descriptor=TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR,
                combine_descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                extract_semantic_tag="vector.extract.i32x4",
                combine_semantic_tag="integer.add.i32",
            ),
        ),
        lane_count=4,
    )

    rule = _descriptor_rule(_test_table(rules).cases[0])

    assert rule.guards[0].enum_keyword == "addi"
    assert len(rule.emit) == 8
    lane_values = []
    for i in range(0, 8, 2):
        immediates = rule.emit[i].immediates
        assert isinstance(immediates, Mapping)
        lane_values.append(immediates["lane"])
    assert lane_values == [0, 1, 2, 3]
    assert rule.emit[0].results == {"dst": ValueRef.temporary("lane0")}
    assert rule.emit[1].operands == {
        "lhs": ValueRef.operand("init"),
        "rhs": ValueRef.temporary("lane0"),
    }
    assert rule.emit[1].results == {"dst": ValueRef.temporary("acc0")}
    assert rule.emit[-1].results == {"dst": ValueRef.result("result")}


def test_unary_and_ternary_descriptor_templates_validate_direct_shapes() -> None:
    unary_rules = unary_descriptor_rules(
        (
            DirectDescriptorCase(
                scalar_arithmetic.scalar_negi,
                TEST_LOW_ALT_NEG_I32_DESCRIPTOR,
                Scalar("i32"),
                semantic_tag="test.alt.integer.neg.i32",
            ),
        ),
        descriptor_input="value",
    )
    unary_table = ContractFragment(
        name="test-low.template.unary",
        descriptor_set=TEST_LOW_ALT_DESCRIPTOR_SET,
        cases=unary_rules,
    )

    ternary_rules = ternary_descriptor_rules(
        (
            DirectDescriptorCase(
                vector.vector_dot4i,
                TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
                {
                    "lhs": Vector("i8", lanes=16),
                    "rhs": Vector("i8", lanes=16),
                    "acc": Vector("i32", lanes=4),
                    "result": Vector("i32", lanes=4),
                },
                semantic_tag="integer.dot4.s8s8",
            ),
        ),
        source_a="lhs",
        source_b="rhs",
        source_c="acc",
        descriptor_a="lhs",
        descriptor_b="rhs",
        descriptor_c="acc",
    )
    ternary_table = ContractFragment(
        name="test-low.template.ternary",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=ternary_rules,
    )

    unary_rule = _descriptor_rule(unary_table.cases[0])
    ternary_rule = _descriptor_rule(ternary_table.cases[0])
    assert [guard.field for guard in unary_rule.guards] == ["input", "result"]
    assert unary_rule.emit[0].operands == {"value": ValueRef.operand("input")}
    assert [guard.field for guard in ternary_rule.guards] == [
        "lhs",
        "rhs",
        "acc",
        "result",
    ]
    assert ternary_rule.emit[0].operands == {
        "lhs": ValueRef.operand("lhs"),
        "rhs": ValueRef.operand("rhs"),
        "acc": ValueRef.operand("acc"),
    }


def test_template_rejects_source_op_shape_mismatch() -> None:
    rules = binary_descriptor_rules(
        (
            DirectDescriptorCase(
                vector.vector_splat,
                TEST_LOW_ADD_V4I32_DESCRIPTOR,
                Vector("i32", lanes=4),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            r"vector.splat: guard value_type field 'lhs' is not an operand "
            r"or result"
        ),
    ):
        _test_table(rules)


def test_template_rejects_descriptor_semantic_tag_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"vector.addi binary descriptor case: descriptor 'test.add.v4i32' "
            r"has semantic tag 'vector.add.i32x4', expected 'vector.sub.i32x4'"
        ),
    ):
        binary_descriptor_rules(
            (
                DirectDescriptorCase(
                    vector.vector_addi,
                    TEST_LOW_ADD_V4I32_DESCRIPTOR,
                    Vector("i32", lanes=4),
                    semantic_tag="vector.sub.i32x4",
                ),
            )
        )


def test_template_rejects_unknown_descriptor_field() -> None:
    rules = binary_descriptor_rules(
        (
            DirectDescriptorCase(
                vector.vector_addi,
                TEST_LOW_ADD_V4I32_DESCRIPTOR,
                Vector("i32", lanes=4),
            ),
        ),
        descriptor_lhs="missing",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor 'test.add.v4i32': descriptor operand binding "
            r"field 'missing' is not a descriptor operand"
        ),
    ):
        _test_table(rules)


def test_template_rejects_unknown_enum_mapping() -> None:
    rules = compare_descriptor_rules(
        vector.vector_cmpi,
        (
            PredicateDescriptorCase(
                "missing",
                TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
            ),
        ),
        operand_type=Vector("i32", lanes=4),
        result_type=Vector("i1", lanes=4),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"vector.cmpi: guard enum_attr_equals field 'predicate' "
            r"has no enum case 'missing'"
        ),
    ):
        _test_table(rules)


def test_template_rejects_incomplete_type_case() -> None:
    with pytest.raises(
        ValueError,
        match=r"vector.splat unary descriptor case: source field 'result' "
        r"has no type pattern",
    ):
        unary_descriptor_rules(
            (
                DirectDescriptorCase(
                    vector.vector_splat,
                    TEST_LOW_ALT_NEG_I32_DESCRIPTOR,
                    {"scalar": Scalar("i32")},
                ),
            ),
            source_input="scalar",
            descriptor_input="value",
        )


def test_reduction_template_rejects_empty_lane_chain() -> None:
    with pytest.raises(ValueError, match=r"reduction lane count must be positive"):
        reduction_descriptor_rules(
            vector.vector_reduce,
            (
                ReductionDescriptorCase(
                    kind="addi",
                    input_type=Vector("i32", lanes=4),
                    accumulator_type=Scalar("i32"),
                    extract_descriptor=TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR,
                    combine_descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                ),
            ),
            lane_count=0,
        )


def _test_table(cases: Sequence[ContractCase]) -> ContractFragment:
    return ContractFragment(
        name="test-low.template.test",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=cases,
    )


def _descriptor_rule(case: ContractCase) -> DescriptorRule:
    assert isinstance(case, DescriptorRule)
    return case
