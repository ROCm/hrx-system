# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import ANY, Dialect, Op, Operand, Result
from loom.target.contracts import (
    CompiledDescriptorRule,
    ContractFragment,
    ContractSystem,
    DescriptorMatrixRule,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    RecipeRule,
    ValueAliasRule,
    ValueElideRule,
    ValueRef,
    Vector,
    compile_contract_fragment,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_ADD_V4I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


def test_compile_contract_fragment_packs_populated_op_spans() -> None:
    table = ContractFragment(
        name="test-low.vector",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_addi,
                descriptor=TEST_LOW_ADD_V4I32_DESCRIPTOR,
                guards=[
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_V4I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    )
                ],
            ),
            ValueAliasRule(
                source_op=vector.vector_fragment,
                source=ValueRef.operand("data"),
                result=ValueRef.result("result"),
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={"vector": ALL_VECTOR_OPS},
        descriptor_rule_rows={0: CompiledDescriptorRule(0)},
        lower_rule_indices={0: 0, 1: 1},
    )

    spans_by_name = {span.op_name: span for span in compiled.op_spans}
    assert spans_by_name.keys() == {"vector.addi", "vector.fragment"}

    addi_span = spans_by_name["vector.addi"]
    assert addi_span.op_kind == (
        vector.vector_ops.dialect_id << 8
    ) | ALL_VECTOR_OPS.index(vector.vector_addi)
    assert addi_span.case_start == 0
    assert addi_span.case_count == 1
    assert compiled.cases[0].system == ContractSystem.DESCRIPTOR_RULE
    assert compiled.cases[0].row_index == 0
    assert compiled.descriptor_rules[0].rule_index == 0

    fragment_span = spans_by_name["vector.fragment"]
    assert fragment_span.case_start == 1
    assert fragment_span.case_count == 1
    assert compiled.cases[1].system == ContractSystem.VALUE_ALIAS
    assert compiled.cases[1].row_index == 1


def test_compile_contract_fragment_uses_supplied_descriptor_rule_rows() -> None:
    table = ContractFragment(
        name="test-low.authored-order",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            ),
            DescriptorRule(
                source_op=vector.vector_addi,
                descriptor=TEST_LOW_ADD_V4I32_DESCRIPTOR,
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={"vector": ALL_VECTOR_OPS},
        descriptor_rule_rows={
            0: CompiledDescriptorRule(rule_index=9),
            1: CompiledDescriptorRule(rule_index=2),
        },
        lower_rule_indices={0: 9, 1: 2},
    )

    spans_by_name = {span.op_name: span for span in compiled.op_spans}
    addi_span = spans_by_name["vector.addi"]
    reduce_span = spans_by_name["vector.reduce"]
    assert addi_span.case_start == 0
    assert compiled.cases[addi_span.case_start].row_index == 1
    assert compiled.descriptor_rules[1].rule_index == 2
    assert reduce_span.case_start == 1
    assert compiled.cases[reduce_span.case_start].row_index == 0
    assert compiled.descriptor_rules[0].rule_index == 9


def test_compile_contract_fragment_records_value_elide_cases() -> None:
    table = ContractFragment(
        name="test-low.elide",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueElideRule(
                source_op=vector.vector_extract,
                values=(ValueRef.result("result"),),
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={"vector": ALL_VECTOR_OPS},
        descriptor_rule_rows={},
        lower_rule_indices={0: 7},
    )

    extract_span = compiled.op_spans[0]
    assert extract_span.op_name == "vector.extract"
    assert extract_span.case_start == 0
    assert extract_span.case_count == 1
    assert compiled.cases[0].system == ContractSystem.VALUE_ELIDE
    assert compiled.cases[0].row_index == 7


def test_compile_contract_fragment_records_recipe_cases() -> None:
    table = ContractFragment(
        name="test-low.recipe",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            RecipeRule(
                source_op=vector.vector_addi,
                guards=(
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={"vector": ALL_VECTOR_OPS},
        descriptor_rule_rows={},
        lower_rule_indices={0: 11},
    )

    addi_span = compiled.op_spans[0]
    assert addi_span.op_name == "vector.addi"
    assert addi_span.case_start == 0
    assert addi_span.case_count == 1
    assert compiled.cases[0].system == ContractSystem.RECIPE_RULE
    assert compiled.cases[0].row_index == 11


def test_compile_contract_fragment_records_descriptor_matrix_cases() -> None:
    table = ContractFragment(
        name="test-low.matrix",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorMatrixRule(
                source_op=vector.vector_mma,
                source="vector_mma",
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={"vector": ALL_VECTOR_OPS},
        descriptor_rule_rows={},
        lower_rule_indices={},
    )

    mma_span = compiled.op_spans[0]
    assert mma_span.op_name == "vector.mma"
    assert mma_span.case_start == 0
    assert mma_span.case_count == 1
    assert compiled.cases[0].system == ContractSystem.DESCRIPTOR_MATRIX
    assert compiled.cases[0].row_index == 0
    assert tuple(row.source for row in compiled.descriptor_matrices) == ("vector_mma",)


def test_compile_contract_fragment_omits_unpopulated_dialect_gaps() -> None:
    low_dialect = Dialect("gap_low", dialect_id=3)
    high_dialect = Dialect("gap_high", dialect_id=5)
    low_op = Op(
        "gap_low.alias",
        group=low_dialect,
        operands=(Operand("input", ANY),),
        results=(Result("result", ANY),),
    )
    high_op = Op(
        "gap_high.alias",
        group=high_dialect,
        operands=(Operand("input", ANY),),
        results=(Result("result", ANY),),
    )
    table = ContractFragment(
        name="dense.gap",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueAliasRule(
                source_op=low_op,
                source=ValueRef.operand("input"),
                result=ValueRef.result("result"),
            ),
            ValueAliasRule(
                source_op=high_op,
                source=ValueRef.operand("input"),
                result=ValueRef.result("result"),
            ),
        ],
    )

    compiled = compile_contract_fragment(
        table,
        dialect_ops={
            "gap_low": (low_op,),
            "gap_high": (high_op,),
        },
        descriptor_rule_rows={},
        lower_rule_indices={0: 0, 1: 1},
    )

    assert [span.op_name for span in compiled.op_spans] == [
        "gap_low.alias",
        "gap_high.alias",
    ]
    assert [span.op_kind for span in compiled.op_spans] == [3 << 8, 5 << 8]
    assert compiled.op_spans[1].case_start == 1
    assert compiled.op_spans[1].case_count == 1


def test_compile_contract_fragment_rejects_missing_dialect_ops() -> None:
    table = ContractFragment(
        name="missing.scalar",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueAliasRule(
                source_op=scalar_arithmetic.scalar_addi,
                source=ValueRef.operand("lhs"),
                result=ValueRef.result("result"),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"op 'scalar.addi' is not present in dialect_ops",
    ):
        compile_contract_fragment(
            table,
            dialect_ops={"vector": ALL_VECTOR_OPS},
            descriptor_rule_rows={},
            lower_rule_indices={},
        )


def test_compile_contract_fragment_rejects_mismatched_dialect_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"dialect_ops key 'not_scalar' does not match dialect 'scalar'",
    ):
        compile_contract_fragment(
            ContractFragment(
                name="empty",
                descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            ),
            dialect_ops={"not_scalar": ALL_SCALAR_OPS},
            descriptor_rule_rows={},
            lower_rule_indices={},
        )
