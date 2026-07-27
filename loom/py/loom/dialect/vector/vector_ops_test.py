# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS, FastMathFlags
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dsl import FLOAT, I1, INTEGER, Op

SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS = {
    "assume": "scalar.assume refines scalar predicate facts, not vector lanes.",
}


def _op_suffix(op: Op) -> str:
    return op.name.split(".", 1)[1]


def test_vector_seed_mirrors_scalar_spelling_for_lanewise_ops() -> None:
    scalar_names = {_op_suffix(op) for op in ALL_SCALAR_OPS}
    vector_names = {_op_suffix(op) for op in ALL_VECTOR_OPS}

    missing = sorted(scalar_names - vector_names - set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS))
    unknown_exclusions = sorted(set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS) - scalar_names)
    stale_exclusions = sorted(set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS) & vector_names)

    assert not missing, f"scalar ops missing vector mirrors: {missing}"
    assert not unknown_exclusions, f"unknown scalar/vector exclusions: {unknown_exclusions}"
    assert not stale_exclusions, f"stale scalar/vector exclusions: {stale_exclusions}"
    assert "sqrt" not in vector_names
    assert "fpext" not in vector_names


def test_vector_fields_do_not_use_scalar_family_constraints() -> None:
    forbidden = {FLOAT, INTEGER, I1}
    for op in ALL_VECTOR_OPS:
        for field in (*op.operands, *op.results):
            assert field.type_constraint not in forbidden, (op.name, field.name)


def test_vector_float_flags_use_scalar_fastmath_flags() -> None:
    scalar_flags = {case.keyword: case.value for case in FastMathFlags.cases}
    vector_float_ops = [op for op in ALL_VECTOR_OPS if any(attr.name == "fastmath" for attr in op.attrs)]

    assert vector_float_ops
    for op in vector_float_ops:
        fastmath_attr = next(attr for attr in op.attrs if attr.name == "fastmath")
        assert fastmath_attr.enum_def is FastMathFlags, op.name

    reduce_fastmath_ops = {op.name for op in vector_float_ops if op.name.startswith("vector.reduce")}
    assert reduce_fastmath_ops == {"vector.reduce", "vector.reduce.axes"}
    assert scalar_flags["contract"] == 32
    assert scalar_flags["fast"] == 127
