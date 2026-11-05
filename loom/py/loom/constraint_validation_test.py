# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Python declarative constraint predicates."""

from dataclasses import dataclass
from typing import Any

from loom import ir
from loom.builders import default_ops
from loom.dsl import (
    REGISTER,
    VIEW,
    BlockArgCount,
    BlockArgsMatchElementTypes,
    BlockArgsMatchTypes,
    BlockArgsSatisfy,
    ClosedValueTuple,
    ConditionForwardedCountMatchesBlockArgs,
    ConditionForwardedTypesMatchBlockArgs,
    IterArgsMatchResults,
    OperandDictionary,
    SameEncoding,
    VariadicValuesMatch,
    YieldCountMatches,
    YieldTypesMatch,
)
from loom.ir import (
    ENCODING_LAYOUT_TYPE,
    F32,
    I32,
    INDEX,
    DynamicDim,
    DynamicEncoding,
    RegisterType,
    ShapedType,
    StaticDim,
    TypeKind,
)


@dataclass(frozen=True)
class _FakeValue:
    type: Any


@dataclass(frozen=True)
class _FakeIdentifiedValue:
    type: Any
    value_id: int


@dataclass(frozen=True)
class _FakeRegion:
    entry_args: list[_FakeValue] | None
    terminator_operands: list[_FakeValue] | None


def test_builtin_constraints_have_python_validators() -> None:
    missing = [
        f"{op.name}: {constraint.name}"
        for op in default_ops()
        for constraint in op.constraints
        if constraint.validate is None
    ]
    assert not missing, "missing Python validators: " + ", ".join(missing)


def test_same_encoding() -> None:
    @dataclass(frozen=True)
    class FakeType:
        encoding: object

    lhs = _FakeValue(FakeType("layout"))
    same = _FakeValue(FakeType("layout"))
    other = _FakeValue(FakeType("other"))
    constraint = SameEncoding("a", "b")

    assert constraint.check({"a": lhs, "b": same})[0]
    assert not constraint.check({"a": lhs, "b": other})[0]


def test_closed_value_tuple_matches_exact_carrier_dependencies() -> None:
    extent = _FakeIdentifiedValue(INDEX, 1)
    layout = _FakeIdentifiedValue(ENCODING_LAYOUT_TYPE, 2)
    carrier_type = ShapedType(
        TypeKind.VIEW,
        F32,
        (DynamicDim(extent.value_id),),
        DynamicEncoding(layout.value_id),
    )
    carrier = _FakeIdentifiedValue(carrier_type, 3)
    constraint = ClosedValueTuple("values", VIEW)

    assert constraint.check({"values": [extent, layout, carrier]})[0]
    assert not constraint.check({"values": [extent, carrier]})[0]
    assert not constraint.check({"values": [layout, extent, carrier]})[0]
    assert not constraint.check({"values": [extent, extent, layout, carrier]})[0]


def test_operand_dictionary_permutations() -> None:
    constraint = OperandDictionary("params", "names")
    for count in (0, 1, 63, 64, 65, 129):
        names = ir.CanonicalAttrDict(
            (f"parameter_{index:03}", count - index - 1) for index in range(count)
        )
        assert constraint.check(
            {"params": [ir.Value("input", I32)] * count, "names": names}
        ) == (True, "")
    assert constraint.check({"params": [], "names": None}) == (True, "")


def test_operand_dictionary_rejects_invalid_ordinals() -> None:
    constraint = OperandDictionary("params", "names")
    value = ir.Value("input", I32)
    for ordinal in (-1, 2, 2**63, True, 0.0, "0", None):
        names = ir.CanonicalAttrDict((("alpha", ordinal), ("beta", 1)))
        assert not constraint.check({"params": [value, value], "names": names})[0], (
            ordinal
        )
    duplicate = ir.CanonicalAttrDict((("alpha", 1), ("beta", 1)))
    assert not constraint.check({"params": [value, value], "names": duplicate})[0]
    for names in (None, 0, [], ir.CanonicalAttrDict()):
        assert not constraint.check({"params": [value], "names": names})[0]


def test_region_entry_args() -> None:
    i32 = _FakeValue(ir.I32)
    f32 = _FakeValue(ir.F32)
    one_i32_arg = _FakeRegion([i32], None)

    count = BlockArgCount("body", "inputs")
    assert count.check({"body": one_i32_arg, "inputs": [i32]})[0]
    assert not count.check({"body": one_i32_arg, "inputs": []})[0]

    satisfy = BlockArgsSatisfy("body", REGISTER)
    register_arg = _FakeRegion([_FakeValue(RegisterType(1, 0))], None)
    assert satisfy.check({"body": register_arg})[0]
    assert not satisfy.check({"body": one_i32_arg})[0]

    shaped_i32 = _FakeValue(ShapedType(TypeKind.VECTOR, I32, (StaticDim(1),)))
    element_types = BlockArgsMatchElementTypes("body", "inputs")
    assert element_types.check({"body": one_i32_arg, "inputs": [shaped_i32]})[0]
    assert not element_types.check(
        {"body": _FakeRegion([f32], None), "inputs": [shaped_i32]}
    )[0]

    types = BlockArgsMatchTypes("body", "inputs")
    assert types.check({"body": one_i32_arg, "inputs": [i32]})[0]
    assert not types.check({"body": one_i32_arg, "inputs": [f32]})[0]


def test_condition_forwarding() -> None:
    i1 = _FakeValue(ir.I1)
    i32 = _FakeValue(ir.I32)
    f32 = _FakeValue(ir.F32)
    one_i32_arg = _FakeRegion([i32], None)

    count = ConditionForwardedCountMatchesBlockArgs("before", "after", "inputs")
    assert count.check(
        {
            "before": _FakeRegion([i32], [i1, i32]),
            "after": one_i32_arg,
            "inputs": [i32],
        }
    )[0]
    assert not count.check(
        {
            "before": _FakeRegion([i32], [i1]),
            "after": one_i32_arg,
            "inputs": [i32],
        }
    )[0]

    types = ConditionForwardedTypesMatchBlockArgs("before", "after", "inputs")
    assert types.check(
        {
            "before": _FakeRegion([i32], [i1, i32]),
            "after": one_i32_arg,
            "inputs": [i32],
        }
    )[0]
    assert not types.check(
        {
            "before": _FakeRegion([i32], [i1, f32]),
            "after": one_i32_arg,
            "inputs": [i32],
        }
    )[0]
    assert types.check(
        {
            "before": _FakeRegion([i32], [i1, i32]),
            "after": _FakeRegion([f32], None),
            "inputs": [i32],
        }
    )[0], "malformed target args must not cascade into forwarding errors"


def test_yield_and_variadic_values() -> None:
    i32 = _FakeValue(ir.I32)
    f32 = _FakeValue(ir.F32)

    count = YieldCountMatches("body", "results")
    assert count.check({"body": _FakeRegion([], [i32]), "results": [i32]})[0]
    assert not count.check({"body": _FakeRegion([], []), "results": [i32]})[0]

    types = YieldTypesMatch("body", "results")
    assert types.check({"body": _FakeRegion([], [i32]), "results": [i32]})[0]
    assert not types.check({"body": _FakeRegion([], [f32]), "results": [i32]})[0]

    header_count = YieldCountMatches("body", "header")
    assert header_count.check(
        {"body": _FakeRegion([], [i32]), "header": _FakeRegion([i32], None)}
    )[0]
    assert not header_count.check(
        {"body": _FakeRegion([], []), "header": _FakeRegion([i32], None)}
    )[0]

    header_types = YieldTypesMatch("body", "header")
    assert header_types.check(
        {"body": _FakeRegion([], [i32]), "header": _FakeRegion([i32], None)}
    )[0]
    assert not header_types.check(
        {"body": _FakeRegion([], [f32]), "header": _FakeRegion([i32], None)}
    )[0]

    variadic = VariadicValuesMatch("lhs", "rhs")
    assert variadic.check({"lhs": [i32], "rhs": [i32]})[0]
    assert not variadic.check({"lhs": [i32], "rhs": [f32]})[0]

    iter_args = IterArgsMatchResults("iter_args", "results")
    assert iter_args.check({"iter_args": [i32], "results": [i32]})[0]
    assert not iter_args.check({"iter_args": [i32], "results": []})[0]
