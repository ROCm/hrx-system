# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numeric expression semantics in exported C++."""

import pytest

from loom.format.cxx import load_module, render_module


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("vector", [False, True])
def test_integer_division_preserves_each_uses_signedness(
    width: int, vector: bool
) -> None:
    value_type = f"vector<2xi{width}>" if vector else f"i{width}"
    dialect = "vector" if vector else "scalar"
    source = f"""\
func.def @divide(%lhs: {value_type}, %rhs: {value_type}) -> ({value_type}, {value_type}, {value_type}, {value_type}) {{
  %signed_quotient = {dialect}.divsi %lhs, %rhs : {value_type}
  %signed_remainder = {dialect}.remsi %lhs, %rhs : {value_type}
  %unsigned_quotient = {dialect}.divui %lhs, %rhs : {value_type}
  %unsigned_remainder = {dialect}.remui %lhs, %rhs : {value_type}
  func.return %signed_quotient, %signed_remainder, %unsigned_quotient, %unsigned_remainder : {value_type}, {value_type}, {value_type}, {value_type}
}}
"""
    result = render_module(load_module(source.encode()))
    unsigned_type = (
        f"UInt{width}x2"
        if vector
        else {
            8: "unsigned char",
            16: "unsigned short",
            32: "unsigned",
            64: "unsigned long long",
        }[width]
    )
    convert = "loom::convert" if vector else "static_cast"
    for name, operator in (("quotient", "/"), ("remainder", "%")):
        assert f"signed_{name} = lhs {operator} rhs;" in result
        assert (
            f"unsigned_{name} = {convert}<{unsigned_type}>(lhs) {operator} "
            f"{convert}<{unsigned_type}>(rhs);"
        ) in result


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        (4, "4ll"),
        (-12, "-12ll"),
        (-(1 << 63), "(-9223372036854775807ll - 1ll)"),
        ((1 << 63) - 1, "9223372036854775807ll"),
    ],
)
def test_signed_i64_literals_keep_their_type_when_inlined(
    value: int, literal: str
) -> None:
    source = f"""\
func.def @divide(%lhs: i64) -> (i64) {{
  %1 = scalar.constant {value} : i64
  %result = scalar.divsi %lhs, %1 : i64
  func.return %result : i64
}}
"""
    result = render_module(load_module(source.encode()))
    assert f"long long result = lhs / {literal};" in result
