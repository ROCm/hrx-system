# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.format.text.parser import ParseError, Parser
from loom.format.text.printer import Printer


def _roundtrip(text: str) -> str:
    parser = Parser()
    parser.register_ops(ALL_FUNC_OPS)
    parser.register_ops(ALL_INDEX_OPS)
    parser.register_ops(ALL_SCALAR_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(text)
    printer = Printer()
    printer.register_ops(ALL_FUNC_OPS)
    printer.register_ops(ALL_INDEX_OPS)
    printer.register_ops(ALL_SCALAR_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module).strip()


def test_simple() -> None:
    text = "func.def @f() {\n  func.return\n}"
    assert _roundtrip(text) == text


def test_explicit_dim_arg() -> None:
    # Explicit definition of %m.
    text = "func.def @f(%m: index, %arg: tile<[%m]xf32>) {\n  func.return %arg : tile<[%m]xf32>\n}"
    assert _roundtrip(text) == text


def test_forward_reference_dim() -> None:
    # Define %m AFTER its use in %arg's type.
    text = "func.def @f(%arg: tile<[%m]xf32>, %m: index) {\n  func.return %arg : tile<[%m]xf32>\n}"
    assert _roundtrip(text) == text


def test_named_result() -> None:
    # Named result %n.
    text = """func.def @f() -> (%n: index, tile<[%n]xf32>) {
  %c0 = index.constant 0 : index
  %t = scalar.constant 0 : tile<0xf32>
  func.return %c0, %t : index, tile<0xf32>
}"""
    # Roundtrip check.
    rt = _roundtrip(text)
    assert "@f" in rt
    assert "%n: index" in rt
    assert "tile<[%n]xf32>" in rt
    assert "tile<0xf32>" in rt


def test_tied_result_to_function_arg() -> None:
    text = "func.def @f(%x: f32) -> (%x as f32) {\n  func.return %x : f32\n}"
    assert _roundtrip(text) == text


def test_decl_tied_result_to_function_arg() -> None:
    text = "func.decl @f(%x: f32) -> (%x as f32)"
    assert _roundtrip(text) == text


def test_ssa_encoding() -> None:
    # SSA encoding value in type.
    text = "func.def @f(%enc: encoding, %arg: tile<4xf32, %enc>) {\n  func.return %arg : tile<4xf32, %enc>\n}"
    assert _roundtrip(text) == text


def test_unresolved_placeholder() -> None:
    # %m is never defined.
    text = "func.def @f(%arg: tile<[%m]xf32>) {\n  func.return %arg : tile<[%m]xf32>\n}"
    with pytest.raises(
        ParseError, match="unresolved forward reference to '%m'"
    ) as exc_info:
        _roundtrip(text)
    assert exc_info.value.location.line == 1
    assert exc_info.value.location.column == 25


def test_numeric_hash_attr_fail() -> None:
    text = "func.def @f() -> (tile<[#0]xf32>) {\n  func.return\n}"
    with pytest.raises(ParseError, match="expected identifier after '#'"):
        _roundtrip(text)


def test_isolation() -> None:
    # %n (named result) should NOT be visible in function body.
    text = """func.def @f() -> (%n: index, tile<[%n]xf32>) {
  %x = index.add %n, %n : index
  func.return %x, %x : index, tile<[%n]xf32>
}"""
    with pytest.raises(ParseError, match="undefined SSA value '%n'"):
        _roundtrip(text)
