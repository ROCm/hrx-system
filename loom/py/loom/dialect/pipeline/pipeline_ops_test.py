# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for portable pipeline text and bytecode behavior."""

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.pipeline import ALL_PIPELINE_OPS, ALL_PIPELINE_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.format.text.tokenizer import ParseError
from loom.ir import Module

_OPS = (
    *ALL_TEST_OPS,
    *ALL_FUNC_OPS,
    *ALL_BUFFER_OPS,
    *ALL_INDEX_OPS,
    *ALL_PIPELINE_OPS,
)
_TYPES = (*ALL_BUILTIN_TYPES, *ALL_PIPELINE_TYPES)


def _parse_module(text: str) -> Module:
    parser = Parser()
    parser.register_ops(_OPS)
    parser.register_types(_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    printer = Printer()
    printer.register_ops(_OPS)
    printer.register_types(_TYPES)
    return printer.print_module(module)


def _roundtrip(text: str) -> None:
    module = _parse_module(text)
    assert _print_module(module) == text
    assert _print_module(read_module(write_module(module))) == text


def test_generic_pipeline_roundtrip() -> None:
    _roundtrip("pipeline.def @generic(%batch: index) run(%input: buffer) {\n}\n")


@pytest.mark.parametrize("outer", ["serial", "concurrent"])
@pytest.mark.parametrize("inner", ["serial", "concurrent"])
def test_schedule_composition_roundtrip(outer: str, inner: str) -> None:
    _roundtrip(
        """pipeline.decl @stage() run(%state: buffer)

pipeline.def @ordered() run(%state: buffer) {
  pipeline.OUTER {
    pipeline.compose @stage(%state) : (buffer)
    pipeline.INNER {
      pipeline.compose @stage(%state) : (buffer)
    }
  }
}
""".replace("OUTER", outer).replace("INNER", inner)
    )


@pytest.mark.parametrize("scope", ["", "<kernel>"])
def test_pipeline_declaration_roundtrip(scope: str) -> None:
    _roundtrip(f"pipeline.decl{scope} @stage(%width: index) run(%input: channel<tile<[%width]xi32>>)\n")


def test_pipeline_declaration_empty_signature_roundtrip() -> None:
    _roundtrip("pipeline.decl<kernel> public @resident() run()\n")


def test_command_pipeline_scope_is_not_a_language_state() -> None:
    with pytest.raises(ParseError, match="invalid enum value 'command'"):
        _parse_module("pipeline.def<command> @unsupported() run() {\n}\n")


def test_strand_captures_dependent_types() -> None:
    _roundtrip(
        """func.decl @process(%width: index, %input: channel<tile<[%width]xi32>>, %count: index)

pipeline.def @placed(%column: index, %width: index) run(%input: channel<tile<[%width]xi32>>, %count: index) {
  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
    func.call @process(%width, %input, %count) : (index, channel<tile<[%width]xi32>>, index)
  }
}
"""
    )
