# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for command dialect text and bytecode behavior."""

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.command import ALL_COMMAND_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.ir import Module


def _parse_module(text: str) -> Module:
    parser = Parser()
    parser.register_ops([*ALL_TEST_OPS, *ALL_COMMAND_OPS])
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    printer = Printer()
    printer.register_ops([*ALL_TEST_OPS, *ALL_COMMAND_OPS])
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


def _roundtrip(text: str) -> None:
    module = _parse_module(text)
    assert _print_module(module) == text
    assert _print_module(read_module(write_module(module))) == text


def test_program_definition_and_launch_roundtrip() -> None:
    text = """test.target<low_core> @gfx1100

command.program.def public target(@gfx1100) @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer) {
  command.return
}

command.program.def @driver(%token_count: index) launch(%parameters: buffer, %transient: buffer) {
  command.program.launch @decode[%token_count](%parameters, %transient) : [index](buffer, buffer)
  command.return
}
"""
    module = _parse_module(text)
    definitions = {symbol.name: symbol.op for symbol in module.symbols if symbol.op is not None and symbol.op.name == "command.program.def"}
    assert definitions["decode"].attributes["specialization_count"] == 1
    assert definitions["driver"].attributes["specialization_count"] == 1
    _roundtrip(text)


def test_program_declaration_roundtrip() -> None:
    text = "command.program.decl @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer)\n"
    module = _parse_module(text)
    declaration = module.symbols[0].op
    assert declaration is not None
    assert declaration.attributes["specialization_count"] == 1
    _roundtrip(text)


def test_program_without_specializations_roundtrip() -> None:
    _roundtrip("command.program.def @static_program() launch(%parameters: buffer) {\n  command.return\n}\n")


def test_parameter_roundtrip() -> None:
    _roundtrip(
        """command.program.def @parameters(%layer: index) launch(%parameters: buffer) {
  %embedding = command.parameter %parameters, "token_embd.weight" : view<1024xi8>
  %query = command.parameter %parameters, "blk.{}.attn_q.weight"[%layer] : view<256xi8>
  command.return
}
"""
    )


def test_structured_command_schedules_roundtrip() -> None:
    _roundtrip(
        """command.program.def @scheduled() launch(%parameters: buffer) {
  command.serial {
    command.concurrent {
      command.program.launch @leaf(%parameters) : (buffer)
    }
  }
  command.return
}

command.program.def @leaf() launch(%parameters: buffer) {
  command.return
}
"""
    )
