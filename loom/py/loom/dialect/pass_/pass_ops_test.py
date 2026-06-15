# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the pass pipeline control dialect."""

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.pass_ import ALL_PASS_OPS
from loom.format.bytecode.reader import BytecodeReader
from loom.format.bytecode.writer import BytecodeWriter
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer


def _roundtrip_text(source: str) -> str:
    parser = Parser()
    parser.register_ops(ALL_PASS_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(source)
    printer = Printer()
    printer.register_ops(ALL_PASS_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


class TestPassText:
    def test_pipeline_roundtrip(self) -> None:
        source = """pass.pipeline<module> @cleanup pipeline {
  canonicalize(max_iterations = 10)
  repeat until_converged(max_iterations = 8) {
    cse
    dce
  }
}
"""
        assert _roundtrip_text(source) == source

    def test_func_pipeline_roundtrip(self) -> None:
        source = """pass.pipeline<func> @function_cleanup pipeline {
  for func {
    where name(value = "matmul") {
      vector-memory-footprint(budget_bytes = 4096)
    }
  }
}
"""
        assert _roundtrip_text(source) == source

    def test_debug_ops_roundtrip(self) -> None:
        source = """pass.pipeline<module> @debug pipeline {
  call @cleanup
  fail "expected cleanup to run"
  halt "inspect IR"
}

pass.pipeline<module> @cleanup pipeline {
}
"""
        assert _roundtrip_text(source) == source

    def test_canonical_pipeline_region_still_parses(self) -> None:
        source = """pass.pipeline<module> @cleanup {
  pass.run<canonicalize> {max_iterations = 10}
  pass.yield
}
"""
        expected = """pass.pipeline<module> @cleanup pipeline {
  canonicalize(max_iterations = 10)
}
"""
        assert _roundtrip_text(source) == expected


class TestPassBytecode:
    def test_pipeline_bytecode_roundtrip(self) -> None:
        source = """pass.pipeline<module> @cleanup pipeline {
  unknown-pass(message = "preserve structurally")
}
"""
        parser = Parser()
        parser.register_ops(ALL_PASS_OPS)
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(source)

        data = BytecodeWriter(module, op_decls=ALL_PASS_OPS).write()
        loaded = BytecodeReader(data, op_decls=ALL_PASS_OPS).read()

        printer = Printer()
        printer.register_ops(ALL_PASS_OPS)
        printer.register_types(ALL_BUILTIN_TYPES)
        assert printer.print_module(loaded) == source
