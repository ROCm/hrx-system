# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-language bytecode coverage for structural register value types."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.low import ALL_LOW_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.ir import EnumArrayAttr, Module, RegisterType


def _run_loom_format(arguments: list[object]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"loom-format exited with {result.returncode}:\n{result.stderr}"
        )
    return result


def _typed_register_module() -> tuple[Module, RegisterType]:
    parser = Parser()
    parser.register_ops(ALL_LOW_OPS)
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(
        "low.func.decl target<test.low.core> "
        "@typed_identity(%arg: reg<test.ptr : vector<4xi16>>) -> "
        "(reg<test.ptr : vector<4xi16>>)\n"
        "test.func @enum_arrays() {\n"
        "  test.enum_array_attrs [low, high, low] "
        "using [middle, <42>, middle]\n"
        "  test.yield\n"
        "}\n"
    )
    register_types = [
        value.type for value in module.values if isinstance(value.type, RegisterType)
    ]
    if len(register_types) != 2 or register_types[0] != register_types[1]:
        raise AssertionError("typed declaration did not preserve its signature types")
    return module, register_types[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("expected the C loom-format binary path")
    loom_format = Path(sys.argv[1])
    module, register_type = _typed_register_module()

    with tempfile.TemporaryDirectory(prefix="loom-bytecode-interop-") as temp_dir:
        temp_path = Path(temp_dir)
        python_bytecode_path = temp_path / "python.loombc"
        c_bytecode_path = temp_path / "c.loombc"
        python_bytecode_path.write_bytes(write_module(module))

        _run_loom_format(
            [
                loom_format,
                "--from=bc",
                "--to=bc",
                f"--output={c_bytecode_path}",
                python_bytecode_path,
            ]
        )
        loaded = read_module(c_bytecode_path.read_bytes())
        if not any(value.type == register_type for value in loaded.values):
            raise AssertionError(
                "Python reader did not recover the C-written structural register type"
            )
        enum_function = next(
            symbol.op for symbol in loaded.symbols if symbol.name == "enum_arrays"
        )
        if enum_function is None:
            raise AssertionError("Python reader did not recover enum_arrays")
        enum_op = enum_function.regions[0].blocks[0].ops[0]
        if enum_op.attributes["required_values"] != EnumArrayAttr([1, 255, 1]):
            raise AssertionError("required enum array did not survive C bytecode")
        if enum_op.attributes["optional_values"] != EnumArrayAttr([7, 42, 7]):
            raise AssertionError("open enum array did not survive C bytecode")


if __name__ == "__main__":
    main()
