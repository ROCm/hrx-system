# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for metadata dialect text and bytecode behavior."""

from loom.dialect.metadata import ALL_METADATA_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.verify import verify_module


def _roundtrip(text: str) -> None:
    parser = Parser()
    parser.register_ops(ALL_METADATA_OPS)
    module = parser.parse(text)

    printer = Printer()
    printer.register_ops(ALL_METADATA_OPS)
    assert printer.print_module(module) == text

    loaded = read_module(write_module(module), op_decls=ALL_METADATA_OPS)
    loaded_text = printer.print_module(loaded)
    assert loaded_text == text, f"Bytecode round-trip failed.\nInput:\n{text}\nOutput:\n{loaded_text}"


def test_module_metadata_roundtrip() -> None:
    _roundtrip(
        'metadata.module "help.summary" = "Example model"\n'
        'metadata.module "model.enabled" = true\n'
        'metadata.module "model.payload" = bytes("00feff")\n'
        'metadata.module "model.revision" = u64(18446744073709551615)\n'
    )


def test_module_metadata_key_must_be_nonempty() -> None:
    parser = Parser()
    parser.register_ops(ALL_METADATA_OPS)
    module = parser.parse('metadata.module "" = true\n')

    diagnostics = verify_module(module, ops=ALL_METADATA_OPS)

    assert diagnostics.has_errors
    assert "keyed module record key must be non-empty" in str(diagnostics.diagnostics[0])
