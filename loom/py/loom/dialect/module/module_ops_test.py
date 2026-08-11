# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for compile-time module composition metadata."""

import pytest

from loom.builders import module_builder
from loom.dialect.module import ALL_MODULE_OPS
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.format.text.tokenizer import ParseError
from loom.ir import SymbolName, SymbolNameSet
from loom.verify import verify_module


def _roundtrip_text(source: str) -> str:
    parser = Parser()
    parser.register_ops(ALL_MODULE_OPS)
    module = parser.parse(source)
    assert not verify_module(module, ops=ALL_MODULE_OPS).has_errors

    printer = Printer()
    printer.register_ops(ALL_MODULE_OPS)
    return printer.print_module(module)


def test_import_roundtrip_accepts_shared_availability_anchor() -> None:
    source = 'module.import "motif/format/ggml.loom" [@decode_q4, @decode_q6]\nmodule.import "target/amdgpu/format/ggml.loom" [@decode_q4]\n'

    assert _roundtrip_text(source) == source


def test_import_roundtrip_sorts_anchors_by_exact_name() -> None:
    source = 'module.import "provider" [@zeta, @alpha, @middle]\n'
    expected = 'module.import "provider" [@alpha, @middle, @zeta]\n'

    assert _roundtrip_text(source) == expected


def test_import_parser_rejects_duplicate_anchor() -> None:
    parser = Parser()
    parser.register_ops(ALL_MODULE_OPS)

    with pytest.raises(ParseError, match="duplicate symbol name '@decode_q4'"):
        parser.parse('module.import "provider" [@decode_q4, @decode_q4]\n')


def test_dynamic_builder_uses_unambiguous_module_namespace() -> None:
    module, builder = module_builder(ops=ALL_MODULE_OPS)

    builder.module_.import_(
        provider="motif/format/ggml.loom",
        symbols=["decode_q6", "decode_q4"],
    )

    assert len(module.body.ops) == 1
    import_op = module.body.ops[0]
    assert import_op.attributes["provider"] == "motif/format/ggml.loom"
    assert import_op.attributes["symbols"] == SymbolNameSet([SymbolName("decode_q4"), SymbolName("decode_q6")])


def test_dynamic_builder_rejects_duplicate_anchor() -> None:
    _, builder = module_builder(ops=ALL_MODULE_OPS)

    with pytest.raises(ValueError, match="duplicate symbol name '@decode_q4'"):
        builder.module_.import_(
            provider="provider",
            symbols=["decode_q4", "decode_q4"],
        )
