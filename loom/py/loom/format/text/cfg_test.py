# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CFG identity and direct type bindings through public text serialization."""

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.cfg import ALL_CFG_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import ParseError, Parser
from loom.format.text.printer import Printer
from loom.ir import Block, Module
from loom.verify import verify_module


def _formats() -> tuple[Parser, Printer]:
    parser = Parser()
    printer = Printer()
    for format in (parser, printer):
        format.register_types(ALL_BUILTIN_TYPES)
        for ops in (ALL_CFG_OPS, ALL_FUNC_OPS, ALL_SCF_OPS, ALL_TEST_OPS):
            format.register_ops(ops)
    return parser, printer


def _roundtrip(module: Module) -> tuple[Module, str]:
    parser, printer = _formats()
    text = printer.print_module(module)
    loaded = parser.parse(text, verify=True)
    assert printer.print_module(loaded) == text
    return loaded, text


def test_cfg_argument_keeps_dimension_and_encoding_bindings() -> None:
    parser, _ = _formats()
    module = parser.parse(
        "func.def @f(%extent: index, %layout: encoding, "
        "%value: tile<[%extent]xf32, %layout>) {\n"
        "  cfg.br ^next(%value: tile<[%extent]xf32, %layout>)\n"
        "^next(%forwarded: tile<[%extent]xf32, %layout>):\n"
        "  test.use %forwarded : tile<[%extent]xf32, %layout>\n"
        "  func.return\n"
        "}\n",
        verify=True,
    )
    loaded, _ = _roundtrip(module)
    for candidate in (module, loaded, read_module(write_module(module))):
        first, second = candidate.body.ops[0].regions[0].blocks
        assert first.ops[0].successors[0] is second
        argument = candidate.values[second.arg_ids[0]]
        assert argument.dim_bindings == {0: first.arg_ids[0]}
        assert argument.encoding_binding == first.arg_ids[1]
        assert second.ops[0].operands[0] == second.arg_ids[0]


@pytest.mark.parametrize("label", ["", "loop", "0"])
def test_loop_successors_keep_identity_with_reused_labels(label: str) -> None:
    parser, printer = _formats()
    module = parser.parse(
        "func.def @loop(%condition: i1) {\n"
        "^entry:\n"
        "  cfg.br ^header\n"
        "^header:\n"
        "  cfg.br ^body\n"
        "^body:\n"
        "  cfg.cond_br %condition, ^header, ^exit\n"
        "^exit:\n"
        "  func.return\n"
        "}\n",
        verify=True,
    )
    blocks = module.body.ops[0].regions[0].blocks
    for block in blocks:
        block.label = label
    before = list(blocks)
    loaded, _ = _roundtrip(module)
    assert all(
        block is original for block, original in zip(blocks, before, strict=True)
    )
    assert [block.label for block in blocks] == [label] * 4
    entry, header, body, exit = loaded.body.ops[0].regions[0].blocks
    assert entry.ops[0].successors[0] is header
    assert header.ops[0].successors[0] is body
    assert body.ops[0].successors[0] is header
    assert body.ops[0].successors[1] is exit
    assert body.ops[0].operands[0] == entry.arg_ids[0]
    assert len({entry.label, header.label, body.label, exit.label}) == 4
    # Single-op printing uses the containing module's label plan too.
    assert printer.print_operation(blocks[0].ops[0], module).startswith("cfg.br ^")


def test_print_order_follows_dominance_without_mutating_block_storage() -> None:
    parser, _ = _formats()
    module = parser.parse(
        "func.def @f() {\n"
        "  cfg.br ^define\n"
        "^define:\n"
        "  %value = test.constant 7 : index\n"
        "  cfg.br ^use\n"
        "^use:\n"
        "  test.use %value : index\n"
        "  func.return\n"
        "}\n",
        verify=True,
    )
    region = module.body.ops[0].regions[0]
    entry, definition, use = region.blocks
    region.blocks[:] = [entry, use, definition]
    loaded, text = _roundtrip(module)
    assert text.index("^define:") < text.index("^use:")
    assert region.blocks[1] is use
    first, definition, use = loaded.body.ops[0].regions[0].blocks
    assert first.ops[0].successors[0] is definition
    assert definition.ops[1].successors[0] is use
    assert use.ops[0].operands[0] == definition.ops[0].results[0]


def test_labels_are_local_to_independent_regions() -> None:
    parser, _ = _formats()
    module = parser.parse(
        "func.def @first() {\n^entry:\n  cfg.br ^done\n"
        "^done:\n  func.return\n}\n"
        "func.def @second() {\n^entry:\n  cfg.br ^done\n"
        "^done:\n  func.return\n}\n",
        verify=True,
    )
    loaded, _ = _roundtrip(module)
    first, second = [op.regions[0].blocks for op in loaded.body.ops]
    assert first[0].ops[0].successors[0] is first[1]
    assert second[0].ops[0].successors[0] is second[1]
    assert first[1] is not second[1]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("cfg.br ^missing", "undefined block label"),
        ("^same:\n cfg.br ^same\n^same:\n func.return", "duplicate block label"),
        (
            "scf.if %condition { cfg.br ^outer }\n^outer:\n func.return",
            "undefined block label",
        ),
        (
            "test.use %later : index\n%later = test.constant 1 : index\nfunc.return",
            "undefined SSA value",
        ),
        (
            "cfg.br ^next(%later: index)\n^next:\n%later = test.constant 1 : index\nfunc.return",
            "undefined SSA value",
        ),
    ],
)
def test_invalid_labels_and_ssa_uses_fail_during_parse(body: str, message: str) -> None:
    parser, _ = _formats()
    with pytest.raises(ParseError, match=message):
        parser.parse(f"func.def @f(%condition: i1) {{\n{body}\n}}")


def test_unterminated_region_fails_without_reparsing_an_empty_block() -> None:
    parser, _ = _formats()
    with pytest.raises(ParseError, match="RBRACE"):
        parser.parse("func.def @f() { cfg.br ^end\n^end:")


def test_verifier_rejects_foreign_successor_identity() -> None:
    parser, _ = _formats()
    module = parser.parse("func.def @f() { ^entry: cfg.br ^entry }")
    branch = module.body.ops[0].regions[0].blocks[0].ops[0]
    branch.successors[0] = Block(label="entry")
    diagnostics = verify_module(module, ops=(*ALL_FUNC_OPS, *ALL_CFG_OPS))
    assert any("successor is outside" in d.message for d in diagnostics.diagnostics)
