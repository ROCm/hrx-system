# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for encoding dialect text behavior."""

from collections.abc import Sequence

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.encoding import (
    ALL_ENCODING_OPS,
)
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import Op
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.format.text.tokenizer import ParseError
from loom.ir import CanonicalAttrDict, EncodingInstance, Module


def _all_roundtrip_ops() -> Sequence[Op]:
    return list(ALL_TEST_OPS) + list(ALL_ENCODING_OPS)


def _parse_module(text: str) -> Module:
    parser = Parser()
    parser.register_ops(_all_roundtrip_ops())
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    printer = Printer()
    printer.register_ops(_all_roundtrip_ops())
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


def _module_text(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class TestEncodingDefineRoundTrip:
    def test_inline_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "test.func @f() {",
                "  %enc = encoding.define #q8_0<block=32> : encoding<schema>",
                "  test.yield",
                "}",
            )
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["spec"] == EncodingInstance(
            name="q8_0",
            params=(("block", 32),),
        )
        assert module.encodings == [EncodingInstance(name="q8_0", params=(("block", 32),))]

    def test_dynamic_params_print_in_canonical_order(self) -> None:
        module = _parse_module(
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{scales = %scales : tensor<[%group_size]xf32>, "
            "group_size = %group_size : index} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        printed = _print_module(module)
        assert printed == (
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{group_size = %group_size : index, "
            "scales = %scales : tensor<[%group_size]xf32>} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["param_names"] == CanonicalAttrDict((("group_size", 0), ("scales", 1)))

    def test_rejects_static_dynamic_duplicate(self) -> None:
        with pytest.raises(ParseError, match="both static and dynamic"):
            _parse_module(
                _module_text(
                    "test.func @f(%block: index) {",
                    "  %enc = encoding.define #q8_0<block=32> {block = %block : index} : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_static_encoding_rejects_ssa_parameter(self) -> None:
        with pytest.raises(ParseError, match="cannot use an SSA value"):
            _parse_module(
                _module_text(
                    "test.func @f(%group_size: index) {",
                    "  %enc = encoding.define #q8_0<group_size=%group_size> : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_alias_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "#enc = #q8_0<block=32>",
                "test.func @f() {",
                "  %enc = encoding.define #enc : encoding<schema>",
                "  test.yield",
                "}",
            )
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.attributes["spec"] == EncodingInstance(
            name="q8_0",
            alias="enc",
            params=(("block", 32),),
        )
        assert module.encodings == [
            EncodingInstance(
                name="q8_0",
                alias="enc",
                params=(("block", 32),),
            )
        ]
