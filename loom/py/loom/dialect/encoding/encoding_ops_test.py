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
    ALL_ENCODING_FAMILIES,
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
    parser.register_encoding_families(ALL_ENCODING_FAMILIES)
    return parser.parse(text)


def _print_module(module: Module, *, use_aliases: bool = True) -> str:
    printer = Printer(use_aliases=use_aliases)
    printer.register_ops(_all_roundtrip_ops())
    printer.register_types(ALL_BUILTIN_TYPES)
    printer.register_encoding_families(ALL_ENCODING_FAMILIES)
    return printer.print_module(module)


def _module_text(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class TestEncodingDefineRoundTrip:
    def test_dense_shaped_attachment_is_implicit(self) -> None:
        module = _parse_module(
            _module_text(
                "test.func @f(%arg: view<4xf32, #encoding.layout.dense>) {",
                "  test.yield",
                "}",
            )
        )

        function = module.symbols[0].op
        assert function is not None
        argument_id = function.regions[0].blocks[0].arg_ids[0]
        argument = module.values[argument_id]
        assert argument.type.encoding is None
        assert _print_module(module) == _module_text(
            "test.func @f(%arg: view<4xf32>) {",
            "  test.yield",
            "}",
        )

    def test_inline_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "test.func @f() {",
                "  %enc = encoding.define #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes> : encoding<schema>",
                "  test.yield",
                "}",
            )
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["spec"] == EncodingInstance(
            name="encoding.operand",
            params=(
                ("element_format", "i8"),
                ("payload_elements", 32),
                ("payload_packing", "dense_lanes"),
            ),
        )
        assert module.encodings == [
            EncodingInstance(
                name="encoding.operand",
                params=(
                    ("element_format", "i8"),
                    ("payload_elements", 32),
                    ("payload_packing", "dense_lanes"),
                ),
            )
        ]

    def test_canonical_numeric_alias_has_structural_identity(self) -> None:
        module = _parse_module(
            _module_text(
                "test.func @f() {",
                "  %enc = encoding.define #encoding.f8e4m3fn : encoding<schema>",
                "  test.yield",
                "}",
            )
        )

        assert module.encodings == [
            EncodingInstance(
                name="encoding.operand",
                params=(
                    ("element_format", "f8e4m3fn"),
                    ("payload_elements", 1),
                    ("payload_packing", "dense_lanes"),
                ),
            )
        ]
        assert _print_module(module) == _module_text(
            "test.func @f() {",
            "  %enc = encoding.define #encoding.f8e4m3fn : encoding<schema>",
            "  test.yield",
            "}",
        )
        assert _print_module(module, use_aliases=False) == _module_text(
            "test.func @f() {",
            "  %enc = encoding.define #encoding.operand<element_format=f8e4m3fn, payload_elements=1, payload_packing=dense_lanes> : encoding<schema>",
            "  test.yield",
            "}",
        )

    def test_canonical_numeric_alias_rejects_fixed_parameter_override(self) -> None:
        with pytest.raises(ParseError, match="parameter cannot be restated"):
            _parse_module(
                _module_text(
                    "test.func @f() {",
                    "  %enc = encoding.define #encoding.f8e4m3fn<element_format=f16> : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_dynamic_params_print_in_canonical_order(self) -> None:
        module = _parse_module(
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes> "
            "{scales = %scales : tensor<[%group_size]xf32>, "
            "group_size = %group_size : index} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        printed = _print_module(module)
        expected = (
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #encoding.i8<payload_elements=32> "
            "{group_size = %group_size : index, "
            "scales = %scales : tensor<[%group_size]xf32>} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        assert printed == expected, printed
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["param_names"] == CanonicalAttrDict((("group_size", 0), ("scales", 1)))

    def test_rejects_static_dynamic_duplicate(self) -> None:
        with pytest.raises(ParseError, match="both static and dynamic"):
            _parse_module(
                _module_text(
                    "test.func @f(%payload_elements: index) {",
                    "  %enc = encoding.define #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes> {payload_elements = %payload_elements : index} : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_static_encoding_rejects_ssa_parameter(self) -> None:
        with pytest.raises(ParseError, match="cannot use an SSA value"):
            _parse_module(
                _module_text(
                    "test.func @f(%group_size: index) {",
                    "  %enc = encoding.define #encoding.operand<element_format=i8, payload_elements=%group_size, payload_packing=dense_lanes> : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_alias_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "#enc = #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes>",
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
            name="encoding.operand",
            alias="enc",
            params=(
                ("element_format", "i8"),
                ("payload_elements", 32),
                ("payload_packing", "dense_lanes"),
            ),
        )
        assert module.encodings == [
            EncodingInstance(
                name="encoding.operand",
                alias="enc",
                params=(
                    ("element_format", "i8"),
                    ("payload_elements", 32),
                    ("payload_packing", "dense_lanes"),
                ),
            )
        ]
