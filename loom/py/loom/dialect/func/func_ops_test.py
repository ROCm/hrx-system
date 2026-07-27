# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for func dialect text and bytecode behavior."""

from collections.abc import Sequence

import pytest

from loom.dialect.func import ALL_FUNC_OPS
from loom.dsl import Op
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.ir import Module


def _all_roundtrip_ops() -> Sequence[Op]:
    """Returns ops used by func dialect text round-trip fixtures."""
    from loom.dialect.test import ALL_TEST_OPS

    return list(ALL_TEST_OPS) + list(ALL_FUNC_OPS)


def _parse_module(text: str) -> Module:
    """Parses a module with func and test dialect fixtures registered."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.format.text.parser import Parser

    parser = Parser()
    parser.register_ops(_all_roundtrip_ops())
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    """Prints a module with func and test dialect fixtures registered."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.format.text.printer import Printer

    printer = Printer()
    printer.register_ops(_all_roundtrip_ops())
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


def _roundtrip(text: str) -> None:
    """Parse text, print, assert identical."""
    module = _parse_module(text)
    printed = _print_module(module)

    assert printed == text, f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"


def _module_text(*lines: str) -> str:
    """Returns newline-terminated Loom text from greppable source lines."""
    return "\n".join(lines) + "\n"


class TestFuncDefRoundTrip:
    def test_simple(self) -> None:
        _roundtrip("func.def @add(%a: f32, %b: f32) -> (f32) {\n  %r = test.addi %a, %b : f32\n  func.return %r : f32\n}\n")

    def test_no_results(self) -> None:
        _roundtrip("func.def @noop(%a: f32) {\n  func.return %a : f32\n}\n")

    def test_public(self) -> None:
        _roundtrip("func.def public @entry(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_public_device(self) -> None:
        _roundtrip("func.def public device @kernel(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_device_only(self) -> None:
        _roundtrip("func.def device @helper(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_with_predicates(self) -> None:
        _roundtrip("func.def @constrained(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)] {\n  func.return %a : tensor<[%M]xf32>\n}\n")


class TestFuncDeclRoundTrip:
    def test_simple(self) -> None:
        _roundtrip("func.decl @extern_add(%a: f32, %b: f32) -> (f32)\n")

    def test_public(self) -> None:
        _roundtrip("func.decl public @exported(%a: f32) -> (f32)\n")

    def test_with_predicates(self) -> None:
        _roundtrip("func.decl @constrained(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)]\n")

    def test_no_results(self) -> None:
        _roundtrip("func.decl @sink(%a: f32)\n")

    def test_def_without_body_is_error(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.text.parser import Parser
        from loom.format.text.tokenizer import ParseError

        parser = Parser()
        parser.register_ops(list(ALL_TEST_OPS) + list(ALL_FUNC_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        with pytest.raises(ParseError, match="LBRACE"):
            parser.parse("func.def @bad(%a: f32) -> (f32)\n")

    def test_decl_with_body_is_error(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.text.parser import Parser
        from loom.format.text.tokenizer import ParseError

        parser = Parser()
        parser.register_ops(list(ALL_TEST_OPS) + list(ALL_FUNC_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        with pytest.raises(ParseError, match=r"LBRACE|top-level op"):
            parser.parse("func.decl @bad(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")


class TestFuncImportParsing:
    def test_basic_import(self) -> None:
        module = _parse_module(_module_text('func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)'))
        assert len(module.symbols) == 1
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.source_module == "linalg_lib"
        assert sym.source_symbol == ""
        assert sym.name == "matmul"
        assert sym.op is not None
        assert not sym.op.regions

    def test_import_with_alias(self) -> None:
        module = _parse_module(_module_text('func.decl import("math_lib", "matmul_v2") @my_matmul(%a: f32) -> (f32)'))
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.source_module == "math_lib"
        assert sym.source_symbol == "matmul_v2"
        assert sym.name == "my_matmul"

    def test_public_import(self) -> None:
        module = _parse_module(_module_text('func.decl public import("upstream") @relu(%x: f32) -> (f32)'))
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.is_public
        assert sym.source_module == "upstream"

    def test_import_with_types(self) -> None:
        module = _parse_module(_module_text('func.decl import("kernels") @conv(%N: index, %w: tensor<3x3xf32>, %x: tensor<[%N]xf32>) -> (tensor<[%N]xf32>)'))
        sym = module.symbols[0]
        assert sym.is_import
        op = sym.op
        assert op is not None
        assert len(op.operands) == 3
        assert len(op.results) == 1

    def test_public_import_canonical_order(self) -> None:
        module = _parse_module(_module_text('func.decl public import("lib") @f(%a: f32) -> (f32)'))
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.is_public

    def test_non_import_has_no_source(self) -> None:
        module = _parse_module(_module_text("func.decl @extern(%a: f32) -> (f32)"))
        sym = module.symbols[0]
        assert not sym.is_import
        assert sym.source_module == ""

    def test_import_in_multi_function_module(self) -> None:
        module = _parse_module(
            _module_text(
                'func.decl import("lib") @imported(%a: f32) -> (f32)',
                "",
                "func.def @local(%x: f32) -> (f32) {",
                "  func.return %x : f32",
                "}",
            )
        )
        assert len(module.symbols) == 2
        syms = {s.name: s for s in module.symbols}
        assert syms["imported"].is_import
        assert syms["imported"].source_module == "lib"
        assert not syms["local"].is_import


class TestFuncImportRoundTrip:
    def test_basic_import_roundtrip(self) -> None:
        _roundtrip(_module_text('func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)'))

    def test_import_alias_roundtrip(self) -> None:
        _roundtrip(_module_text('func.decl import("math_lib", "matmul") @my_matmul(%a: f32) -> (f32)'))

    def test_public_import_roundtrip(self) -> None:
        _roundtrip(_module_text('func.decl public import("upstream") @relu(%x: f32) -> (f32)'))

    def test_mixed_module_roundtrip(self) -> None:
        _roundtrip(
            _module_text(
                'func.decl import("lib") @imported(%a: f32) -> (f32)',
                "",
                "func.def @local(%x: f32) -> (f32) {",
                "  func.return %x : f32",
                "}",
            )
        )


class TestFuncImportCrossFormatRoundTrip:
    def _cross_roundtrip(self, text: str, expected: str | None = None) -> None:
        module = _parse_module(text)
        loaded = read_module(write_module(module))
        printed = _print_module(loaded)
        target = expected if expected is not None else text
        assert printed == target, f"Cross-format round-trip mismatch:\n  expected: {target!r}\n  got:      {printed!r}"

    def test_import_survives_bytecode(self) -> None:
        self._cross_roundtrip(
            _module_text('func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)'),
        )

    def test_import_alias_survives_bytecode(self) -> None:
        self._cross_roundtrip(
            _module_text('func.decl import("math_lib", "matmul") @my_matmul(%a: f32) -> (f32)'),
        )

    def test_import_metadata_preserved(self) -> None:
        module = _parse_module(_module_text('func.decl import("math_lib", "original") @alias(%a: f32) -> (f32)'))
        loaded = read_module(write_module(module))
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.source_module == "math_lib"
        assert sym.source_symbol == "original"
        assert sym.name == "alias"

    def test_mixed_module_survives_bytecode(self) -> None:
        self._cross_roundtrip(
            _module_text(
                'func.decl import("lib") @imported(%a: f32) -> (f32)',
                "",
                "func.def @local(%x: f32) -> (f32) {",
                "  func.return %x : f32",
                "}",
            ),
        )


class TestFuncCallApplyRoundTrip:
    """func.call and func.apply are body ops — they appear inside functions."""

    def test_call_in_function(self) -> None:
        _roundtrip("func.def @caller(%a: f32) -> (f32) {\n  %r = func.call @callee(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_apply_in_function(self) -> None:
        _roundtrip("func.def @test_template(%a: f32) -> (f32) {\n  %r = func.apply<my.template>(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_call_multiple_args_and_results(self) -> None:
        _roundtrip("func.def @multi(%a: f32, %b: i32) -> (f32, i32) {\n  %r0, %r1 = func.call @process(%a, %b) : (f32, i32) -> (f32, i32)\n  func.return %r0, %r1 : f32, i32\n}\n")


class TestFuncReturnRoundTrip:
    def test_no_operands(self) -> None:
        _roundtrip("func.def @empty() {\n  func.return\n}\n")

    def test_no_operands_initializer(self) -> None:
        _roundtrip("func.def initializer @setup() {\n  func.return\n}\n")

    def test_single_value(self) -> None:
        _roundtrip("func.def @ret(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_multiple_values(self) -> None:
        _roundtrip("func.def @multi_ret(%a: f32, %b: i32) -> (f32, i32) {\n  func.return %a, %b : f32, i32\n}\n")


class TestMultipleFunctions:
    def test_decl_then_def(self) -> None:
        _roundtrip("func.decl @extern(%a: f32) -> (f32)\n\nfunc.def @main(%a: f32) -> (f32) {\n  %r = func.call @extern(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_multiple_defs(self) -> None:
        _roundtrip("func.def @f1(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n\nfunc.def @f2(%a: i32) -> (i32) {\n  func.return %a : i32\n}\n")


class TestFuncTemplateUkernelRoundTrip:
    def test_template_basic(self) -> None:
        _roundtrip(
            _module_text(
                "func.template<tile.contract> @impl(%a: tile<4xf32>) -> (tile<4xf32>) {",
                "  func.return %a : tile<4xf32>",
                "}",
            )
        )

    def test_template_with_priority(self) -> None:
        _roundtrip(
            _module_text(
                "func.template<tile.contract> priority(10) @high_priority(%a: tile<4xf32>) -> (tile<4xf32>) {",
                "  func.return %a : tile<4xf32>",
                "}",
            )
        )

    def test_template_device_cc(self) -> None:
        _roundtrip(
            _module_text(
                "func.template<tile.contract> device @device_impl(%a: tile<4xf32>) -> (tile<4xf32>) {",
                "  func.return %a : tile<4xf32>",
                "}",
            )
        )

    def test_ukernel_basic(self) -> None:
        _roundtrip(_module_text("func.ukernel<tile.contract> @asm_impl(%a: tile<4xf32>) -> (tile<4xf32>)"))

    def test_ukernel_device(self) -> None:
        _roundtrip(_module_text("func.ukernel<tile.contract> device @asm_device(%a: tile<4xf32>) -> (tile<4xf32>)"))

    def test_ukernel_with_priority(self) -> None:
        _roundtrip(_module_text("func.ukernel<tile.contract> priority(5) @prioritized(%a: f32) -> (f32)"))

    def test_template_implements_stored(self) -> None:
        module = _parse_module(
            _module_text(
                "func.template<tile.contract> @impl(%a: f32) -> (f32) {",
                "  func.return %a : f32",
                "}",
            )
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("implements") == "tile.contract"
        assert op.attributes.get("priority") is None

    def test_template_priority_stored(self) -> None:
        module = _parse_module(
            _module_text(
                "func.template<tile.contract> priority(42) @impl(%a: f32) -> (f32) {",
                "  func.return %a : f32",
                "}",
            )
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("implements") == "tile.contract"
        assert op.attributes.get("priority") == 42
