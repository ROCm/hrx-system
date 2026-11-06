# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored managed type declarations and resolved handle identity."""

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES, ptr_type, ref_type
from loom.dialect.test import ALL_TEST_OPS
from loom.dialect.type import ALL_TYPE_OPS
from loom.format.text.parser import Parser, parse_type_string
from loom.format.text.printer import Printer, print_type
from loom.format.text.tokenizer import ParseError
from loom.ir import FunctionType, Module, ReferenceOwnership, ReferenceType


def _parser() -> Parser:
    parser = Parser()
    parser.register_ops((*ALL_TYPE_OPS, *ALL_TEST_OPS))
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def test_external_identity_is_independent_of_local_spelling() -> None:
    module = _parser().parse(
        'type.decl import("input", "decoder") @Decoder\n'
        'type.decl import("input", "decoder") @Alias\n'
        'type.decl import("other", "decoder") @Other\n'
    )
    owned = ref_type(module=module, declaration="Decoder")
    alias = ref_type(module=module, declaration="Alias")
    borrowed = ptr_type(module=module, declaration="Decoder")
    other = ref_type(module=module, declaration="Other")
    assert owned is alias
    assert owned != borrowed
    assert owned != other
    assert owned == ReferenceType(
        ReferenceOwnership.OWNED, "input", "decoder", "Decoder"
    )
    assert print_type(owned) == "ref<@Decoder>"
    assert print_type(borrowed) == "ptr<@Decoder>"
    assert not any(symbol.is_import for symbol in module.symbols)
    reconstructed = Module(symbols=list(module.symbols))
    assert ref_type(module=reconstructed, declaration="Decoder") == owned
    assert parse_type_string(
        "(ptr<@Alias>) -> (ref<@Decoder>)", module=module
    ) == FunctionType((borrowed,), (owned,))


def test_authored_handles_roundtrip_without_provider_context() -> None:
    source = (
        'type.decl import("media", "image.frame") @ImageFrame\n\n'
        "test.decl @inspect(%frame: ptr<@ImageFrame>, %callback: (ref<@ImageFrame>) -> (ptr<@ImageFrame>))\n"
    )
    module = _parser().parse(source)
    printer = Printer()
    printer.register_ops((*ALL_TYPE_OPS, *ALL_TEST_OPS))
    printer.register_types(ALL_BUILTIN_TYPES)
    text = printer.print_module(module)
    assert text == source
    assert printer.print_module(_parser().parse(text)) == text


@pytest.mark.parametrize("mode", ["ref", "ptr"])
@pytest.mark.parametrize("prefix", ["", "test.record @Missing\n"])
def test_handle_requires_published_type_declaration(mode: str, prefix: str) -> None:
    with pytest.raises(ParseError, match="preceding managed type declaration"):
        _parser().parse(prefix + f"test.decl @use(%value: {mode}<@Missing>)\n")


@pytest.mark.parametrize("part", ["", "bad\0name", "bad\ud800name"])
def test_payload_key_is_nonempty_nul_free_utf8(part: str) -> None:
    with pytest.raises(ValueError, match=r"NUL-free UTF-8|surrogates not allowed"):
        ReferenceType(ReferenceOwnership.OWNED, "input", part, "Decoder")


def test_builder_rejects_non_symbol_parameter() -> None:
    with pytest.raises(TypeError, match="must be a symbol name"):
        ref_type(module=Module(), declaration=4)
