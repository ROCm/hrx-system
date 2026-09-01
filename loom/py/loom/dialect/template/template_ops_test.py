# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for compile-time template family operations."""

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS, ALL_FUNC_TYPES
from loom.dialect.target import ALL_TARGET_OPS, ALL_TARGET_PARAMETERIZED_ATTRS
from loom.dialect.template import ALL_TEMPLATE_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.format.text.tokenizer import ParseError
from loom.ir import Module, ParameterizedAttrArray
from loom.verify import verify_module


def _parse(source: str) -> Module:
    ops = [*ALL_FUNC_OPS, *ALL_TARGET_OPS, *ALL_TEMPLATE_OPS]
    parser = Parser()
    parser.register_ops(ops)
    parser.register_types((*ALL_BUILTIN_TYPES, *ALL_FUNC_TYPES))
    parser.register_parameterized_attrs(ALL_TARGET_PARAMETERIZED_ATTRS)
    module = parser.parse(source)
    assert not verify_module(module, ops=ops).has_errors
    return module


def _print(module: Module) -> str:
    ops = [*ALL_FUNC_OPS, *ALL_TARGET_OPS, *ALL_TEMPLATE_OPS]
    printer = Printer()
    printer.register_ops(ops)
    printer.register_types((*ALL_BUILTIN_TYPES, *ALL_FUNC_TYPES))
    return printer.print_module(module)


def _roundtrip(source: str) -> str:
    return _print(_parse(source))


def test_family_definition_apply_and_exact_call_roundtrip() -> None:
    source = """\
template.decl public @demo.scale(%value: i32) -> (i32)

template.def<@demo.scale> priority(10) @scale_fast(%value: i32) -> (i32) {
  template.return %value : i32
}

func.def public @entry(%value: i32) -> (i32) {
  %selected = template.apply<@demo.scale>(%value) : (i32) -> (i32)
  %exact = template.call @scale_fast(%selected) : (i32) -> (i32)
  func.return %exact : i32
}
"""

    assert _roundtrip(source) == source


def test_family_definition_survives_bytecode() -> None:
    source = """\
template.decl @demo.scale(%value: i32) -> (i32)

template.def<@demo.scale> priority(42) @scale(%value: i32) -> (i32) {
  template.return %value : i32
}
"""

    loaded = read_module(write_module(_parse(source)))
    assert _print(loaded) == source

    provider = loaded.symbols[1].op
    assert provider is not None
    family = provider.attributes.get("family")
    assert family == "demo.scale"
    assert provider.attributes.get("priority") == 42


def test_provider_requirements_and_predicates_are_distinct() -> None:
    source = """\
template.decl @demo.tile(%size: index, %value: f32) -> (f32)

template.def<@demo.tile> requires [#target.subgroup.size<64>] @wave64(%size: index, %value: f32) -> (f32) where [eq(%size, 32)] {
  template.return %value : f32
}
"""

    module = _parse(source)
    provider = module.symbols[1].op
    assert provider is not None
    requirements = provider.attributes.get("requires")
    predicates = provider.attributes.get("predicates")
    assert isinstance(requirements, ParameterizedAttrArray)
    assert requirements.values[0].family_name == "target.subgroup.size"
    assert isinstance(predicates, list)
    assert len(predicates) == 1


def test_provider_rejects_requirements_after_priority() -> None:
    source = """\
template.decl @demo.scale(%value: i32) -> (i32)

template.def<@demo.scale> priority(10) requires [#target.subgroup.size<64>] @scale(%value: i32) -> (i32) {
  template.return %value : i32
}
"""

    with pytest.raises(ParseError):
        _parse(source)


def test_family_and_provider_requirements_roundtrip() -> None:
    source = """\
template.decl requires [#target.subgroup.size<32>] @demo.wave(%value: i32) -> (i32)

template.ukernel<@demo.wave> requires [#target.subgroup.size<32>] priority(20) @wave_asm(%value: i32) -> (i32)
"""

    assert _roundtrip(source) == source
