# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for target-low op text behavior."""

from collections.abc import Mapping
from typing import Any

import pytest

from loom.builders import module_builder
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.low import ALL_LOW_OPS
from loom.format.text.parser import NameScope, Parser
from loom.format.text.printer import Printer
from loom.ir import Block, Module, RegisterType, StorageSpace, StorageType, Type, Value
from loom.stable_id import stable_id_from_string
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET
from loom.value_tree import OUT

_TEST_LOW_CORE_STABLE_ID = stable_id_from_string(TEST_LOW_CORE_DESCRIPTOR_SET.key)
_TEST_PTR_REGISTER_CLASS_ID = next(i for i, register_class in enumerate(TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes) if register_class.name == "test.ptr")


def _parser() -> Parser:
    parser = Parser()
    parser.register_ops(ALL_LOW_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _printer() -> Printer:
    printer = Printer()
    printer.register_ops(ALL_LOW_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer


def _scope_with_values(
    *names_and_types: tuple[str, Type],
) -> tuple[Module, NameScope]:
    module = Module()
    scope = NameScope()
    for name, value_type in names_and_types:
        value_id = module.add_value(Value(name=name, type=value_type))
        scope.define(name, value_id)
    return module, scope


def _test_ptr_register_type(unit_count: int = 1) -> RegisterType:
    return RegisterType(
        _TEST_LOW_CORE_STABLE_ID,
        _TEST_PTR_REGISTER_CLASS_ID,
        unit_count,
        "test.ptr",
    )


def _parse_and_print(text: str, module: Module, scope: NameScope) -> tuple[Mapping[str, Any], str]:
    op = _parser().parse_operation_from_text(text, module=module, scope=scope)
    return op.attributes, _printer().print_operation(op, module)


def test_storage_address_omitted_default_offset_is_restored() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot : low.storage<workgroup>",
            "-> reg<test.ptr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == source


def test_storage_address_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot {offset = 0} :",
            "low.storage<workgroup> -> reg<test.ptr>",
        ]
    )
    expected = " ".join(
        [
            "%addr = low.storage.address %slot : low.storage<workgroup>",
            "-> reg<test.ptr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected


def test_storage_address_non_default_offset_is_preserved() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot {offset = 16} :",
            "low.storage<workgroup> -> reg<test.ptr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 16
    assert text == source


def test_spill_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "low.spill %value, %slot {offset = 0} :",
            "reg<test.ptr x4>, low.storage<private>",
        ]
    )
    expected = " ".join(
        [
            "low.spill %value, %slot :",
            "reg<test.ptr x4>, low.storage<private>",
        ]
    )
    module, scope = _scope_with_values(
        ("value", _test_ptr_register_type(4)),
        ("slot", StorageType(StorageSpace.PRIVATE)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected


def test_reload_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "%reload = low.reload %slot {offset = 0} :",
            "low.storage<private> -> reg<test.ptr x4>",
        ]
    )
    expected = " ".join(
        [
            "%reload = low.reload %slot :",
            "low.storage<private> -> reg<test.ptr x4>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.PRIVATE)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected


def test_structured_call_inputs_and_results_round_trip() -> None:
    source = "%first, %second = low.func.call @native([(%a, out), (%b, out)], []) : (reg<test.ptr>, reg<test.ptr>) -> (reg<test.ptr>, reg<test.ptr>)"
    module, scope = _scope_with_values(("a", _test_ptr_register_type()), ("b", _test_ptr_register_type()))
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["shape"] == [10, 11, 10, 0, 4, 10, 0, 4, 3]
    assert text == source


def test_flat_call_keeps_implicit_grouping() -> None:
    source = "%result = low.func.call @ordinary(%a) : (reg<test.ptr>) -> (reg<test.ptr>)"
    module, scope = _scope_with_values(("a", _test_ptr_register_type()))
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs.get("shape") is None
    assert text == source


def test_call_builder_preserves_nested_result_slots() -> None:
    block = Block()
    module, builder = module_builder(insertion_block=block, ops=ALL_LOW_OPS, types=ALL_BUILTIN_TYPES)
    value = builder.value("input", _test_ptr_register_type())
    builder.low.call(callee="native", operands=([(value, OUT)], []), results=[_test_ptr_register_type()], names=["result"])
    op = block.ops[-1]
    assert op.attributes["shape"] == [10, 7, 10, 0, 4, 3]
    assert _printer().print_operation(op, module) == ("%result = low.func.call @native([(%input, out)], []) : (reg<test.ptr>) -> (reg<test.ptr>)")


def test_flat_call_preserves_entire_operand_domain() -> None:
    count = 65535
    source = "low.func.call @ordinary(" + ", ".join(["%a"] * count) + ") : (" + ", ".join(["reg<test.ptr>"] * count) + ")"
    module, scope = _scope_with_values(("a", _test_ptr_register_type()))
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs.get("shape") is None
    assert text == source

    block = Block()
    _, builder = module_builder(insertion_block=block, ops=ALL_LOW_OPS, types=ALL_BUILTIN_TYPES)
    value = builder.value("input", _test_ptr_register_type())
    builder.low.call(callee="ordinary", operands=[value] * count, results=[])
    assert block.ops[-1].attributes.get("shape") is None
    assert len(block.ops[-1].operands) == count
    with pytest.raises(ValueError, match="shape domain"):
        builder.low.call(callee="native", operands=([value] * (count - 1),), results=[])


def test_optional_shape_still_uses_stored_count_domain() -> None:
    from loom.format.text.parser import ParseError

    source = "low.func.call @native(" + "[" * 65535 + "]" * 65535 + ") : ()"
    with pytest.raises(ParseError, match="shape domain"):
        _parser().parse_operation_from_text(source)
