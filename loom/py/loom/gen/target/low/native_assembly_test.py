# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from pathlib import Path

import pytest

from loom.gen.support.string_pool import CStringBank
from loom.gen.target.low.low_descriptors import compile_descriptor_set_family, descriptor_set_family_string_values, generate_descriptor_set_family
from loom.gen.target.low.native_assembly import NativeAsmValuePool
from loom.target.low_descriptors import NativeAsmValue, NativeAsmValueKind
from loom.target.test.descriptors import TEST_LOW_ADD_I32_DESCRIPTOR, TEST_LOW_CORE_DESCRIPTOR_SET


def _family(key: str, literal: str):
    descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    form = replace(
        descriptor.asm_forms[0],
        native_assembly_values=(
            NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="rhs"),
            NativeAsmValue(NativeAsmValueKind.LITERAL, literal=literal),
            NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
        ),
    )
    spec = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(replace(descriptor, key=key, asm_forms=(form,)),))
    view = replace(spec, key=spec.key + ".view", function_name=spec.function_name + "_view", descriptors=(replace(spec.descriptors[0], asm_forms=(replace(form, native_assembly_values=()),)),))
    compiled, _ = compile_descriptor_set_family(spec, (spec, view))
    return spec, view, compiled


def _pool(spans):
    return NativeAsmValuePool.build(spans, symbol="test_native_values", header="test/native_values.h")


def test_shared_recipes_preserve_descriptor_local_coordinates_and_literals(tmp_path: Path):
    families = (_family("test.first", "off"), _family("test.second", "off"), _family("test.third", "null"))
    spans = [compiled.asm_forms[0].native_assembly_values for _, _, compiled in families]
    assert spans[0] != spans[1]  # The literal has different provider-local labels.
    bank = _pool(reversed(spans))  # Force at least one provider's start to move.
    assert len(bank.values) == 6
    assert len(bank.starts) == 2
    manifest_path = tmp_path / "native_values.json"
    manifest_path.write_text(bank.manifest())
    assert NativeAsmValuePool.read(manifest_path) == bank

    primary = CStringBank.build(
        frozenset().union(*(descriptor_set_family_string_values(spec, (spec, view)) for spec, view, _ in families)),
        symbol="test_strings",
        header="test/strings.h",
    )
    for spec, view, compiled in families:
        storage = compiled.asm_table_storage
        remapped = bank.layouts_for(storage)
        for original, shared in zip(storage.layouts, remapped, strict=True):
            assert replace(shared, native_assembly_value_start=original.native_assembly_value_start) == original
            old_values = storage.native_values[original.native_assembly_value_start : original.native_assembly_value_start + original.native_assembly_value_count]
            new_values = bank.values[shared.native_assembly_value_start : shared.native_assembly_value_start + shared.native_assembly_value_count]
            assert new_values == tuple(replace(value, literal_label=None) for value in old_values)
        source = generate_descriptor_set_family(spec, (spec, view), primary_strings=primary, native_asm_values=bank).source
        assert "static const loom_low_native_asm_value_t" not in source
        assert source.count(".native_asm_values = test_native_values,") == 2
        assert source.count(".native_asm_value_count = 6,") == 2
    source = bank.emit_source(primary, None)
    assert source.count(".kind = LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL,") == 2


def test_recipe_keys_retain_every_interpreted_field():
    _, _, compiled = _family("test.add", "off")
    span = compiled.asm_forms[0].native_assembly_values
    value = span[0]
    variants = (value, replace(value, kind=NativeAsmValueKind.NEGATED_OPERAND), replace(value, index=1), replace(value, bit_width=16), replace(value, target_format_id=1))
    assert len(_pool((variant,) for variant in variants).starts) == len(variants)


def test_incomplete_shared_pool_rejects_the_provider():
    _, _, selected = _family("test.selected", "off")
    _, _, other = _family("test.other", "null")
    bank = _pool((selected.asm_forms[0].native_assembly_values,))
    with pytest.raises(ValueError, match="does not cover descriptor recipe"):
        bank.layouts_for(other.asm_table_storage)


def test_shared_pool_preserves_compact_span_limits():
    _, _, compiled = _family("test.add", "off")
    value = compiled.asm_forms[0].native_assembly_values[0]
    with pytest.raises(ValueError, match=r"recipe length.*u8"):
        _pool(((value,) * 256,))
    with pytest.raises(ValueError, match="u16-indexed table"):
        _pool((replace(value, index=index),) + (value,) * 254 for index in range(258))


def test_empty_shared_pool_has_no_runtime_rows():
    bank = _pool(((),))
    strings = CStringBank.build(frozenset(), symbol="test_strings", header="test/strings.h")
    assert not bank.values
    assert not bank.starts
    assert "enum {" not in bank.emit_source(strings, None)
