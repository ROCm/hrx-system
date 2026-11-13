# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declaration contracts for the C type registry."""

from dataclasses import replace

import pytest

from loom.assembly import COMMA, EncodingOf, OptionalGroup, Param, ScalarOf, ShapeOf, kw
from loom.builtin_types import channel_type, encoding_type, ptr_type, read_type, ref_type, storage_type, view_type, write_type
from loom.dsl import EncodingParam, EnumCase, ScalarParam, ShapeParam, TypeDef
from loom.gen.ops.type_registry import generate_type_builders, generate_type_registry


def test_view_alignment_is_part_of_the_declared_format() -> None:
    _, _, source = generate_type_registry([view_type])
    assert "{LOOM_TYPE_FMT_ALIGNMENT, 3, 0}" in source
    assert "LOOM_KW_ALIGN" in source


def test_managed_handles_resolve_symbols_instead_of_allocating_parameter_slots() -> None:
    header, builders = generate_type_builders([ref_type, ptr_type])
    _, _, metadata = generate_type_registry([ref_type, ptr_type])
    for name, mode in (("ref", "OWNED"), ("ptr", "BORROWED")):
        assert f"loom_{name}_type_make(loom_module_t* module, loom_symbol_ref_t declaration" in header
        assert f"return loom_reference_type_make(module, LOOM_REFERENCE_OWNERSHIP_{mode}, declaration, out_type);" in builders
        assert f".reference_ownership = LOOM_REFERENCE_OWNERSHIP_{mode}," in metadata
    assert "LOOM_PARAMETERIZED_TYPE_REF" not in header
    assert "loom_module_make_parameterized_type_by_kind" not in builders


def test_managed_handle_registry_rejects_duplicate_modes() -> None:
    with pytest.raises(ValueError, match="duplicate registry names"):
        generate_type_registry([ptr_type, replace(ptr_type, name="other_ptr")])


@pytest.mark.parametrize("type_def", [channel_type, read_type, write_type])
def test_parameterized_channel_types_retain_their_fact_domain(type_def: TypeDef) -> None:
    _, _, source = generate_type_registry([type_def])
    descriptor = source.split(f"const loom_parameterized_type_descriptor_t loom_{type_def.name}_type_parameterized_descriptor = {{", 1)[1].split("};", 1)[0]
    assert ".fact_domain = &loom_view_fact_domain," in descriptor


def test_indirect_family_keys_follow_declaration_order_not_name_order() -> None:
    types = [view_type, read_type, channel_type, write_type]
    header, builders = generate_type_builders(types)
    _, _, source = generate_type_registry(types)
    for index, name in enumerate(("read", "channel", "write")):
        key = f"LOOM_PARAMETERIZED_TYPE_{name.upper()}"
        assert f"{key} = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_PARAMETERIZED_TYPE_COMMON_NAMESPACE, {index})" in header
        assert f"loom_type_parameterized_kind(type) == {key}" in header
        assert f".kind = {key}," in source
        assert f"loom_module_make_parameterized_type_by_kind(\n      module, {key}," in builders
    registry = source.split("loom_type_registry_parameterized_descriptors[] = {", 1)[1].split("};", 1)[0]
    assert registry.split() == [
        "&loom_read_type_parameterized_descriptor,",
        "&loom_channel_type_parameterized_descriptor,",
        "&loom_write_type_parameterized_descriptor,",
    ]
    assert "LOOM_PARAMETERIZED_TYPE_VIEW" not in header


def test_construction_outputs_are_independent_of_semantic_metadata() -> None:
    changed = replace(channel_type, fact_domain="loom_other_fact_domain", format=[kw("("), Param("payload"), kw(")")])
    assert generate_type_registry([channel_type]) != generate_type_registry([changed])
    assert generate_type_builders([channel_type]) == generate_type_builders([changed])
    header, source = generate_type_builders([channel_type])
    assert "descriptor" not in header + source
    assert "fact_domain" not in header + source
    assert "loom_channel_type_make(" not in generate_type_registry([channel_type])[2]
    changed = replace(channel_type, params=[replace(channel_type.params[0], name="element")], format=[Param("element")])
    assert generate_type_builders([channel_type]) != generate_type_builders([changed])


@pytest.mark.parametrize("type_def", [encoding_type, storage_type])
def test_compact_enum_spellings_have_one_construction_owned_table(type_def: TypeDef) -> None:
    header, source = generate_type_builders([type_def])
    _, _, metadata = generate_type_registry([type_def])
    prefix = f"loom_{type_def.name.replace('.', '_')}_type_{type_def.params[0].name}"
    table = f"{prefix}_enum_names"
    parameter = type_def.params[0]
    assert parameter.enum_def is not None
    last_value = max(case.value for case in parameter.enum_def.cases)
    assert f"extern const loom_bstring_t {table}[{last_value + 1}];" in header
    assert f"const loom_bstring_t {table}[] = {{" in source
    assert f".enum_case_names = {table}," in metadata
    assert f"const loom_bstring_t {table}" not in metadata
    assert "parameterized_descriptor" not in header + source

    first_value = min(case.value for case in parameter.enum_def.cases)
    enum_def = replace(parameter.enum_def, cases=[EnumCase("first", first_value), EnumCase("last", last_value)])
    changed = replace(type_def, params=[replace(parameter, enum_def=enum_def)])
    changed_header, changed_source = generate_type_builders([changed])
    assert header == changed_header
    assert source != changed_source
    assert 'LOOM_BSTRING_REF(5, "first"),\n' + "    NULL,\n" * (last_value - first_value - 1) + '    LOOM_BSTRING_REF(4, "last")' in changed_source


def _compact_tensor_type_def(name: str) -> TypeDef:
    return TypeDef(
        name=name,
        ir_kind="tensor",
        params=[
            ShapeParam("dims"),
            ScalarParam("element_type"),
            EncodingParam("encoding"),
        ],
        format=[
            ShapeOf("dims"),
            kw("x"),
            ScalarOf("element_type"),
            OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
        ],
    )


def test_generate_type_registry_rejects_duplicate_builtin_type_names() -> None:
    type_defs = [
        _compact_tensor_type_def("test.tensor"),
        _compact_tensor_type_def("test.other_tensor"),
    ]

    with pytest.raises(ValueError, match=r"Type kind 'tensor' has duplicate registry names"):
        generate_type_registry(type_defs)


def test_generate_type_registry_rejects_invalid_fact_domain_symbol() -> None:
    type_def = TypeDef(
        name="test.handle",
        fact_domain="loom.test.handle.fact_domain",
    )

    with pytest.raises(ValueError, match=r"TypeDef 'test\.handle': fact_domain must be a C symbol name"):
        generate_type_registry([type_def])
