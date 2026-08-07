# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generated C APIs and metadata for descriptor-backed generic types."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dsl import AttrDef, TypeDef
from loom.gen.ops import c_symbols
from loom.gen.ops.c_enum_attrs import enum_case_c_ident
from loom.gen.ops.c_enums import ATTR_KIND_MAP
from loom.gen.ops.c_names import c_parameterized_attr_enum_name
from loom.gen.ops.c_parameter_values import (
    append_parameter_build_flag_declarations,
    append_parameter_slot_initializers,
    optional_parameters,
    parameter_constructor_declarations,
    parameter_value_accessor_expr,
    parameter_value_c_type,
)


def parameterized_type_defs(types: Sequence[TypeDef]) -> tuple[TypeDef, ...]:
    """Returns generic descriptor-backed type declarations in source order."""

    return tuple(type_def for type_def in types if type_def.uses_attribute_parameters)


def type_api_prefix(type_def: TypeDef) -> str:
    return f"loom_{type_def.name.replace('.', '_')}_type"


def type_descriptor_symbol(type_def: TypeDef) -> str:
    return f"{type_api_prefix(type_def)}_parameterized_descriptor"


def _parameter_prefix(type_def: TypeDef, parameter: AttrDef) -> str:
    return f"{type_api_prefix(type_def)}_{parameter.name}"


def _parameter_enum_type(type_def: TypeDef, parameter: AttrDef) -> str:
    assert parameter.enum_def is not None
    if parameter.enum_def.c_type is not None:
        return parameter.enum_def.c_type
    return f"{_parameter_prefix(type_def, parameter)}_t"


def _parameter_c_type(type_def: TypeDef, parameter: AttrDef) -> str:
    return parameter_value_c_type(parameter, lambda value: _parameter_enum_type(type_def, value))


def _function_parameters(type_def: TypeDef) -> list[str]:
    return parameter_constructor_declarations(
        type_def.params,
        type_api_prefix(type_def),
        lambda parameter: _parameter_c_type(type_def, parameter),
        "loom_type_t* out_type",
    )


def generate_header_lines(types: Sequence[TypeDef]) -> list[str]:
    """Generates typed constructors and fixed-slot accessors."""

    lines: list[str] = []
    for type_def in parameterized_type_defs(types):
        prefix = type_api_prefix(type_def)
        descriptor_symbol = type_descriptor_symbol(type_def)
        for parameter in type_def.params:
            enum_def = parameter.enum_def
            if parameter.attr_type not in ("enum", "enum_array") or enum_def is None or enum_def.c_type is not None:
                continue
            enum_prefix = _parameter_prefix(type_def, parameter)
            enum_tag = f"{enum_prefix}_e"
            const_prefix = enum_prefix.upper()
            max_value = max(case.value for case in enum_def.cases)
            if enum_def.doc:
                lines.append(f"// {enum_def.doc}")
            if parameter.open_enum:
                lines.append(f"typedef uint8_t {enum_prefix}_t;")
                lines.append(f"typedef enum {enum_tag} {{")
            else:
                lines.append(f"typedef enum {enum_tag} {{")
            lines.extend(f"  {const_prefix}_{enum_case_c_ident(case.keyword)} = {case.value}," for case in enum_def.cases)
            lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
            suffix = enum_tag if parameter.open_enum else f"{enum_prefix}_t"
            lines.append(f"}} {suffix};")
            lines.append("")

        optional = optional_parameters(type_def.params)
        if optional:
            append_parameter_build_flag_declarations(lines, prefix, type_def.params)
            lines.append("")

        lines.append(f"extern const loom_parameterized_type_descriptor_t {descriptor_symbol};")
        lines.append(f"static inline bool {prefix}_isa(loom_type_t type) {{")
        lines.append(f"  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &{descriptor_symbol};")
        lines.append("}")
        for index, parameter in enumerate(type_def.params):
            parameter_prefix = _parameter_prefix(type_def, parameter)
            parameter_index = f"{parameter_prefix.upper()}_PARAMETER_INDEX"
            lines.append(f"enum {{ {parameter_index} = {index} }};")
            slot_expr = f"loom_type_parameterized_parameters(type)[{parameter_index}]"
            if parameter.optional:
                lines.append(f"static inline bool {prefix}_has_{parameter.name}(loom_type_t type) {{")
                lines.append(f"  return !loom_attr_is_absent({slot_expr});")
                lines.append("}")
            c_type = _parameter_c_type(type_def, parameter)
            accessor_expr = parameter_value_accessor_expr(parameter, slot_expr, c_type)
            lines.append(f"static inline {c_type} {parameter_prefix}(loom_type_t type) {{")
            lines.append(f"  return {accessor_expr};")
            lines.append("}")
        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(type_def)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ");"
            lines.append(f"{declaration}{suffix}")
        lines.append("")
    return lines


def generate_source_lines(types: Sequence[TypeDef]) -> list[str]:
    """Generates typed constructor implementations."""

    lines: list[str] = []
    for type_def in parameterized_type_defs(types):
        prefix = type_api_prefix(type_def)
        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(type_def)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ") {"
            lines.append(f"{declaration}{suffix}")
        append_parameter_slot_initializers(lines, type_def.name, prefix, type_def.params)
        lines.append("  return loom_module_make_parameterized_type(")
        lines.append(f"      module, &{type_descriptor_symbol(type_def)}, slots, {len(type_def.params)}, out_type);")
        lines.append("}")
        lines.append("")
    return lines


def _bstring(value: str) -> str:
    value_length = len(value.encode())
    if value_length > 255:
        raise ValueError(f"B-string {value!r} exceeds 255 bytes")
    return f'LOOM_BSTRING_REF({value_length}, "{value}")'


def generate_metadata_lines(types: Sequence[TypeDef]) -> list[str]:
    """Generates static parameter schemas and family descriptors."""

    lines: list[str] = []
    for type_def in parameterized_type_defs(types):
        prefix = type_api_prefix(type_def)
        for parameter in type_def.params:
            if parameter.attr_type in ("enum", "enum_array"):
                assert parameter.enum_def is not None
                cases_by_value = {case.value: case.keyword for case in parameter.enum_def.cases}
                max_value = max(cases_by_value)
                lines.append(f"static const loom_bstring_t {prefix}_{parameter.name}_enum_names[] = {{")
                lines.extend(f"    {_bstring(keyword) if (keyword := cases_by_value.get(value)) is not None else 'NULL'}," for value in range(max_value + 1))
                lines.append("};")
            if parameter.symbol_ref is not None:
                flags = c_symbols.symbol_interface_flags(parameter.symbol_ref.interfaces)
                lines.append(f"static const loom_symbol_reference_descriptor_t {prefix}_{parameter.name}_symbol_ref = {{{_bstring(parameter.symbol_ref.name)}, {flags}}};")

        lines.append(f"static const loom_attr_descriptor_t {prefix}_parameter_desc[] = {{")
        for parameter in type_def.params:
            flags: list[str] = []
            if parameter.optional:
                flags.append("LOOM_ATTR_OPTIONAL")
            if parameter.elide_default:
                flags.append("LOOM_ATTR_ELIDE_DEFAULT")
            if parameter.open_enum:
                flags.append("LOOM_ATTR_OPEN_ENUM")
            lines.append("    {")
            lines.append(f"        .name = {_bstring(parameter.name)},")
            lines.append(f"        .attr_kind = {ATTR_KIND_MAP[parameter.attr_type]},")
            if flags:
                lines.append(f"        .flags = {' | '.join(flags)},")
            if parameter.attr_type in ("enum", "enum_array"):
                enum_names = f"{prefix}_{parameter.name}_enum_names"
                lines.append(f"        .enum_max_value = (uint8_t)(IREE_ARRAYSIZE({enum_names}) - 1),")
                lines.append(f"        .enum_case_names = {enum_names},")
            if parameter.symbol_ref is not None:
                lines.append(f"        .reference.symbol_ref = &{prefix}_{parameter.name}_symbol_ref,")
            if parameter.attr_type == "parameterized":
                expected_family = c_parameterized_attr_enum_name(parameter.parameterized_attr)
                lines.append(f"        .reference.parameterized_attr_kind = {expected_family},")
            lines.append("    },")
        lines.append("};")
        lines.append(f"const loom_parameterized_type_descriptor_t {type_descriptor_symbol(type_def)} = {{")
        lines.append(f"    .name = {_bstring(type_def.name)},")
        lines.append(f"    .parameter_count = IREE_ARRAYSIZE({prefix}_parameter_desc),")
        lines.append(f"    .parameter_descriptors = {prefix}_parameter_desc,")
        lines.append("};")
        lines.append("")
    return lines
