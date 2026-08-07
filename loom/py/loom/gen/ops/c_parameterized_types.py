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
from loom.gen.ops.c_parameterized_attrs import (
    _parameter_accessor_expr,
    _parameter_attr_expr,
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
    if parameter.attr_type == "enum":
        return _parameter_enum_type(type_def, parameter)
    return {
        "i64": "int64_t",
        "f64": "double",
        "string": "loom_string_id_t",
        "bool": "bool",
        "enum_array": "loom_enum_array_t",
        "type": "loom_type_id_t",
        "i64_array": "loom_i64_array_t",
        "bytes": "iree_const_byte_span_t",
        "encoding": "uint16_t",
        "symbol": "loom_symbol_ref_t",
        "dict": "loom_named_attr_slice_t",
        "parameterized": "loom_attribute_t",
    }[parameter.attr_type]


def _optional_parameters(type_def: TypeDef) -> tuple[AttrDef, ...]:
    return tuple(parameter for parameter in type_def.params if parameter.optional)


def _build_flags_type(type_def: TypeDef) -> str:
    return f"{type_api_prefix(type_def)}_build_flags_t"


def _build_flag_name(type_def: TypeDef, parameter: AttrDef) -> str:
    return f"{type_api_prefix(type_def).upper()}_BUILD_FLAG_HAS_{parameter.name.upper()}"


def _function_parameters(type_def: TypeDef) -> list[str]:
    parameters = ["    loom_module_t* module"]
    if _optional_parameters(type_def):
        parameters.append(f"    {_build_flags_type(type_def)} build_flags")
    parameters.extend(f"    {_parameter_c_type(type_def, parameter)} {parameter.name}" for parameter in type_def.params)
    parameters.append("    loom_type_t* out_type")
    return parameters


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

        optional_parameters = _optional_parameters(type_def)
        if optional_parameters:
            if len(optional_parameters) <= 31:
                lines.append(f"enum {prefix}_build_flag_bits_e {{")
                for index, parameter in enumerate(optional_parameters):
                    lines.append(f"  {_build_flag_name(type_def, parameter)} = 1u << {index},")
                lines.append("};")
            else:
                for index, parameter in enumerate(optional_parameters):
                    lines.append(f"#define {_build_flag_name(type_def, parameter)} (UINT64_C(1) << {index})")
            storage_type = "uint32_t" if len(optional_parameters) <= 32 else "uint64_t"
            lines.append(f"typedef {storage_type} {_build_flags_type(type_def)};")
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
            accessor_expr = f"({c_type})loom_attr_as_enum({slot_expr})" if parameter.attr_type == "enum" else _parameter_accessor_expr(parameter, slot_expr)
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


def _emit_range_check(
    lines: list[str],
    type_def: TypeDef,
    parameter: AttrDef,
    condition: str,
    limit: str,
    indent: str,
) -> None:
    lines.append(f"{indent}if ({condition}) {{")
    lines.append(f"{indent}  return iree_make_status(")
    lines.append(f"{indent}      IREE_STATUS_RESOURCE_EXHAUSTED,")
    lines.append(f'{indent}      "{type_def.name} parameter {parameter.name} exceeds {limit}");')
    lines.append(f"{indent}}}")


def generate_source_lines(types: Sequence[TypeDef]) -> list[str]:
    """Generates typed constructor implementations."""

    lines: list[str] = []
    for type_def in parameterized_type_defs(types):
        prefix = type_api_prefix(type_def)
        optional_parameters = _optional_parameters(type_def)
        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(type_def)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ") {"
            lines.append(f"{declaration}{suffix}")
        if optional_parameters:
            known_flags = " | ".join(_build_flag_name(type_def, parameter) for parameter in optional_parameters)
            lines.append(f"  if (build_flags & ~({known_flags})) {{")
            lines.append("    return iree_make_status(")
            lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
            lines.append(f'        "{type_def.name} has unknown build flag bits");')
            lines.append("  }")
        lines.append(f"  loom_attribute_t slots[{len(type_def.params)}] = {{0}};")
        for index, parameter in enumerate(type_def.params):
            indent = "  "
            if parameter.optional:
                lines.append(f"  if (iree_any_bit_set(build_flags, {_build_flag_name(type_def, parameter)})) {{")
                indent = "    "
            if parameter.attr_type in ("enum_array", "i64_array", "dict"):
                _emit_range_check(
                    lines,
                    type_def,
                    parameter,
                    f"{parameter.name}.count > UINT16_MAX",
                    "UINT16_MAX elements",
                    indent,
                )
            elif parameter.attr_type == "bytes":
                _emit_range_check(
                    lines,
                    type_def,
                    parameter,
                    f"{parameter.name}.data_length > UINT32_MAX",
                    "UINT32_MAX bytes",
                    indent,
                )
            lines.append(f"{indent}slots[{index}] = {_parameter_attr_expr(parameter)};")
            if parameter.optional:
                lines.append("  }")
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
