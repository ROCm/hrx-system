# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generated C APIs for descriptor-backed parameterized attributes."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dsl import AttrDef, ParameterizedAttrDef
from loom.gen.ops.c_names import (
    c_parameterized_attr_api_prefix as _c_family_api_prefix,
)
from loom.gen.ops.c_names import (
    c_parameterized_attr_enum_name as _c_family_enum_name,
)
from loom.gen.ops.c_names import c_parameterized_attr_prefix as _c_family_prefix


def _parameter_enum_type(
    families: Sequence[ParameterizedAttrDef],
    family: ParameterizedAttrDef,
    parameter: AttrDef,
) -> str:
    enum_def = parameter.enum_def
    assert enum_def is not None
    if enum_def.c_type is not None:
        return enum_def.c_type
    for candidate_family in families:
        for candidate in candidate_family.parameters:
            if candidate.enum_def is enum_def:
                return f"{_c_family_prefix(candidate_family)}_{candidate.name}_t"
    raise AssertionError(f"enum for {family.name}.{parameter.name} was not declared")


def _parameter_c_type(
    families: Sequence[ParameterizedAttrDef],
    family: ParameterizedAttrDef,
    parameter: AttrDef,
) -> str:
    if parameter.attr_type == "enum":
        return _parameter_enum_type(families, family, parameter)
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


def _parameter_attr_expr(parameter: AttrDef) -> str:
    name = parameter.name
    return {
        "i64": f"loom_attr_i64({name})",
        "f64": f"loom_attr_f64({name})",
        "string": f"loom_attr_string({name})",
        "bool": f"loom_attr_bool({name})",
        "enum": f"loom_attr_enum((uint8_t){name})",
        "enum_array": (f"loom_attr_enum_array({name}.values, (uint16_t){name}.count)"),
        "type": f"loom_attr_type({name})",
        "i64_array": (f"loom_attr_i64_array((int64_t*){name}.values, (uint16_t){name}.count)"),
        "bytes": (f"loom_attr_bytes({name}.data, (uint32_t){name}.data_length)"),
        "encoding": f"loom_attr_encoding({name})",
        "symbol": f"loom_attr_symbol({name})",
        "dict": (f"loom_make_canonical_attr_dict({name}.entries, {name}.count)"),
        "parameterized": name,
    }[parameter.attr_type]


def _parameter_accessor_expr(parameter: AttrDef, slot_expr: str) -> str:
    return {
        "i64": f"loom_attr_as_i64({slot_expr})",
        "f64": f"loom_attr_as_f64({slot_expr})",
        "string": f"loom_attr_as_string_id({slot_expr})",
        "bool": f"loom_attr_as_bool({slot_expr})",
        "enum_array": f"loom_attr_as_enum_array({slot_expr})",
        "type": f"loom_attr_as_type_id({slot_expr})",
        "i64_array": f"loom_attr_as_i64_array({slot_expr})",
        "bytes": f"loom_attr_as_bytes({slot_expr})",
        "encoding": f"loom_attr_as_encoding_id({slot_expr})",
        "symbol": f"loom_attr_as_symbol({slot_expr})",
        "dict": f"loom_attr_as_dict({slot_expr})",
        "parameterized": slot_expr,
    }[parameter.attr_type]


def _optional_parameters(
    family: ParameterizedAttrDef,
) -> tuple[AttrDef, ...]:
    return tuple(parameter for parameter in family.parameters if parameter.optional)


def _build_flags_type(family: ParameterizedAttrDef) -> str:
    return f"{_c_family_api_prefix(family)}_build_flags_t"


def _build_flag_name(family: ParameterizedAttrDef, parameter: AttrDef) -> str:
    return f"{_c_family_api_prefix(family).upper()}_BUILD_FLAG_HAS_{parameter.name.upper()}"


def _function_parameters(families: Sequence[ParameterizedAttrDef], family: ParameterizedAttrDef) -> list[str]:
    parameters = ["    loom_module_t* module"]
    if _optional_parameters(family):
        parameters.append(f"    {_build_flags_type(family)} build_flags")
    parameters.extend(f"    {_parameter_c_type(families, family, parameter)} {parameter.name}" for parameter in family.parameters)
    parameters.append("    loom_attribute_t* out_attr")
    return parameters


def generate_parameterized_attr_header_lines(
    families: Sequence[ParameterizedAttrDef],
) -> list[str]:
    """Generates family constructors and fixed-slot accessors."""
    lines: list[str] = []
    for family in families:
        prefix = _c_family_api_prefix(family)
        family_enum = _c_family_enum_name(family)
        optional_parameters = _optional_parameters(family)
        if family.doc:
            lines.append(f"// {family.doc}")
        if optional_parameters:
            if len(optional_parameters) <= 31:
                lines.append(f"enum {prefix}_build_flag_bits_e {{")
                for index, parameter in enumerate(optional_parameters):
                    lines.append(f"  {_build_flag_name(family, parameter)} = 1u << {index},")
                lines.append("};")
            else:
                for index, parameter in enumerate(optional_parameters):
                    lines.append(f"#define {_build_flag_name(family, parameter)} (UINT64_C(1) << {index})")
            storage_type = "uint32_t" if len(optional_parameters) <= 32 else "uint64_t"
            lines.append(f"typedef {storage_type} {_build_flags_type(family)};")

        lines.append(f"static inline bool {prefix}_isa(loom_attribute_t attr) {{")
        lines.append(f"  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == {family_enum};")
        lines.append("}")
        for index, parameter in enumerate(family.parameters):
            parameter_prefix = f"{prefix}_{parameter.name}"
            parameter_index = f"{parameter_prefix.upper()}_PARAMETER_INDEX"
            lines.append(f"enum {{ {parameter_index} = {index} }};")
            slot_expr = f"loom_attr_as_parameterized_slots(attr)[{parameter_index}]"
            if parameter.optional:
                lines.append(f"static inline bool {prefix}_has_{parameter.name}(loom_attribute_t attr) {{")
                lines.append(f"  return !loom_attr_is_absent({slot_expr});")
                lines.append("}")
            c_type = _parameter_c_type(families, family, parameter)
            accessor_expr = f"({c_type})loom_attr_as_enum({slot_expr})" if parameter.attr_type == "enum" else _parameter_accessor_expr(parameter, slot_expr)
            lines.append(f"static inline {c_type} {parameter_prefix}(loom_attribute_t attr) {{")
            lines.append(f"  return {accessor_expr};")
            lines.append("}")

        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(families, family)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ");"
            lines.append(f"{declaration}{suffix}")
        lines.append("")
    return lines


def _emit_range_check(
    lines: list[str],
    family: ParameterizedAttrDef,
    parameter: AttrDef,
    condition: str,
    limit: str,
    indent: str,
) -> None:
    lines.append(f"{indent}if ({condition}) {{")
    lines.append(f"{indent}  return iree_make_status(")
    lines.append(f"{indent}      IREE_STATUS_RESOURCE_EXHAUSTED,")
    lines.append(f'{indent}      "{family.name} parameter {parameter.name} exceeds {limit}");')
    lines.append(f"{indent}}}")


def generate_parameterized_attr_source_lines(
    families: Sequence[ParameterizedAttrDef],
) -> list[str]:
    """Generates family constructor implementations."""
    lines: list[str] = []
    for family in families:
        prefix = _c_family_api_prefix(family)
        family_enum = _c_family_enum_name(family)
        optional_parameters = _optional_parameters(family)
        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(families, family)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ") {"
            lines.append(f"{declaration}{suffix}")
        if optional_parameters:
            known_flags = " | ".join(_build_flag_name(family, parameter) for parameter in optional_parameters)
            lines.append(f"  if (build_flags & ~({known_flags})) {{")
            lines.append("    return iree_make_status(")
            lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
            lines.append(f'        "{family.name} has unknown build flag bits");')
            lines.append("  }")
        if family.parameters:
            lines.append(f"  loom_attribute_t slots[{len(family.parameters)}] = {{0}};")
        else:
            lines.append("  const loom_attribute_t* slots = NULL;")
        for index, parameter in enumerate(family.parameters):
            indent = "  "
            if parameter.optional:
                lines.append(f"  if (iree_any_bit_set(build_flags, {_build_flag_name(family, parameter)})) {{")
                indent = "    "
            if parameter.attr_type in ("enum_array", "i64_array", "dict"):
                _emit_range_check(
                    lines,
                    family,
                    parameter,
                    f"{parameter.name}.count > UINT16_MAX",
                    "UINT16_MAX elements",
                    indent,
                )
            elif parameter.attr_type == "bytes":
                _emit_range_check(
                    lines,
                    family,
                    parameter,
                    f"{parameter.name}.data_length > UINT32_MAX",
                    "UINT32_MAX bytes",
                    indent,
                )
            lines.append(f"{indent}slots[{index}] = {_parameter_attr_expr(parameter)};")
            if parameter.optional:
                lines.append("  }")
        lines.append("  return loom_module_make_parameterized_attr(")
        lines.append(f"      module, {family_enum}, slots, {len(family.parameters)}, out_attr);")
        lines.append("}")
        lines.append("")
    return lines
