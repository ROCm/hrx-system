# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C generation for descriptor-backed parameter values."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loom.dsl import AttrDef


def parameter_value_c_type(parameter: AttrDef, enum_type: Callable[[AttrDef], str]) -> str:
    """Returns the generated C value type for a parameter."""

    if parameter.attr_type == "enum":
        return enum_type(parameter)
    return {
        "i64": "int64_t",
        "f64": "double",
        "string": "loom_string_id_t",
        "bool": "bool",
        "enum_array": "loom_enum_array_t",
        "signed_enum_set": "loom_signed_enum_set_t",
        "type": "loom_type_id_t",
        "i64_array": "loom_i64_array_t",
        "bytes": "iree_const_byte_span_t",
        "encoding": "uint16_t",
        "symbol": "loom_symbol_ref_t",
        "dict": "loom_named_attr_slice_t",
        "parameterized": "loom_attribute_t",
        "parameterized_array": "loom_parameterized_attr_array_t",
    }[parameter.attr_type]


def parameter_value_constructor_expr(parameter: AttrDef) -> str:
    """Returns an expression constructing a parameter's attribute value."""

    name = parameter.name
    return {
        "i64": f"loom_attr_i64({name})",
        "f64": f"loom_attr_f64({name})",
        "string": f"loom_attr_string({name})",
        "bool": f"loom_attr_bool({name})",
        "enum": f"loom_attr_enum((uint8_t){name})",
        "enum_array": f"loom_attr_enum_array({name}.values, (uint16_t){name}.count)",
        "signed_enum_set": f"loom_attr_signed_enum_set({name}.words, (uint16_t){name}.word_count)",
        "type": f"loom_attr_type({name})",
        "i64_array": f"loom_attr_i64_array((int64_t*){name}.values, (uint16_t){name}.count)",
        "bytes": f"loom_attr_bytes({name}.data, (uint32_t){name}.data_length)",
        "encoding": f"loom_attr_encoding({name})",
        "symbol": f"loom_attr_symbol({name})",
        "dict": f"loom_make_canonical_attr_dict({name}.entries, {name}.count)",
        "parameterized": name,
        "parameterized_array": f"loom_attr_parameterized_array({name}.values, {name}.count)",
    }[parameter.attr_type]


def parameter_value_accessor_expr(parameter: AttrDef, slot_expr: str, c_type: str) -> str:
    """Returns an expression extracting a parameter from an attribute slot."""

    return {
        "i64": f"loom_attr_as_i64({slot_expr})",
        "f64": f"loom_attr_as_f64({slot_expr})",
        "string": f"loom_attr_as_string_id({slot_expr})",
        "bool": f"loom_attr_as_bool({slot_expr})",
        "enum": f"({c_type})loom_attr_as_enum({slot_expr})",
        "enum_array": f"loom_attr_as_enum_array({slot_expr})",
        "signed_enum_set": f"loom_attr_as_signed_enum_set({slot_expr})",
        "type": f"loom_attr_as_type_id({slot_expr})",
        "i64_array": f"loom_attr_as_i64_array({slot_expr})",
        "bytes": f"loom_attr_as_bytes({slot_expr})",
        "encoding": f"loom_attr_as_encoding_id({slot_expr})",
        "symbol": f"loom_attr_as_symbol({slot_expr})",
        "dict": f"loom_attr_as_dict({slot_expr})",
        "parameterized": slot_expr,
        "parameterized_array": f"loom_attr_as_parameterized_array({slot_expr})",
    }[parameter.attr_type]


def optional_parameters(parameters: Sequence[AttrDef]) -> tuple[AttrDef, ...]:
    """Returns optional parameter declarations in source order."""

    return tuple(parameter for parameter in parameters if parameter.optional)


def parameter_build_flags_type(api_prefix: str) -> str:
    return f"{api_prefix}_build_flags_t"


def parameter_build_flag_name(api_prefix: str, parameter: AttrDef) -> str:
    return f"{api_prefix.upper()}_BUILD_FLAG_HAS_{parameter.name.upper()}"


def append_parameter_build_flag_declarations(lines: list[str], api_prefix: str, parameters: Sequence[AttrDef]) -> None:
    """Appends build flag declarations for optional parameters."""

    optional = optional_parameters(parameters)
    if not optional:
        return
    if len(optional) <= 31:
        lines.append(f"enum {api_prefix}_build_flag_bits_e {{")
        for index, parameter in enumerate(optional):
            lines.append(f"  {parameter_build_flag_name(api_prefix, parameter)} = 1u << {index},")
        lines.append("};")
    else:
        for index, parameter in enumerate(optional):
            lines.append(f"#define {parameter_build_flag_name(api_prefix, parameter)} (UINT64_C(1) << {index})")
    storage_type = "uint32_t" if len(optional) <= 32 else "uint64_t"
    lines.append(f"typedef {storage_type} {parameter_build_flags_type(api_prefix)};")


def parameter_constructor_declarations(
    parameters: Sequence[AttrDef],
    api_prefix: str,
    parameter_c_type: Callable[[AttrDef], str],
    result_declaration: str,
) -> list[str]:
    """Returns declarations for a generated parameterized constructor."""

    declarations = ["    loom_module_t* module"]
    if optional_parameters(parameters):
        declarations.append(f"    {parameter_build_flags_type(api_prefix)} build_flags")
    declarations.extend(f"    {parameter_c_type(parameter)} {parameter.name}" for parameter in parameters)
    declarations.append(f"    {result_declaration}")
    return declarations


def append_parameter_slot_initializers(
    lines: list[str],
    family_name: str,
    api_prefix: str,
    parameters: Sequence[AttrDef],
) -> None:
    """Appends constructor validation and fixed-slot value initialization."""

    optional = optional_parameters(parameters)
    if optional:
        known_flags = " | ".join(parameter_build_flag_name(api_prefix, parameter) for parameter in optional)
        lines.append(f"  if (build_flags & ~({known_flags})) {{")
        lines.append("    return iree_make_status(")
        lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
        lines.append(f'        "{family_name} has unknown build flag bits");')
        lines.append("  }")
    if parameters:
        lines.append(f"  loom_attribute_t slots[{len(parameters)}] = {{0}};")
    else:
        lines.append("  const loom_attribute_t* slots = NULL;")
    for index, parameter in enumerate(parameters):
        indent = "  "
        if parameter.optional:
            lines.append(f"  if (iree_any_bit_set(build_flags, {parameter_build_flag_name(api_prefix, parameter)})) {{")
            indent = "    "
        if parameter.attr_type in (
            "enum_array",
            "i64_array",
            "dict",
            "parameterized_array",
        ):
            _append_parameter_range_check(
                lines,
                family_name,
                parameter,
                f"{parameter.name}.count > UINT16_MAX",
                "UINT16_MAX elements",
                indent,
            )
        elif parameter.attr_type == "signed_enum_set":
            _append_parameter_range_check(
                lines,
                family_name,
                parameter,
                f"{parameter.name}.word_count > LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT",
                "LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT words per polarity",
                indent,
            )
        elif parameter.attr_type == "bytes":
            _append_parameter_range_check(
                lines,
                family_name,
                parameter,
                f"{parameter.name}.data_length > UINT32_MAX",
                "UINT32_MAX bytes",
                indent,
            )
        lines.append(f"{indent}slots[{index}] = {parameter_value_constructor_expr(parameter)};")
        if parameter.optional:
            lines.append("  }")


def _append_parameter_range_check(
    lines: list[str],
    family_name: str,
    parameter: AttrDef,
    condition: str,
    limit: str,
    indent: str,
) -> None:
    lines.append(f"{indent}if ({condition}) {{")
    lines.append(f"{indent}  return iree_make_status(")
    lines.append(f"{indent}      IREE_STATUS_RESOURCE_EXHAUSTED,")
    lines.append(f'{indent}      "{family_name} parameter {parameter.name} exceeds {limit}");')
    lines.append(f"{indent}}}")
