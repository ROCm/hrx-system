# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generated C APIs for descriptor-backed parameterized attributes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.dsl import AttrDef, ParameterizedAttrDef
from loom.gen.ops.c_names import (
    c_parameterized_attr_api_prefix as _c_family_api_prefix,
)
from loom.gen.ops.c_names import (
    c_parameterized_attr_enum_name as _c_family_enum_name,
)
from loom.gen.ops.c_names import c_parameterized_attr_prefix as _c_family_prefix
from loom.gen.ops.c_parameter_values import (
    append_parameter_build_flag_declarations,
    append_parameter_slot_initializers,
    parameter_constructor_declarations,
    parameter_value_accessor_expr,
    parameter_value_c_type,
)


def _parameter_enum_type(
    families: Sequence[ParameterizedAttrDef],
    shared_enum_types: Mapping[int, str],
    family: ParameterizedAttrDef,
    parameter: AttrDef,
) -> str:
    enum_def = parameter.enum_def
    assert enum_def is not None
    if enum_def.c_type is not None:
        return enum_def.c_type
    shared_enum_type = shared_enum_types.get(id(enum_def))
    if shared_enum_type is not None:
        return shared_enum_type
    for candidate_family in families:
        for candidate in candidate_family.parameters:
            if candidate.enum_def is enum_def:
                return f"{_c_family_prefix(candidate_family)}_{candidate.name}_t"
    raise AssertionError(f"enum for {family.name}.{parameter.name} was not declared")


def _parameter_c_type(
    families: Sequence[ParameterizedAttrDef],
    shared_enum_types: Mapping[int, str],
    family: ParameterizedAttrDef,
    parameter: AttrDef,
) -> str:
    return parameter_value_c_type(
        parameter,
        lambda value: _parameter_enum_type(families, shared_enum_types, family, value),
    )


def _function_parameters(
    families: Sequence[ParameterizedAttrDef],
    shared_enum_types: Mapping[int, str],
    family: ParameterizedAttrDef,
) -> list[str]:
    return parameter_constructor_declarations(
        family.parameters,
        _c_family_api_prefix(family),
        lambda parameter: _parameter_c_type(families, shared_enum_types, family, parameter),
        "loom_attribute_t* out_attr",
    )


def generate_parameterized_attr_header_lines(
    families: Sequence[ParameterizedAttrDef],
    shared_enum_types: Mapping[int, str] | None = None,
) -> list[str]:
    """Generates family constructors and fixed-slot accessors."""
    shared_enum_types = shared_enum_types or {}
    lines: list[str] = []
    for family in families:
        prefix = _c_family_api_prefix(family)
        family_enum = _c_family_enum_name(family)
        if family.doc:
            lines.append(f"// {family.doc}")
        append_parameter_build_flag_declarations(lines, prefix, family.parameters)

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
            c_type = _parameter_c_type(families, shared_enum_types, family, parameter)
            accessor_expr = parameter_value_accessor_expr(parameter, slot_expr, c_type)
            lines.append(f"static inline {c_type} {parameter_prefix}(loom_attribute_t attr) {{")
            lines.append(f"  return {accessor_expr};")
            lines.append("}")

        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(families, shared_enum_types, family)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ");"
            lines.append(f"{declaration}{suffix}")
        lines.append("")
    return lines


def generate_parameterized_attr_source_lines(
    families: Sequence[ParameterizedAttrDef],
    shared_enum_types: Mapping[int, str] | None = None,
) -> list[str]:
    """Generates family constructor implementations."""
    shared_enum_types = shared_enum_types or {}
    lines: list[str] = []
    for family in families:
        prefix = _c_family_api_prefix(family)
        family_enum = _c_family_enum_name(family)
        lines.append(f"iree_status_t {prefix}_make(")
        function_parameters = _function_parameters(families, shared_enum_types, family)
        for index, declaration in enumerate(function_parameters):
            suffix = "," if index + 1 < len(function_parameters) else ") {"
            lines.append(f"{declaration}{suffix}")
        append_parameter_slot_initializers(lines, family.name, prefix, family.parameters)
        lines.append("  return loom_module_make_parameterized_attr(")
        lines.append(f"      module, {family_enum}, slots, {len(family.parameters)}, out_attr);")
        lines.append("}")
        lines.append("")
    return lines
