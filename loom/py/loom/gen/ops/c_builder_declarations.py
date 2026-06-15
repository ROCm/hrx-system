# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C ops.h builder declaration generation for Loom ops."""

from __future__ import annotations

from loom.dsl import Op
from loom.fields import compute_layout
from loom.gen.ops import c_builder_model
from loom.gen.ops.c_enum_attrs import SharedEnumMap
from loom.gen.ops.c_names import c_prefix as _c_prefix


def _generate_build_flags_declaration(prefix: str, params: list[dict[str, object]]) -> list[str]:
    flag_params = c_builder_model.build_flag_params(params)
    if not flag_params:
        return []
    flag_count = len(flag_params)
    storage_type = c_builder_model.build_flags_storage_type(flag_count)
    bit_literal = c_builder_model.build_flag_bit_literal(flag_count)

    lines = [f"enum {prefix}_build_flag_bits_e {{"]
    for index, param in enumerate(flag_params):
        lines.append(f"  {c_builder_model.build_flag_bit_name(prefix, param)} = {bit_literal} << {index},")
    lines.append("};")
    lines.append(f"typedef {storage_type} {c_builder_model.build_flags_type_name(prefix)};")
    return lines


def _generate_builder_declaration(op: Op, prefix: str, shared_enums: SharedEnumMap) -> list[str]:
    """Generates the C builder function declaration for a complex op."""
    params = c_builder_model.extract_c_params(op, shared_enums)
    layout = compute_layout(op)
    lines: list[str] = []
    c_params = c_builder_model.build_c_param_list(params, layout, prefix)

    # Format as multi-line declaration.
    lines.append(f"iree_status_t {prefix}_build(")
    for i, p in enumerate(c_params):
        comma = "," if i < len(c_params) - 1 else ");"
        lines.append(f"    {p}{comma}")

    return lines


def generate_builder_header_lines(op: Op, shared_enums: SharedEnumMap) -> list[str]:
    """Generates build flag and builder function declarations for ops.h."""
    prefix = _c_prefix(op)
    params = c_builder_model.extract_c_params(op, shared_enums)
    lines = _generate_build_flags_declaration(prefix, params)
    pattern = c_builder_model.detect_builder_pattern(op)
    if pattern == "BINARY":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, loom_value_id_t lhs,")
        lines.append("    loom_value_id_t rhs, loom_type_t result_type,")
        lines.append("    loom_location_id_t location, loom_op_t** out_op);")
    elif pattern == "BINARY_WITH_FLAGS":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
        lines.append("    loom_value_id_t lhs, loom_value_id_t rhs,")
        lines.append("    loom_type_t result_type, loom_location_id_t location,")
        lines.append("    loom_op_t** out_op);")
    elif pattern == "UNARY":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, loom_value_id_t input,")
        lines.append("    loom_type_t result_type, loom_location_id_t location,")
        lines.append("    loom_op_t** out_op);")
    elif pattern == "UNARY_WITH_FLAGS":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
        lines.append("    loom_value_id_t input, loom_type_t result_type,")
        lines.append("    loom_location_id_t location, loom_op_t** out_op);")
    elif pattern == "CAST":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, loom_value_id_t input,")
        lines.append("    loom_type_t input_type, loom_type_t result_type,")
        lines.append("    loom_location_id_t location, loom_op_t** out_op);")
    elif pattern == "COMPARISON":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, uint8_t predicate,")
        lines.append("    loom_value_id_t lhs, loom_value_id_t rhs,")
        lines.append("    loom_type_t operand_type, loom_type_t result_type,")
        lines.append("    loom_location_id_t location, loom_op_t** out_op);")
    elif pattern == "COMPARISON_WITH_FLAGS":
        lines.append(f"iree_status_t {prefix}_build(")
        lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
        lines.append("    uint8_t predicate, loom_value_id_t lhs,")
        lines.append("    loom_value_id_t rhs, loom_type_t operand_type,")
        lines.append("    loom_type_t result_type, loom_location_id_t location,")
        lines.append("    loom_op_t** out_op);")
    else:
        lines.extend(_generate_builder_declaration(op, prefix, shared_enums))
    return lines
