# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C builders.c source generation for Loom ops."""

from __future__ import annotations

from collections.abc import Sequence

from loom.assembly import StableKeyRef
from loom.dsl import Op
from loom.fields import FieldKind, compute_layout
from loom.gen.ops import c_builder_model, c_queries
from loom.gen.ops.c_enum_attrs import SharedEnumMap
from loom.gen.ops.c_enum_attrs import collect_shared_enums as _collect_shared_enums
from loom.gen.ops.c_names import COPYRIGHT
from loom.gen.ops.c_names import c_enum_name as _c_enum_name
from loom.gen.ops.c_names import c_prefix as _c_prefix
from loom.gen.support.generated_file import line_comment_header


def _emit_builder_count_check(
    lines: list[str],
    *,
    count: str,
    max_value: str,
    label: str,
) -> None:
    """Emits a C range check before narrowing a host-size builder count."""
    lines.append("  IREE_RETURN_IF_ERROR(loom_builder_check_count_range(")
    lines.append(f'      {count}, {max_value}, IREE_SV("{label}")));')


def _emit_builder_i64_array_storage(
    lines: list[str],
    *,
    source: str,
    count: str,
    storage: str,
    op_name: str,
    field_name: str,
) -> None:
    """Emits C code that copies an i64-array attr payload into the builder."""
    lines.append(f"  int64_t* {storage} = NULL;")
    lines.append("  IREE_RETURN_IF_ERROR(loom_builder_copy_i64_array_attr_storage(")
    lines.append(f'      builder, {source}, {count}, IREE_SV("{op_name} {field_name}"),')
    lines.append(f"      &{storage}));")


def _emit_builder_predicate_list_storage(
    lines: list[str],
    *,
    source: str,
    count: str,
    storage: str,
    op_name: str,
    field_name: str,
) -> None:
    """Emits C code that copies a predicate-list attr payload into the builder."""
    lines.append(f"  loom_predicate_t* {storage} = NULL;")
    lines.append("  IREE_RETURN_IF_ERROR(")
    lines.append("      loom_builder_copy_predicate_list_attr_storage(")
    lines.append(f'          builder, {source}, {count}, IREE_SV("{op_name} {field_name}"),')
    lines.append(f"          &{storage}));")


def _generate_builder_implementation(
    op: Op,
    prefix: str,
    enum_name: str,
    shared_enums: SharedEnumMap,
) -> list[str]:
    """Generates the C builder function implementation for a complex op."""
    params = c_builder_model.extract_c_params(op, shared_enums)
    layout = compute_layout(op)
    func_args_are_operands = c_queries.func_args_are_operands(op)
    params_by_name = {str(param["name"]): param for param in params if "name" in param}
    lines: list[str] = []

    # Compute counts for op_alloc.
    # Operands: fixed + variadic. For variadic, the count comes from a parameter.
    has_variadic_result = layout.variadic_result is not None

    # Figure out how to compute operand_count at runtime.
    fixed_operand_count = layout.fixed_operand_count
    variadic_operand_param = None
    for param in params:
        if param["kind"] in ("operand_variadic", "binding_list"):
            variadic_operand_param = param["name"]
            break
        if param["kind"] == "operand_dict":
            variadic_operand_param = param["name"]
            break
        if param["kind"] == "index_list":
            variadic_operand_param = param["dynamic_field"]
            break
        if param["kind"] == "func_args" and func_args_are_operands:
            variadic_operand_param = param["name"]
            break

    c_params = c_builder_model.build_c_param_list(params, layout, prefix)

    lines.append(f"iree_status_t {prefix}_build(")
    for i, p in enumerate(c_params):
        comma = "," if i < len(c_params) - 1 else ") {"
        lines.append(f"    {p}{comma}")

    # Validate all host-size counts before narrowing to the op storage width.
    if layout.segmented_operands:
        for param in params:
            if param["kind"] in (
                "operand_variadic",
                "binding_list",
                "operand_dict",
                "index_list",
            ) or (param["kind"] == "func_args" and func_args_are_operands):
                count_name = param["dynamic_field"] if param["kind"] == "index_list" else param["name"]
                _emit_builder_count_check(
                    lines,
                    count=f"{count_name}_count",
                    max_value="UINT16_MAX",
                    label=f"{op.name} {count_name} operand segment",
                )
    elif variadic_operand_param:
        max_variadic_operand_count = "UINT16_MAX"
        if fixed_operand_count:
            max_variadic_operand_count = f"UINT16_MAX - {fixed_operand_count}"
        _emit_builder_count_check(
            lines,
            count=f"{variadic_operand_param}_count",
            max_value=max_variadic_operand_count,
            label=f"{op.name} operand",
        )
    if has_variadic_result:
        _emit_builder_count_check(
            lines,
            count="result_count",
            max_value="UINT16_MAX",
            label=f"{op.name} result",
        )
    for param in params:
        if param["kind"] == "auto_region_table":
            _emit_builder_count_check(
                lines,
                count=f"{param['keys_field']}_count",
                max_value=f"UINT8_MAX - {layout.fixed_region_count}",
                label=f"{op.name} region",
            )
    if any(param["kind"] == "tied_results" for param in params):
        _emit_builder_count_check(
            lines,
            count="tied_result_count",
            max_value="UINT16_MAX",
            label=f"{op.name} tied result",
        )
    for param in params:
        if param["kind"] == "index_list":
            _emit_builder_count_check(
                lines,
                count=f"{param['static_field']}_count",
                max_value="UINT16_MAX",
                label=f"{op.name} static index",
            )
        elif param["kind"] == "predicate_list":
            _emit_builder_count_check(
                lines,
                count=f"{param['name']}_count",
                max_value="UINT16_MAX",
                label=f"{op.name} predicate",
            )
        elif param["kind"] == "block_args":
            _emit_builder_count_check(
                lines,
                count=f"{param['name']}_count",
                max_value="UINT16_MAX",
                label=f"{op.name} block argument",
            )
        elif param["kind"] == "attr" and param["attr_type"] == "i64_array":
            _emit_builder_count_check(
                lines,
                count=f"{param['name']}_count",
                max_value="UINT16_MAX",
                label=f"{op.name} i64 array attribute",
            )

    optional_operand_params = [param for param in params if param["kind"] == "operand" and param.get("optional")]
    if layout.segmented_operands:
        lines.append(f"  uint16_t operand_segment_counts[{len(op.operands)}] = {{0}};")
        lines.append("  uint32_t operand_count_32 = 0;")
        for operand in op.operands:
            desc = layout.fields[operand.name]
            operand_param = params_by_name.get(operand.name)
            if operand.variadic:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': variadic operand '{operand.name}' has no builder parameter")
                count_name = operand_param["dynamic_field"] if operand_param["kind"] == "index_list" else operand_param["name"]
                lines.append(f"  operand_segment_counts[{desc.index}] = (uint16_t){count_name}_count;")
                lines.append(f"  operand_count_32 += (uint32_t){count_name}_count;")
            elif operand.optional:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': optional operand '{operand.name}' has no builder parameter")
                optional_flag = c_builder_model.build_flag_bit_name(prefix, operand_param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    operand_segment_counts[{desc.index}] = 1;")
                lines.append("    operand_count_32 += 1;")
                lines.append("  }")
            else:
                lines.append(f"  operand_segment_counts[{desc.index}] = 1;")
                lines.append("  operand_count_32 += 1;")
        lines.append("  IREE_RETURN_IF_ERROR(loom_builder_check_count_range(")
        lines.append(f'      operand_count_32, UINT16_MAX, IREE_SV("{op.name} operand")));')
        lines.append("  uint16_t operand_count = (uint16_t)operand_count_32;")
        operand_count_expr = "operand_count"
    elif optional_operand_params:
        optional_operand_params.sort(key=lambda param: param["index"])
        lines.append(f"  uint16_t operand_count = {fixed_operand_count};")
        previous_optional_flag = ""
        for param in optional_operand_params:
            optional_flag = c_builder_model.build_flag_bit_name(prefix, param)
            if previous_optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag}) &&")
                lines.append(f"      !iree_any_bit_set(build_flags, {previous_optional_flag})) {{")
                lines.append("    return iree_make_status(")
                lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
                lines.append(f'        "{op.name} optional operand {param["name"]} requires preceding optional operand");')
                lines.append("  }")
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
            lines.append(f"    operand_count = {param['index'] + 1};")
            lines.append("  }")
            previous_optional_flag = optional_flag
        operand_count_expr = "operand_count"
    elif variadic_operand_param:
        operand_count_expr = f"{fixed_operand_count} + (uint16_t){variadic_operand_param}_count"
    else:
        operand_count_expr = str(fixed_operand_count)

    # Compute result count expression.
    if has_variadic_result:
        result_count_expr = "(uint16_t)result_count"
    else:
        result_count_expr = str(layout.fixed_result_count)

    successor_count_expr = str(len(op.successors))

    region_count_expr = str(len(op.regions))
    for param in params:
        if param["kind"] == "auto_region_table":
            region_count_expr = f"{layout.fixed_region_count} + (uint8_t){param['keys_field']}_count"
            break
    optional_auto_regions = [param for param in params if param["kind"] == "auto_region" and param.get("optional")]
    if optional_auto_regions:
        optional_auto_regions.sort(key=lambda param: param["region_index"])
        lines.append(f"  uint8_t region_count = {layout.required_region_count};")
        previous_optional_flag = ""
        for param in optional_auto_regions:
            optional_flag = c_builder_model.build_flag_bit_name(prefix, param)
            if previous_optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag}) &&")
                lines.append(f"      !iree_any_bit_set(build_flags, {previous_optional_flag})) {{")
                lines.append("    return iree_make_status(")
                lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
                lines.append(f'        "{op.name} optional region {param["name"]} requires preceding optional region");')
                lines.append("  }")
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
            lines.append(f"    region_count = {param['region_index'] + 1};")
            lines.append("  }")
            previous_optional_flag = optional_flag
        region_count_expr = "region_count"
    attr_count = len(c_queries.non_flags_attrs(op))

    # Compute tied result count expression.
    static_ties = c_builder_model.static_tied_results(op)
    has_tied_param = any(p["kind"] == "tied_results" for p in params)
    if static_ties:
        tied_count_expr = str(len(static_ties))
    elif has_tied_param:
        tied_count_expr = "(uint16_t)tied_result_count"
    else:
        tied_count_expr = "0"

    if layout.segmented_operands:
        allocate_fn = "loom_builder_allocate_segmented_op_with_successors" if op.successors else "loom_builder_allocate_segmented_op"
    else:
        allocate_fn = "loom_builder_allocate_op_with_successors" if op.successors else "loom_builder_allocate_op"
    lines.append(f"  IREE_RETURN_IF_ERROR({allocate_fn}(")
    lines.append(f"      builder, {enum_name}, {operand_count_expr},")
    if layout.segmented_operands:
        lines.append("      operand_segment_counts, IREE_ARRAYSIZE(operand_segment_counts),")
    if op.successors:
        lines.append(f"      {result_count_expr}, {successor_count_expr}, {region_count_expr}, {tied_count_expr},")
    else:
        lines.append(f"      {result_count_expr}, {region_count_expr}, {tied_count_expr},")
    lines.append(f"      {attr_count}, location, out_op));")
    lines.extend(f"  (*out_op)->instance_flags = {param['name']};" for param in params if param["kind"] == "instance_flags")

    if layout.segmented_operands:
        lines.append("  uint16_t operand_offset = 0;")
        for operand in op.operands:
            operand_param = params_by_name.get(operand.name)
            if operand.variadic:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': variadic operand '{operand.name}' has no builder parameter")
                if operand_param["kind"] in ("operand_variadic", "binding_list"):
                    name = operand_param["name"]
                    lines.append(f"  if ({name}_count > 0) {{")
                    lines.append("    memcpy(loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"           {name}, {name}_count * sizeof(loom_value_id_t));")
                    lines.append("  }")
                    lines.append(f"  operand_offset += (uint16_t){name}_count;")
                elif operand_param["kind"] == "operand_dict":
                    name = operand_param["name"]
                    attr_index = operand_param["names_attr_index"]
                    lines.append("  IREE_RETURN_IF_ERROR(loom_builder_set_operand_dict(")
                    lines.append(f"      builder, loom_make_named_value_slice({name}, {name}_count),")
                    lines.append("      loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"      &loom_op_attrs(*out_op)[{attr_index}]));")
                    lines.append(f"  operand_offset += (uint16_t){name}_count;")
                elif operand_param["kind"] == "index_list":
                    dyn = operand_param["dynamic_field"]
                    lines.append(f"  if ({dyn}_count > 0) {{")
                    lines.append("    memcpy(loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"           {dyn}, {dyn}_count * sizeof(loom_value_id_t));")
                    lines.append("  }")
                    lines.append(f"  operand_offset += (uint16_t){dyn}_count;")
                elif operand_param["kind"] == "func_args" and func_args_are_operands:
                    lines.append("  for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {")
                    lines.append("    loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                    lines.append("    IREE_RETURN_IF_ERROR(")
                    lines.append("        loom_builder_define_value(builder, arg_types[_i], &_arg_id));")
                    lines.append("    loom_op_operands(*out_op)[operand_offset + _i] = _arg_id;")
                    lines.append("  }")
                    lines.append("  operand_offset += (uint16_t)arg_types_count;")
            elif operand.optional:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': optional operand '{operand.name}' has no builder parameter")
                optional_flag = c_builder_model.build_flag_bit_name(prefix, operand_param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_operands(*out_op)[operand_offset++] = {operand.name};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_operands(*out_op)[operand_offset++] = {operand.name};")
    else:
        # Fill in fixed operands.
        for param in params:
            if param["kind"] != "operand":
                continue
            if param.get("optional"):
                optional_flag = c_builder_model.build_flag_bit_name(prefix, param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_operands(*out_op)[{param['index']}] = {param['name']};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_operands(*out_op)[{param['index']}] = {param['name']};")

        # Fill in variadic operands (memcpy from the array parameter).
        for param in params:
            if param["kind"] in ("operand_variadic", "binding_list"):
                name = param["name"]
                lines.append(f"  if ({name}_count > 0) {{")
                lines.append(f"    memcpy(loom_op_operands(*out_op) + {fixed_operand_count},")
                lines.append(f"           {name}, {name}_count * sizeof(loom_value_id_t));")
                lines.append("  }")
            elif param["kind"] == "operand_dict":
                name = param["name"]
                operand_index = param["operand_index"]
                attr_index = param["names_attr_index"]
                lines.append("  IREE_RETURN_IF_ERROR(loom_builder_set_operand_dict(")
                lines.append(f"      builder, loom_make_named_value_slice({name}, {name}_count),")
                lines.append(f"      loom_op_operands(*out_op) + {operand_index},")
                lines.append(f"      &loom_op_attrs(*out_op)[{attr_index}]));")
            elif param["kind"] == "index_list":
                dyn = param["dynamic_field"]
                lines.append(f"  if ({dyn}_count > 0) {{")
                lines.append(f"    memcpy(loom_op_operands(*out_op) + {fixed_operand_count},")
                lines.append(f"           {dyn}, {dyn}_count * sizeof(loom_value_id_t));")
                lines.append("  }")
            elif param["kind"] == "func_args" and func_args_are_operands:
                lines.append("  for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {")
                lines.append("    loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append("    IREE_RETURN_IF_ERROR(")
                lines.append("        loom_builder_define_value(builder, arg_types[_i], &_arg_id));")
                lines.append(f"    loom_op_operands(*out_op)[{fixed_operand_count} + _i] = _arg_id;")
                lines.append("  }")

    # Fill in fixed successors.
    lines.extend(f"  loom_op_successors(*out_op)[{param['index']}] = {param['name']};" for param in params if param["kind"] == "successor")

    def implicit_block_arg_type_expr(arg_type_kw: str) -> str:
        scalar_type = c_builder_model.IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
        if scalar_type is not None:
            return f"loom_type_scalar({scalar_type})"
        if arg_type_kw.startswith("type_of:"):
            source_field = arg_type_kw.removeprefix("type_of:")
            source_desc = layout.fields.get(source_field)
            if source_desc is None or source_desc.kind != FieldKind.OPERAND:
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be an operand field")
            if source_desc.variadic or source_desc.optional:
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be a required fixed operand")
            source_param = params_by_name.get(source_field)
            if source_param is None or source_param.get("kind") != "operand":
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be a fixed operand")
            return f"loom_module_value_type(builder->module, {source_field})"
        raise ValueError(f"Unknown implicit arg type: {arg_type_kw}")

    def emit_define_implicit_block_arg(indent: str, arg_type_kw: str) -> None:
        scalar_type = c_builder_model.IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
        lines.append(f"{indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
        lines.append(f"{indent}      builder, _block,")
        if scalar_type is not None:
            lines.append(f"{indent}      loom_type_scalar({scalar_type}), &_arg_id));")
        else:
            lines.append(f"{indent}      {implicit_block_arg_type_expr(arg_type_kw)}, &_arg_id));")
        lines.append(f"{indent}}}")

    def emit_auto_region(slot_expr: str, name: str, implicit_args: tuple[tuple[str, str], ...]) -> None:
        has_block_args = bool(implicit_args)
        lines.append(f"  // Auto-create {name} region with entry block.")
        lines.append("  {")
        if has_block_args:
            lines.append("    loom_block_t* _block = NULL;")
        lines.append("    IREE_RETURN_IF_ERROR(loom_builder_create_region(")
        lines.append(f"        builder, *out_op, {slot_expr}, {'&_block' if has_block_args else 'NULL'}));")
        for _arg_name, arg_type_kw in implicit_args:
            emit_define_implicit_block_arg("    ", arg_type_kw)
        lines.append("  }")

    # Auto-create regions with typed entry block args.
    for param in params:
        if param["kind"] != "auto_region":
            continue
        idx = param["region_index"]
        name = param["name"]
        optional_region = bool(param.get("optional"))
        region_indent = "    " if optional_region else "  "
        inner_indent = region_indent + "  "
        if optional_region:
            optional_flag = c_builder_model.build_flag_bit_name(prefix, param)
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
        has_block_args = bool(param.get("implicit_args")) or bool(param.get("binding")) or bool(param.get("arg_source")) or bool(param.get("block_args")) or bool(param.get("func_args"))
        lines.append(f"{region_indent}// Auto-create {name} region with entry block.")
        lines.append(f"{region_indent}{{")
        if has_block_args:
            lines.append(f"{inner_indent}loom_block_t* _block = NULL;")
        lines.append(f"{inner_indent}IREE_RETURN_IF_ERROR(loom_builder_create_region(")
        lines.append(f"{inner_indent}    builder, *out_op, {idx}, {'&_block' if has_block_args else 'NULL'}));")

        # Implicit args (e.g., loop IV).
        for _arg_name, arg_type_kw in param.get("implicit_args", ()):
            emit_define_implicit_block_arg(inner_indent, arg_type_kw)

        # Binding list args (capture or element).
        binding = param.get("binding")
        block_args = param.get("block_args")
        if block_args:
            lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {block_args}_count; ++_i) {{")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block, {block_args}[_i], &_arg_id));")
            lines.append(f"{inner_indent}}}")
        elif binding:
            binding_name = binding["name"]
            binding_kind = binding["binding_kind"]
            if binding_kind == "capture":
                lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {binding_name}_count; ++_i) {{")
                lines.append(f"{inner_indent}  loom_type_t _arg_type =")
                lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {binding_name}[_i]);")
                lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
                lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
                lines.append(f"{inner_indent}}}")
            elif binding_kind == "element":
                lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {binding_name}_count; ++_i) {{")
                lines.append(f"{inner_indent}  loom_type_t _operand_type =")
                lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {binding_name}[_i]);")
                lines.append(f"{inner_indent}  loom_type_t _arg_type =")
                lines.append(f"{inner_indent}      loom_type_scalar(loom_type_element_type(_operand_type));")
                lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
                lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
                lines.append(f"{inner_indent}}}")

        # Region args sourced from an existing value field. This is used for
        # regions whose entry args are semantically linked to a terminator or
        # loop edge, while the builder can still seed them from the op's input
        # values before the body is populated.
        arg_source = param.get("arg_source")
        if arg_source and not binding:
            lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {arg_source}_count; ++_i) {{")
            lines.append(f"{inner_indent}  loom_type_t _arg_type =")
            lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {arg_source}[_i]);")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
            lines.append(f"{inner_indent}}}")

        # FuncArgs: entry block args typed from the arg_types parameter.
        if param.get("func_args") and not binding:
            lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {{")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block, arg_types[_i], &_arg_id));")
            lines.append(f"{inner_indent}}}")

        lines.append(f"{region_indent}}}")
        if optional_region:
            lines.append("  }")

    for param in params:
        if param["kind"] != "auto_region_table":
            continue
        emit_auto_region(
            str(param["default_region_index"]),
            param["default_region_name"],
            param["default_implicit_args"],
        )
        lines.append(f"  for (iree_host_size_t _case = 0; _case < {param['keys_field']}_count; ++_case) {{")
        if param["case_implicit_args"]:
            lines.append("    loom_block_t* _block = NULL;")
        lines.append("    IREE_RETURN_IF_ERROR(loom_builder_create_region(")
        lines.append(f"        builder, *out_op, {param['case_region_index']} + _case,")
        lines.append(f"        {'&_block' if param['case_implicit_args'] else 'NULL'}));")
        if param["case_implicit_args"]:
            for _arg_name, arg_type_kw in param["case_implicit_args"]:
                emit_define_implicit_block_arg("    ", arg_type_kw)
        lines.append("  }")

    # Fill in attributes.
    for param in params:
        if param["kind"] == "predicate_list":
            idx = param["attr_index"]
            name = param["name"]
            storage = f"_{name}_storage"
            _emit_builder_predicate_list_storage(
                lines,
                source=name,
                count=f"{name}_count",
                storage=storage,
                op_name=op.name,
                field_name=name,
            )
            if param.get("optional"):
                lines.append(f"  if ({name}_count > 0) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list({storage}, (uint16_t){name}_count);")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list({storage}, (uint16_t){name}_count);")
        elif param["kind"] == "index_list":
            static_field = param["static_field"]
            idx = param["static_attr_index"]
            storage = f"_{static_field}_storage"
            _emit_builder_i64_array_storage(
                lines,
                source=static_field,
                count=f"{static_field}_count",
                storage=storage,
                op_name=op.name,
                field_name=static_field,
            )
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){static_field}_count);")
        elif param["kind"] == "symbol":
            idx = param["attr_index"]
            optional_flag = c_builder_model.build_flag_bit_name(prefix, param) if c_builder_model.optional_param_uses_build_flag(param) else ""
            if optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_symbol({param['name']});")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_symbol({param['name']});")
        elif param["kind"] == "attr":
            idx = param["attr_index"]
            attr_type = param["attr_type"]
            name = param["name"]
            is_optional = param.get("optional", False)
            constructor_map = {
                "i64": f"loom_attr_i64({name})",
                "f64": f"loom_attr_f64({name})",
                "string": f"loom_attr_string({name})",
                "bool": f"loom_attr_bool({name})",
                "enum": f"loom_attr_enum({name})",
                "symbol": f"loom_attr_symbol({name})",
                "type": f"loom_attr_type({name})",
                "encoding": f"loom_attr_encoding({name})",
                "any": name,
            }
            constructor = constructor_map.get(attr_type, name)
            optional_flag = c_builder_model.build_flag_bit_name(prefix, param) if c_builder_model.optional_param_uses_build_flag(param) else ""
            if attr_type == "i64_array":
                storage = f"_{name}_storage"
                _emit_builder_i64_array_storage(
                    lines,
                    source=name,
                    count=f"{name}_count",
                    storage=storage,
                    op_name=op.name,
                    field_name=name,
                )
                if is_optional:
                    lines.append(f"  if ({name}_count > 0) {{")
                    lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){name}_count);")
                    lines.append("  }")
                else:
                    lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){name}_count);")
            elif attr_type == "dict":
                if is_optional:
                    lines.append(f"  if ({name}.count > 0) {{")
                    lines.append("    IREE_RETURN_IF_ERROR(")
                    lines.append("        loom_module_make_canonical_attr_dict(")
                    lines.append(f"            builder->module, {name},")
                    lines.append(f"            &loom_op_attrs(*out_op)[{idx}]));")
                    lines.append("  }")
                else:
                    lines.append("  IREE_RETURN_IF_ERROR(")
                    lines.append("      loom_module_make_canonical_attr_dict(")
                    lines.append(f"          builder->module, {name},")
                    lines.append(f"          &loom_op_attrs(*out_op)[{idx}]));")
            elif optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = {constructor};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = {constructor};")
        elif param["kind"] == "descriptor_ref":
            idx = param["attr_index"]
            ordinal_idx = param["ordinal_attr_index"]
            name = param["name"]
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_string({name});")
            lines.append(f"  loom_op_attrs(*out_op)[{ordinal_idx}] = loom_attr_i64(-1);")
        elif param["kind"] == "stable_key_ref":
            idx = param["attr_index"]
            stable_id_idx = param["stable_id_attr_index"]
            name = param["name"]
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_string({name});")
            lines.append(f"  loom_op_attrs(*out_op)[{stable_id_idx}] = loom_attr_i64((int64_t)")
            lines.append("      loom_stable_id_from_string(")
            lines.append(f"          builder->module->strings.entries[{name}]));")

    # Define result values in the module's value table.
    for param in params:
        if param["kind"] == "result_type":
            lines.append("  IREE_RETURN_IF_ERROR(loom_builder_define_result(")
            lines.append("      builder, result_type, &loom_op_results(*out_op)[0]));")
        elif param["kind"] == "result_types":
            if has_variadic_result:
                lines.append("  IREE_RETURN_IF_ERROR(loom_builder_define_results(")
                lines.append("      builder, result_types, result_count, loom_op_results(*out_op)));")
            else:
                lines.append("  IREE_RETURN_IF_ERROR(loom_builder_define_result(")
                lines.append("      builder, result_type, &loom_op_results(*out_op)[0]));")

    # Populate tied result metadata.
    if static_ties:
        for tie_index, (result_idx, operand_idx) in enumerate(static_ties):
            lines.append(f"  loom_op_tied_results(*out_op)[{tie_index}] = (loom_tied_result_t){{.result_index = {result_idx}, .operand_index = {operand_idx}, .has_type_change = true}};")
    elif has_tied_param:
        lines.append("  IREE_RETURN_IF_ERROR(loom_builder_copy_tied_results(")
        lines.append("      tied_results, tied_result_count, *out_op));")

    lines.append("  return loom_builder_finalize_op(builder, *out_op);")
    lines.append("}")

    return lines


# ============================================================================
# builders.c generation
# ============================================================================


def generate_builders_c(dialect_name: str, ops: Sequence[Op], *, include_path: str | None = None) -> str:
    """Generates the builders.c file for a dialect."""
    lines: list[str] = []
    shared_enums = _collect_shared_enums(dialect_name, ops)

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/ops.h"')
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append('#include "loom/ir/module.h"')
    lines.append('#include "loom/ops/builder_macros.h"')
    if any(isinstance(element, StableKeyRef) for op in ops for element in c_builder_model.flatten_format(op.format)):
        lines.append('#include "loom/util/stable_id.h"')
    lines.append("")

    for op in ops:
        prefix = _c_prefix(op)
        enum_name = _c_enum_name(op)
        pattern = c_builder_model.detect_builder_pattern(op)

        if pattern == "BINARY":
            lines.append(f"LOOM_DEFINE_BINARY_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "BINARY_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_BINARY_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "UNARY":
            lines.append(f"LOOM_DEFINE_UNARY_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "UNARY_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_UNARY_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "CAST":
            lines.append(f"LOOM_DEFINE_CAST_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "COMPARISON":
            lines.append(f"LOOM_DEFINE_COMPARISON_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "COMPARISON_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_COMPARISON_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        else:
            lines.extend(_generate_builder_implementation(op, prefix, enum_name, shared_enums))

        lines.append("")

    return "\n".join(lines)
