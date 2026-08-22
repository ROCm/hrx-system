# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C metadata table generation for Loom dialect ops."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loom.dsl import (
    ATTR_TYPE_BOOL,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_F64,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_PREDICATE_LIST,
    AttrDef,
    ConditionRefinementTruth,
    ContractFamily,
    EffectKind,
    EncodingFamilyDef,
    EnumDef,
    Op,
    OperandOwnershipEffect,
    ParameterizedAttrDef,
    ResultOwnershipEffect,
    TargetLikeInterface,
    TypeConstraint,
)
from loom.fields import FieldKind, compute_layout
from loom.gen.ops import c_format, c_interfaces, c_queries, c_symbols, c_traits
from loom.gen.ops.c_enum_attrs import (
    collect_encoding_auxiliary_key_enum as _collect_encoding_auxiliary_key_enum,
)
from loom.gen.ops.c_enum_attrs import collect_shared_enums as _collect_shared_enums
from loom.gen.ops.c_enum_attrs import enum_names_array_name as _enum_names_array_name
from loom.gen.ops.c_enums import (
    ATTR_KIND_MAP,
    CONSTRAINT_MAP,
    FIELD_CATEGORY_MAP,
    LOOM_FIELD_REF_MAX_INDEX,
    OPERAND_OWNERSHIP_EFFECT_MAP,
    OPERAND_ROLE_MAP,
    OPERAND_ROLE_MASK_MAP,
    OWNERSHIP_CARRIER_MAP,
    RESULT_OWNERSHIP_EFFECT_MAP,
    TYPE_CONSTRAINT_MAP,
)
from loom.gen.ops.c_enums import error_ref_literal as _error_ref_literal
from loom.gen.ops.c_names import COPYRIGHT
from loom.gen.ops.c_names import c_dialect_enum as _c_dialect_enum
from loom.gen.ops.c_names import c_encoding_enum_prefix as _c_encoding_enum_prefix
from loom.gen.ops.c_names import (
    c_encoding_family_descriptor_name as _c_encoding_family_descriptor_name,
)
from loom.gen.ops.c_names import (
    c_encoding_family_prefix as _c_encoding_family_prefix,
)
from loom.gen.ops.c_names import (
    c_parameterized_attr_enum_name as _c_parameterized_attr_enum_name,
)
from loom.gen.ops.c_names import (
    c_parameterized_attr_prefix as _c_parameterized_attr_prefix,
)
from loom.gen.ops.c_names import c_prefix as _c_prefix
from loom.gen.ops.c_names import (
    validate_encoding_family_c_names as _validate_encoding_family_c_names,
)
from loom.gen.support import c_arrays
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.c import c_string_literal as _c_string_literal
from loom.gen.support.generated_file import line_comment_header
from loom.stable_id import stable_id_from_string as _stable_id_from_string


def _contract_family_mask(contracts: Sequence[ContractFamily]) -> str:
    """Returns a stable C bitmask expression for contract families."""

    unique_contracts = set(contracts)
    if len(unique_contracts) != len(contracts):
        duplicate_names = sorted(family.name for family in contracts if contracts.count(family) > 1)
        raise ValueError(f"duplicate contract families in semantic metadata: {duplicate_names}")
    ordered_contracts = [family for family in ContractFamily if family in unique_contracts]
    if not ordered_contracts:
        return "0"
    return " | ".join(family.c_name for family in ordered_contracts)


def _op_phase_c_name(op: Op) -> str:
    """Returns the C phase enum for an op after applying its dialect default."""

    phase = op.effective_phase
    if phase is None:
        return "LOOM_OP_PHASE_UNSPECIFIED"
    return phase.c_name


def _op_semantics_row(op: Op, condition_refinement_index: int = 0) -> list[str]:
    """Returns a sparse initializer row for one op semantic metadata row."""
    contract_families = _contract_family_mask(op.contracts)
    row = [f".phase = {_op_phase_c_name(op)},"]
    if condition_refinement_index:
        row.append(f".condition_refinement_index = {condition_refinement_index},")
    if contract_families != "0":
        row.append(f".contract_families = {contract_families},")
    return row


def _condition_refinement_truth_flags(truth: ConditionRefinementTruth) -> str:
    """Returns the C truth-edge flags for one refinement declaration."""
    if truth is ConditionRefinementTruth.TRUE:
        return "LOOM_CONDITION_REFINEMENT_TRUTH_TRUE"
    if truth is ConditionRefinementTruth.FALSE:
        return "LOOM_CONDITION_REFINEMENT_TRUTH_FALSE"
    return "LOOM_CONDITION_REFINEMENT_TRUTH_TRUE | LOOM_CONDITION_REFINEMENT_TRUTH_FALSE"


def _emit_condition_refinement_table(lines: list[str], dialect_name: str, ops: Sequence[Op]) -> dict[str, int]:
    """Emits a sparse condition-refinement table and returns one-based indexes."""
    refinement_ops = [op for op in ops if op.condition_refinement is not None]
    if not refinement_ops:
        return {}
    rows: list[list[str]] = []
    indexes: dict[str, int] = {}
    for descriptor_index, op in enumerate(refinement_ops, start=1):
        refinement = op.condition_refinement
        assert refinement is not None
        source_operand_index = next(i for i, operand in enumerate(op.operands) if operand.name == refinement.source)
        indexes[op.name] = descriptor_index
        rows.append(
            [
                f".materialize = {refinement.materialize},",
                f".source_operand_index = {source_operand_index},",
                f".truth_flags = {_condition_refinement_truth_flags(refinement.truth)},",
            ]
        )
    c_arrays.append_struct_array(
        lines,
        "loom_condition_refinement_descriptor_t",
        f"loom_{dialect_name}_condition_refinement_array",
        rows,
    )
    return indexes


def _emit_dialect_table_accessors(
    lines: list[str],
    dialect_name: str,
    parameterized_attrs: Sequence[ParameterizedAttrDef] = (),
    has_condition_refinements: bool = False,
) -> None:
    """Emits dialect-specific wrappers around shared table helper algorithms."""

    vtable_array_name = f"loom_{dialect_name}_vtable_array"
    semantics_array_name = f"loom_{dialect_name}_semantics_array"
    lines.append(f"const loom_op_vtable_t* const* loom_{dialect_name}_dialect_vtables(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append("  return loom_dialect_vtable_array(")
    lines.append(f"      {vtable_array_name}, IREE_ARRAYSIZE({vtable_array_name}),")
    lines.append("      out_count);")
    lines.append("}")
    lines.append("")
    if has_condition_refinements:
        lines.append(f"const loom_condition_refinement_descriptor_t* loom_{dialect_name}_dialect_condition_refinements(")
        lines.append("    iree_host_size_t* out_count) {")
        lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_condition_refinement_array);")
        lines.append(f"  return loom_{dialect_name}_condition_refinement_array;")
        lines.append("}")
        lines.append("")
    if parameterized_attrs:
        lines.append(f"const loom_parameterized_attr_descriptor_t* loom_{dialect_name}_dialect_parameterized_attrs(")
        lines.append("    iree_host_size_t* out_count) {")
        lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_parameterized_attr_array);")
        lines.append(f"  return loom_{dialect_name}_parameterized_attr_array;")
        lines.append("}")
        lines.append("")
    lines.append(f"const loom_op_semantics_t* loom_{dialect_name}_dialect_op_semantics(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append("  return loom_dialect_semantics_array(")
    lines.append(f"      {semantics_array_name}, IREE_ARRAYSIZE({semantics_array_name}),")
    lines.append("      out_count);")
    lines.append("}")
    lines.append("")
    lines.append(f"loom_op_semantics_t loom_{dialect_name}_op_semantics(")
    lines.append("    loom_op_kind_t kind) {")
    lines.append("  return loom_dialect_semantics_lookup(")
    lines.append(f"      kind, {_c_dialect_enum(dialect_name)}, {semantics_array_name},")
    lines.append(f"      IREE_ARRAYSIZE({semantics_array_name}));")
    lines.append("}")
    lines.append("")


def _symbol_retain_attr_index(op: Op) -> int | None:
    """Returns the optional retain marker attribute index for a symbol op."""

    if op.symbol_def is None or op.symbol_def.retain is None:
        return None
    retain_attr_index = c_queries.resolve_attr_index(op, op.symbol_def.retain, "symbol_def.retain")
    retain_attr = op.attrs[retain_attr_index]
    if retain_attr.attr_type != ATTR_TYPE_ENUM:
        raise ValueError(f"Op {op.name!r}: symbol_def.retain {op.symbol_def.retain!r} must name an enum attr")
    return retain_attr_index


def _symbol_value_contract_indices(
    op: Op,
) -> tuple[int, int | None, int | None] | None:
    """Resolves a generated typed-value symbol contract."""

    if op.symbol_def is None or op.symbol_def.value_contract is None:
        return None
    contract = op.symbol_def.value_contract
    result_index = c_queries.resolve_result_index(op, contract.result, "symbol_def.value_contract")
    if op.results[result_index].variadic:
        raise ValueError(f"Op {op.name!r}: symbol value contract result {contract.result!r} must not be variadic")
    value_attr_index = c_queries.resolve_attr_index(op, contract.value, "symbol_def.value_contract") if contract.value is not None else None
    predicates_attr_index = c_queries.resolve_attr_index(op, contract.predicates, "symbol_def.value_contract") if contract.predicates is not None else None
    if predicates_attr_index is not None:
        attr_def = c_queries.non_flags_attrs(op)[predicates_attr_index]
        if attr_def.attr_type != ATTR_TYPE_PREDICATE_LIST:
            raise ValueError(f"Op {op.name!r}: symbol value contract predicates {contract.predicates!r} must name a predicate_list attr")
    return result_index, value_attr_index, predicates_attr_index


def _symbol_kernel_contract_indices(op: Op) -> tuple[int | None, int | None]:
    """Resolves a generated kernel workload-signature contract."""
    if op.symbol_def is None or op.symbol_def.kernel_contract is None:
        return None, None
    contract = op.symbol_def.kernel_contract
    region_index = c_queries.resolve_region_index(op, contract.workload_region, "symbol_def.kernel_contract") if contract.workload_region is not None else None
    operand_index = c_queries.resolve_operand_index(op, contract.workload_operands, "symbol_def.kernel_contract") if contract.workload_operands is not None else None
    if operand_index is not None and not op.operands[operand_index].variadic:
        raise ValueError(f"Op {op.name!r}: kernel workload signature operand {contract.workload_operands!r} must be variadic")
    return region_index, operand_index


def _symbol_kind(op: Op) -> str:
    """Returns the legacy C bytecode symbol kind constant for an op."""
    return op.symbol_def.bytecode_kind if op.symbol_def is not None else "LOOM_SYMBOL_NONE"


def _constraint_arg_ref(
    op: Op,
    constraint_name: str,
    arg_name: str,
    category: int,
    field_index: int,
) -> str:
    """Returns the LOOM_FIELD_REF(...) initializer for one constraint arg."""
    if field_index > LOOM_FIELD_REF_MAX_INDEX:
        raise ValueError(f"Op '{op.name}' constraint {constraint_name}: field '{arg_name}' index {field_index} exceeds LOOM_FIELD_REF 6-bit max {LOOM_FIELD_REF_MAX_INDEX}")
    return f"LOOM_FIELD_REF({category}, {field_index})"


# ============================================================================
# B-string encoding
# ============================================================================


def _bstring_expr(value: str) -> str:
    if len(value.encode()) > 255:
        raise ValueError(f"B-string '{value}' exceeds 255 bytes")
    return f'_BSTRING({len(value.encode())}, "{_c_string_literal(value)}")'


def _op_name_expr(value: str) -> str:
    value_length = len(value.encode())
    namespace_length = len(value.rsplit(".", 1)[0].encode()) if "." in value else 0
    if value_length > 255:
        raise ValueError(f"op name '{value}' exceeds 255 bytes")
    if namespace_length > 255:
        raise ValueError(f"op namespace '{value}' exceeds 255 bytes")
    return f'_OP_NAME({value_length}, {namespace_length}, "{_c_string_literal(value)}")'


def _emit_table_string_macros(lines: list[str], _dialect_name: str) -> None:
    lines.append("#define _BSTRING(length, value) LOOM_BSTRING_REF(length, value)")
    lines.append("#define _OP_NAME(length, namespace_length, value) \\")
    lines.append("  LOOM_OP_NAME_REF(length, namespace_length, value)")
    lines.append("")


def _emit_enum_case_names(lines: list[str], array_name: str, enum_def: EnumDef) -> None:
    cases_by_value = sorted(enum_def.cases, key=lambda case: case.value)
    max_value = max(case.value for case in cases_by_value)
    value_to_name = {case.value: case.keyword for case in cases_by_value}
    c_arrays.append_value_array(
        lines,
        "loom_bstring_t",
        array_name,
        [_bstring_expr(name) if (name := value_to_name.get(value)) is not None else "NULL" for value in range(max_value + 1)],
        trailing_blank=False,
    )


def _emit_parameterized_attr_tables(
    lines: list[str],
    dialect_name: str,
    parameterized_attrs: Sequence[ParameterizedAttrDef],
    shared_enum_names: dict[int, str] | None = None,
) -> None:
    """Emits dialect-owned parameter schemas and family descriptors."""

    for attr_def in parameterized_attrs:
        prefix = _c_parameterized_attr_prefix(attr_def)
        _emit_attr_descriptor_table(lines, prefix, attr_def.parameters, shared_enum_names)

    if parameterized_attrs:
        lines.append(f"static const loom_parameterized_attr_descriptor_t loom_{dialect_name}_parameterized_attr_array[] = {{")
        for attr_def in parameterized_attrs:
            prefix = _c_parameterized_attr_prefix(attr_def)
            lines.append("    {")
            lines.append(f"        .name = {_bstring_expr(attr_def.name)},")
            lines.append(f"        .kind = {_c_parameterized_attr_enum_name(attr_def)},")
            if attr_def.parameters:
                lines.append(f"        .parameter_count = IREE_ARRAYSIZE({prefix}_parameter_desc),")
            primary_parameter_index = attr_def.primary_parameter_index
            if primary_parameter_index is None:
                lines.append("        .primary_parameter_index = LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER,")
            else:
                lines.append(f"        .primary_parameter_index = {primary_parameter_index},")
            if attr_def.parameters:
                lines.append(f"        .parameter_descriptors = {prefix}_parameter_desc,")
            if attr_def.target_condition is not None:
                condition = c_symbols.normalize_c_symbol_reference(attr_def.target_condition)
                lines.append(f"        .target_condition = &{condition},")
            lines.append("    },")
        lines.append("};")
        lines.append("")


def _emit_attr_descriptor_table(
    lines: list[str],
    prefix: str,
    parameters: Sequence[AttrDef],
    shared_enum_names: dict[int, str] | None = None,
) -> str | None:
    """Emits one shared named-attribute descriptor table."""

    if not parameters:
        return None
    for parameter in parameters:
        if parameter.attr_type in ("enum", "enum_array", "signed_enum_set"):
            assert parameter.enum_def is not None
            if shared_enum_names is None or id(parameter.enum_def) not in shared_enum_names:
                _emit_enum_case_names(
                    lines,
                    f"{prefix}_{parameter.name}_enum_names",
                    parameter.enum_def,
                )
        if parameter.symbol_ref is not None:
            c_symbols.append_symbol_reference_descriptor(
                lines,
                f"{prefix}_{parameter.name}_symbol_ref",
                parameter.symbol_ref,
                _bstring_expr(parameter.symbol_ref.name),
            )

    table_name = f"{prefix}_parameter_desc"
    lines.append(f"static const loom_attr_descriptor_t {table_name}[] = {{")
    for parameter in parameters:
        attr_kind = ATTR_KIND_MAP[parameter.attr_type]
        flags_parts = []
        if parameter.optional:
            flags_parts.append("LOOM_ATTR_OPTIONAL")
        if parameter.elide_default:
            flags_parts.append("LOOM_ATTR_ELIDE_DEFAULT")
        if parameter.open_enum:
            flags_parts.append("LOOM_ATTR_OPEN_ENUM")
        if parameter.bare_identifier:
            flags_parts.append("LOOM_ATTR_BARE_IDENTIFIER")
        lines.append("    {")
        lines.append(f"        .name = {_bstring_expr(parameter.name)},")
        lines.append(f"        .attr_kind = {attr_kind},")
        if flags_parts:
            lines.append(f"        .flags = {' | '.join(flags_parts)},")
        if parameter.attr_type in ("enum", "enum_array", "signed_enum_set"):
            enum_names = shared_enum_names[id(parameter.enum_def)] if shared_enum_names is not None and id(parameter.enum_def) in shared_enum_names else f"{prefix}_{parameter.name}_enum_names"
            lines.append(f"        .enum_max_value = (uint8_t)(IREE_ARRAYSIZE({enum_names}) - 1),")
            lines.append(f"        .enum_case_names = {enum_names},")
        if parameter.symbol_ref is not None:
            lines.append(f"        .reference.symbol_ref = &{prefix}_{parameter.name}_symbol_ref,")
        if parameter.attr_type in ("parameterized", "parameterized_array"):
            expected_family = _c_parameterized_attr_enum_name(parameter.parameterized_attr) if parameter.parameterized_attr is not None else "LOOM_PARAMETERIZED_ATTR_KIND_ANY"
            lines.append(f"        .reference.parameterized_attr_kind = {expected_family},")
        lines.append("    },")
    lines.append("};")
    return table_name


def _emit_encoding_enum_case_names(
    lines: list[str],
    encoding_families: Sequence[EncodingFamilyDef],
) -> dict[int, str]:
    """Emits and returns enum name tables shared by encoding schemas."""
    shared_enum_names: dict[int, str] = {}
    for family in encoding_families:
        for parameter in family.parameters:
            enum_def = parameter.enum_def
            if parameter.attr_type not in ("enum", "enum_array", "signed_enum_set") or enum_def is None:
                continue
            enum_id = id(enum_def)
            if enum_id in shared_enum_names:
                continue
            array_name = f"{_c_encoding_enum_prefix(family.group.name, enum_def)}_names"
            _emit_enum_case_names(lines, array_name, enum_def)
            shared_enum_names[enum_id] = array_name
    return shared_enum_names


def _emit_encoding_auxiliary_key_descriptors(
    lines: list[str],
    dialect_name: str,
    encoding_families: Sequence[EncodingFamilyDef],
) -> None:
    """Emits the shared auxiliary-key vocabulary for encoding families."""
    enum_def = _collect_encoding_auxiliary_key_enum(encoding_families)
    if enum_def is None:
        return

    c_prefix = _c_encoding_enum_prefix(dialect_name, enum_def)
    cases_by_value = {case.value: case for case in enum_def.cases}
    descriptor_count = max(cases_by_value) + 1
    descriptor_count_expression = f"{c_prefix.upper()}_COUNT_" if enum_def.c_type is None else str(descriptor_count)
    lines.append(f"const loom_encoding_auxiliary_key_descriptor_t {c_prefix}_descriptors[{descriptor_count_expression}] = {{")
    for value in range(descriptor_count):
        case = cases_by_value.get(value)
        if case is None:
            lines.append("    {0},")
            continue
        stable_id = _stable_id_from_string(case.keyword)
        lines.append("    {")
        lines.append(f"        .name = {_bstring_expr(case.keyword)},")
        lines.append(f"        .stable_id = UINT64_C(0x{stable_id:016x}),")
        lines.append("    },")
    lines.append("};")
    lines.append("")


def _emit_encoding_family_fixed_metadata(lines: list[str], family: EncodingFamilyDef, prefix: str) -> str | None:
    """Emits family-wide constant encoding metadata when present."""
    summary = family.fixed_operand_summary
    record = family.fixed_record
    if summary is None and record is None and not family.required_auxiliary_keys:
        return None

    table_name = f"{prefix}_fixed_metadata"
    lines.append(f"static const loom_encoding_family_fixed_metadata_t {table_name} = {{")
    if summary is not None:
        lines.append("    .operand_summary = {")
        for field_name in (
            "element_format",
            "scale_format",
            "secondary_scale_format",
        ):
            value = getattr(summary, field_name)
            if value:
                lines.append(f"        .{field_name} = UINT64_C(0x{value:x}),")
        for field_name in (
            "payload_packing",
            "scale_topology",
            "affine_policy",
            "rounding_policy",
            "codebook_policy",
            "sparsity_policy",
        ):
            value = getattr(summary, field_name)
            if value:
                lines.append(f"        .{field_name} = UINT32_C(0x{value:x}),")
        if summary.zero_scale_fallback:
            lines.append("        .flags = UINT32_C(0x1),")
        if summary.sparsity_group_nonzero_element_count or summary.sparsity_group_element_count:
            lines.append("        .sparsity_group = {")
            if summary.sparsity_group_nonzero_element_count:
                lines.append(f"            .nonzero_element_count = {summary.sparsity_group_nonzero_element_count},")
            if summary.sparsity_group_element_count:
                lines.append(f"            .element_count = {summary.sparsity_group_element_count},")
            lines.append("        },")
        if summary.payload_register_count:
            lines.append(f"        .payload_register_count = {summary.payload_register_count},")
        if summary.payload_element_count:
            lines.append(f"        .payload_element_count = {summary.payload_element_count},")
        if summary.scale_group_element_count or summary.scale_group_shape:
            lines.append("        .scale_group = {")
            if summary.scale_group_element_count:
                lines.append(f"            .element_count = {summary.scale_group_element_count},")
            if summary.scale_group_shape:
                shape = ", ".join(str(value) for value in summary.scale_group_shape)
                lines.append(f"            .shape = {{{shape}}},")
            lines.append("        },")
        if summary.scale_operand_count:
            lines.append(f"        .scale_operand_count = {summary.scale_operand_count},")
        lines.append("    },")
    if family.required_auxiliary_keys:
        required_key_bits = sum(1 << key.value for key in family.required_auxiliary_keys)
        lines.append(f"    .required_auxiliary_keys = UINT64_C(0x{required_key_bits:x}),")
    if record is not None:
        lines.append("    .record = {")
        lines.append(f"        .logical_element_count = {record.logical_element_count},")
        lines.append(f"        .storage_byte_count = {record.storage_byte_count},")
        lines.append(f"        .required_alignment = {record.required_alignment},")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    return table_name


def _c_encoding_alias_value(parameter: AttrDef, value: Any) -> str:
    """Returns a static module-independent loom_attribute_t initializer."""

    if parameter.attr_type == ATTR_TYPE_I64:
        literal = f"INT64_C({value})" if value >= 0 else f"-INT64_C({-value})"
        return f"{{.kind = LOOM_ATTR_I64, .i64 = {literal}}}"
    if parameter.attr_type == ATTR_TYPE_F64:
        return f"{{.kind = LOOM_ATTR_F64, .f64 = {float(value)!r}}}"
    if parameter.attr_type == ATTR_TYPE_BOOL:
        return f"{{.kind = LOOM_ATTR_BOOL, .raw = {1 if value else 0}}}"
    if parameter.attr_type == ATTR_TYPE_ENUM:
        assert parameter.enum_def is not None
        return f"{{.kind = LOOM_ATTR_ENUM, .raw = {parameter.enum_def.case(value).value}}}"
    raise AssertionError(f"unsupported encoding alias parameter kind {parameter.attr_type!r}")


def _emit_encoding_family_aliases(lines: list[str], family: EncodingFamilyDef, prefix: str) -> str | None:
    """Emits canonical spelling rows for one structural encoding family."""

    if not family.aliases:
        return None
    parameters_by_name = {parameter.name: parameter for parameter in family.parameters}
    parameter_indexes = {parameter.name: index for index, parameter in enumerate(family.parameters)}
    alias_parameter_tables: list[str] = []
    for alias_index, alias in enumerate(family.aliases):
        table_name = f"{prefix}_alias_{alias_index}_parameters"
        alias_parameter_tables.append(table_name)
        lines.append(f"static const loom_encoding_alias_parameter_t {table_name}[] = {{")
        fixed_parameter_names = {parameter_name for parameter_name, _ in alias.fixed_parameters}
        for parameter_name, value in alias.parameters:
            parameter = parameters_by_name[parameter_name]
            lines.append("    {")
            lines.append(f"        .parameter_index = {parameter_indexes[parameter_name]},")
            if parameter_name in fixed_parameter_names:
                lines.append("        .flags = LOOM_ENCODING_ALIAS_PARAMETER_FIXED,")
            lines.append(f"        .value = {_c_encoding_alias_value(parameter, value)},")
            lines.append("    },")
        lines.append("};")
        lines.append("")

    table_name = f"{prefix}_aliases"
    lines.append(f"static const loom_encoding_alias_descriptor_t {table_name}[] = {{")
    for alias, parameter_table in zip(family.aliases, alias_parameter_tables, strict=True):
        lines.append("    {")
        lines.append(f"        .name = {_bstring_expr(alias.name)},")
        lines.append(f"        .parameter_count = IREE_ARRAYSIZE({parameter_table}),")
        lines.append(f"        .parameters = {parameter_table},")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    return table_name


def _emit_encoding_family_alias_discriminator(lines: list[str], family: EncodingFamilyDef, prefix: str) -> tuple[int, str] | None:
    """Emits the direct enum-value-to-alias lookup for one family."""

    discriminator = family.alias_discriminator
    if discriminator is None:
        return None
    assert discriminator.enum_def is not None
    parameter_index = family.parameters.index(discriminator)
    value_count = max(case.value for case in discriminator.enum_def.cases) + 1
    alias_ordinals = [0] * value_count
    for alias_index, alias in enumerate(family.aliases):
        discriminator_keyword = dict(alias.fixed_parameters)[discriminator.name]
        discriminator_value = discriminator.enum_def.case(discriminator_keyword).value
        alias_ordinals[discriminator_value] = alias_index + 1

    table_name = f"{prefix}_alias_ordinals"
    lines.append(f"static const uint8_t {table_name}[{value_count}] = {{")
    for value_index, alias_ordinal in enumerate(alias_ordinals):
        if alias_ordinal != 0:
            lines.append(f"    [{value_index}] = {alias_ordinal},")
    lines.append("};")
    lines.append("")
    return parameter_index, table_name


def _emit_encoding_family_tables(
    lines: list[str],
    encoding_families: Sequence[EncodingFamilyDef],
    shared_enum_names: dict[int, str],
) -> None:
    """Emits generated encoding-family descriptors."""

    _validate_encoding_family_c_names(encoding_families)

    for family in encoding_families:
        prefix = _c_encoding_family_prefix(family)
        parameter_table = _emit_attr_descriptor_table(lines, prefix, family.parameters, shared_enum_names)
        alias_table = _emit_encoding_family_aliases(lines, family, prefix)
        alias_discriminator = _emit_encoding_family_alias_discriminator(lines, family, prefix)
        fixed_metadata = _emit_encoding_family_fixed_metadata(lines, family, prefix)
        dynamic_parameter_table = None
        if family.dynamic_parameters:
            dynamic_parameter_table = f"{prefix}_dynamic_parameter_desc"
            lines.append(f"static const loom_encoding_dynamic_parameter_descriptor_t {dynamic_parameter_table}[] = {{")
            for parameter in family.dynamic_parameters:
                lines.append("    {")
                lines.append(f"        .name = {_bstring_expr(parameter.name)},")
                lines.append(f"        .type_constraint = {TYPE_CONSTRAINT_MAP[parameter.type_constraint]},")
                lines.append("    },")
            lines.append("};")
            lines.append("")
        lines.append(f"const loom_encoding_family_descriptor_t {_c_encoding_family_descriptor_name(family)} = {{")
        lines.append(f"    .name = {_bstring_expr(family.name)},")
        lines.append(f"    .role = {family.role.c_name},")
        if family.implicit_shaped_attachment:
            lines.append("    .family_flags = LOOM_ENCODING_FAMILY_IMPLICIT_SHAPED_ATTACHMENT,")
        if parameter_table is not None:
            lines.append(f"    .parameter_count = IREE_ARRAYSIZE({parameter_table}),")
            lines.append(f"    .parameter_descriptors = {parameter_table},")
        if alias_table is not None:
            lines.append(f"    .alias_count = IREE_ARRAYSIZE({alias_table}),")
            lines.append(f"    .aliases = {alias_table},")
        if alias_discriminator is not None:
            parameter_index, ordinal_table = alias_discriminator
            lines.append(f"    .alias_discriminator_parameter_index = {parameter_index},")
            lines.append(f"    .alias_ordinals_by_discriminator = {ordinal_table},")
        if dynamic_parameter_table is not None:
            lines.append(f"    .dynamic_parameter_count = IREE_ARRAYSIZE({dynamic_parameter_table}),")
            lines.append(f"    .dynamic_parameter_descriptors = {dynamic_parameter_table},")
        if fixed_metadata is not None:
            lines.append(f"    .fixed_metadata = &{fixed_metadata},")
        lines.append("};")
        lines.append("")


# ============================================================================
# tables.c generation
# ============================================================================


def generate_tables_c(
    dialect_name: str,
    dialect_id: int,
    ops: Sequence[Op],
    parameterized_attrs: Sequence[ParameterizedAttrDef] = (),
    encoding_families: Sequence[EncodingFamilyDef] = (),
    *,
    include_path: str | None = None,
    emit_registration: bool = True,
    export_vtables: bool = False,
    private_header: bool = False,
) -> str:
    """Generates the tables.c file for a dialect (.rodata)."""
    lines: list[str] = []
    ops_by_name = {op.name: op for op in ops}

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    if private_header:
        lines.append(f'#include "{include_path}/tables.h"')
    else:
        lines.append(f'#include "{include_path}/ops.h"')
    if c_interfaces.target_like_bundle_table_symbols(ops):
        lines.append("")
        lines.append("#include <stddef.h>")
        lines.append("")
        lines.append('#include "loom/ops/target/facts.h"')
    lines.append('#include "loom/error/error_defs.h"')
    lines.append("")
    if not private_header:
        _emit_table_string_macros(lines, dialect_name)

    # Canonicalize functions are declared in ops.h (not here) so there
    # are no extern declarations in .c files.
    shared_enums = _collect_shared_enums(dialect_name, ops)

    # Symbol definition descriptors may refer to fact domains outside this
    # generated translation unit.
    symbol_fact_domain_symbols = sorted({fact_domain for op in ops if op.symbol_def is not None if (fact_domain := c_symbols.symbol_fact_domain_symbol(op)) is not None})
    if symbol_fact_domain_symbols:
        lines.extend(f"extern const loom_symbol_fact_domain_t {fact_domain};" for fact_domain in symbol_fact_domain_symbols)
        lines.append("")
    interface_c_ptr_symbols = c_interfaces.interface_c_ptr_symbols(ops)
    if interface_c_ptr_symbols:
        lines.extend(f"extern const {c_type} {symbol};" for c_type, symbol in interface_c_ptr_symbols)
        lines.append("")
    target_like_bundle_table_symbols = c_interfaces.target_like_bundle_table_symbols(ops)
    if target_like_bundle_table_symbols:
        lines.extend(f"extern const loom_target_bundle_table_t {symbol};" for symbol in target_like_bundle_table_symbols)
        lines.append("")
    target_like_fact_type_symbols = c_interfaces.target_like_fact_type_symbols(ops)
    if target_like_fact_type_symbols:
        lines.extend(f"extern const loom_target_fact_type_t {symbol};" for symbol in target_like_fact_type_symbols)
        lines.append("")
    target_like_fact_projector_symbols = c_interfaces.target_like_fact_projector_symbols(ops)
    if target_like_fact_projector_symbols:
        lines.extend(f"extern const loom_target_fact_projector_t {symbol};" for symbol in target_like_fact_projector_symbols)
        lines.append("")

    emitted_enum_case_name_arrays: set[str] = set()

    # Op metadata blocks.
    for op in ops:
        prefix = _c_prefix(op)
        layout = compute_layout(op)
        elements = c_format.translate_format_elements(op)
        non_flags = c_queries.non_flags_attrs(op)
        has_flags = c_queries.has_flags_attr(op)

        # Format element array.
        if elements:
            lines.append(f"static const loom_format_element_t {prefix}_format[] = {{")
            for kind, field_index, data in elements:
                lines.append(f"    {{{kind}, {field_index}, {data}}},")
            lines.append("};")

        # Operand descriptors.
        if op.operands:
            effect_map = {effect.operand: effect.kind for effect in op.effects}
            ownership_operand_map = {effect.operand: effect for effect in op.ownership_effects if isinstance(effect, OperandOwnershipEffect)}
            lines.append(f"static const loom_operand_descriptor_t {prefix}_operand_desc[] = {{")
            for operand in op.operands:
                type_constraint = TYPE_CONSTRAINT_MAP[operand.type_constraint]
                flags_parts = []
                if operand.variadic:
                    flags_parts.append("LOOM_OPERAND_VARIADIC")
                if operand.optional:
                    flags_parts.append("LOOM_OPERAND_OPTIONAL")
                effect_kind = effect_map.get(operand.name)
                if effect_kind in (EffectKind.READ, EffectKind.READWRITE):
                    flags_parts.append("LOOM_OPERAND_READS")
                if effect_kind in (EffectKind.WRITE, EffectKind.READWRITE):
                    flags_parts.append("LOOM_OPERAND_WRITES")
                flags = " | ".join(flags_parts) if flags_parts else "0"
                ownership_effect = ownership_operand_map.get(operand.name)
                if ownership_effect is None:
                    ownership_effect_name = "LOOM_OPERAND_OWNERSHIP_NONE"
                    ownership_carrier_name = "LOOM_OWNERSHIP_CARRIER_NONE"
                else:
                    ownership_effect_name = OPERAND_OWNERSHIP_EFFECT_MAP[ownership_effect.kind]
                    ownership_carrier_name = OWNERSHIP_CARRIER_MAP[ownership_effect.carrier]
                role_name = OPERAND_ROLE_MAP[operand.role]
                if ownership_effect is None and role_name == "LOOM_OPERAND_ROLE_NONE":
                    lines.append(f"    {{{_bstring_expr(operand.name)}, {type_constraint}, {flags}}},")
                else:
                    lines.append(f"    {{{_bstring_expr(operand.name)}, {type_constraint}, {flags}, {ownership_effect_name}, {ownership_carrier_name}, {role_name}}},")
            lines.append("};")

        # Result descriptors.
        if op.results:
            ownership_result_map = {effect.result: effect for effect in op.ownership_effects if isinstance(effect, ResultOwnershipEffect)}
            lines.append(f"static const loom_result_descriptor_t {prefix}_result_desc[] = {{")
            for result in op.results:
                type_constraint = TYPE_CONSTRAINT_MAP[result.type_constraint]
                flags_parts = []
                if result.variadic:
                    flags_parts.append("LOOM_RESULT_VARIADIC")
                if result.allocates:
                    flags_parts.append("LOOM_RESULT_ALLOCATES")
                flags = " | ".join(flags_parts) if flags_parts else "0"
                result_ownership_effect = ownership_result_map.get(result.name)
                source_operand_index = "LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE"
                if result_ownership_effect is not None:
                    ownership_effect_name = RESULT_OWNERSHIP_EFFECT_MAP[result_ownership_effect.kind]
                    if result_ownership_effect.source is not None:
                        source_operand_index = str(c_queries.resolve_ownership_source_operand_index(op, result_ownership_effect.source))
                else:
                    ownership_effect_name = "LOOM_RESULT_OWNERSHIP_NONE"
                if result_ownership_effect is None:
                    lines.append(f"    {{{_bstring_expr(result.name)}, {type_constraint}, {flags}}},")
                else:
                    lines.append(f"    {{{_bstring_expr(result.name)}, {type_constraint}, {flags}, {ownership_effect_name}, {source_operand_index}}},")
            lines.append("};")

        # Enum case name arrays. Generated C may expose an external enum alias,
        # a dialect-level shared enum, or a per-op enum typedef, but all three
        # still need one parser/printer keyword table per C symbol name.
        for attr_def in op.attrs:
            if attr_def.attr_type in ("enum", "enum_array", "signed_enum_set") and attr_def.enum_def:
                array_name = _enum_names_array_name(op, attr_def, shared_enums)
                if array_name in emitted_enum_case_name_arrays:
                    continue
                _emit_enum_case_names(lines, array_name, attr_def.enum_def)
                emitted_enum_case_name_arrays.add(array_name)

        # Instance flags case name array.
        if has_flags:
            flags_attr = next(a for a in op.attrs if a.attr_type == ATTR_TYPE_FLAGS)
            assert flags_attr.enum_def is not None, f"flags attr on {op.name} has no enum_def"
            individual_cases = [c for c in flags_attr.enum_def.cases if c.value != 0 and (c.value & (c.value - 1)) == 0]
            individual_cases.sort(key=lambda c: c.value)
            array_name = f"{prefix}_instance_flags_names"
            c_arrays.append_value_array(
                lines,
                "loom_bstring_t",
                array_name,
                [_bstring_expr(case.keyword) for case in individual_cases],
                trailing_blank=False,
            )

        # Attribute symbol-reference descriptors.
        for attr_def in non_flags:
            if attr_def.symbol_ref is None:
                continue
            descriptor_name = f"{prefix}_{attr_def.name}_symbol_ref"
            c_symbols.append_symbol_reference_descriptor(
                lines,
                descriptor_name,
                attr_def.symbol_ref,
                _bstring_expr(attr_def.symbol_ref.name),
            )

        # Attribute descriptors.
        if non_flags:
            lines.append(f"static const loom_attr_descriptor_t {prefix}_attr_desc[] = {{")
            for attr_def in non_flags:
                if attr_def.attr_type not in ATTR_KIND_MAP:
                    raise ValueError(f"attr {attr_def.name!r} on {op.name!r} has unknown attr_type {attr_def.attr_type!r} with no C mapping")
                attr_kind = ATTR_KIND_MAP[attr_def.attr_type]
                flag_names = []
                if attr_def.optional:
                    flag_names.append("LOOM_ATTR_OPTIONAL")
                if attr_def.elide_default:
                    flag_names.append("LOOM_ATTR_ELIDE_DEFAULT")
                if attr_def.open_enum:
                    flag_names.append("LOOM_ATTR_OPEN_ENUM")
                flags = " | ".join(flag_names) if flag_names else "0"
                if attr_def.attr_type in ("enum", "enum_array", "signed_enum_set") and attr_def.enum_def:
                    enum_names = _enum_names_array_name(op, attr_def, shared_enums)
                    enum_max_value = f"(uint8_t)(IREE_ARRAYSIZE({enum_names}) - 1)"
                else:
                    enum_names = "NULL"
                    enum_max_value = "0"
                lines.append("    {")
                lines.append(f"        .name = {_bstring_expr(attr_def.name)},")
                lines.append(f"        .attr_kind = {attr_kind},")
                if flags != "0":
                    lines.append(f"        .flags = {flags},")
                if enum_names != "NULL":
                    lines.append(f"        .enum_max_value = {enum_max_value},")
                    lines.append(f"        .enum_case_names = {enum_names},")
                if attr_def.symbol_ref is not None:
                    lines.append(f"        .reference.symbol_ref = &{prefix}_{attr_def.name}_symbol_ref,")
                if attr_def.attr_type in ("parameterized", "parameterized_array"):
                    expected_family = _c_parameterized_attr_enum_name(attr_def.parameterized_attr) if attr_def.parameterized_attr is not None else "LOOM_PARAMETERIZED_ATTR_KIND_ANY"
                    lines.append(f"        .reference.parameterized_attr_kind = {expected_family},")
                lines.append("    },")
            lines.append("};")

        # Region descriptors.
        if op.regions:
            implicit_terminator = c_traits.implicit_terminator_kind(op, ops_by_name)
            lines.append(f"static const loom_region_descriptor_t {prefix}_region_desc[] = {{")
            func_args_fields = c_queries.func_args_field_names(op)
            for region_def in op.regions:
                region_flags = []
                if region_def.single_block:
                    region_flags.append("LOOM_REGION_SINGLE_BLOCK")
                if region_def.optional:
                    region_flags.append("LOOM_REGION_OPTIONAL")
                if region_def.arg_source in func_args_fields:
                    region_flags.append("LOOM_REGION_PROJECT_FUNC_ARGS")
                buffer_arg_memory_space = region_def.buffer_arg_memory_space
                if buffer_arg_memory_space is not None:
                    if buffer_arg_memory_space != "global":
                        raise ValueError(f"Op '{op.name}' region '{region_def.name}' has unsupported buffer_arg_memory_space '{buffer_arg_memory_space}'")
                    region_flags.append("LOOM_REGION_GLOBAL_BUFFER_ARGS")
                arg_uniform_scope = region_def.arg_uniform_scope
                if arg_uniform_scope is not None:
                    if arg_uniform_scope == "workgroup":
                        region_flags.append("LOOM_REGION_WORKGROUP_UNIFORM_ARGS")
                    elif arg_uniform_scope == "cluster":
                        region_flags.append("LOOM_REGION_CLUSTER_UNIFORM_ARGS")
                    else:
                        raise ValueError(f"Op '{op.name}' region '{region_def.name}' has unsupported arg_uniform_scope '{arg_uniform_scope}'")
                if region_def.command_effects_only:
                    region_flags.append("LOOM_REGION_COMMAND_EFFECTS_ONLY")
                flags = " | ".join(region_flags) if region_flags else "0"
                terminator = c_traits.region_terminator_kind(op, region_def, ops_by_name)
                lines.append(f"    {{{terminator}, {implicit_terminator}, {flags}}},")
            lines.append("};")

        # Constraint table.
        if op.constraints:
            lines.append(f"static const loom_constraint_t {prefix}_constraints[] = {{")
            for constraint in op.constraints:
                constraint_entry = CONSTRAINT_MAP.get(constraint.name)
                if constraint_entry is None:
                    raise ValueError(f"Op '{op.name}': unknown constraint '{constraint.name}'")
                relation_name, property_name = constraint_entry
                if property_name == "$data":
                    if constraint.data is None:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: missing data payload")
                    if not isinstance(constraint.data, int):
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload must be an integer")
                    if constraint.data < 0 or constraint.data > 255:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload out of uint8_t range")
                    property_name = str(constraint.data)
                elif property_name == "$type_constraint_data":
                    if not isinstance(constraint.data, TypeConstraint):
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload must be a TypeConstraint")
                    property_name = TYPE_CONSTRAINT_MAP[constraint.data]
                arg_refs: list[str] = []
                for arg_name in constraint.args:
                    field = layout.fields.get(arg_name)
                    if field is None:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: unknown field '{arg_name}'")
                    if field.kind == FieldKind.SUCCESSOR:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: successor field '{arg_name}' cannot be encoded as a value/type constraint argument")
                    category = FIELD_CATEGORY_MAP[field.kind]
                    arg_refs.append(_constraint_arg_ref(op, constraint.name, arg_name, category, field.index))
                while len(arg_refs) < 4:
                    arg_refs.append("0")
                args_str = ", ".join(arg_refs)
                error_ref = _error_ref_literal(constraint.error) if constraint.error is not None else "LOOM_ERROR_REF_NONE"
                lines.append(f"    {{{relation_name}, {property_name}, {len(constraint.args)}, 0, {{{args_str}}}, {error_ref}}},")
            lines.append("};")

        target_like_iface = c_queries.find_interface(op, TargetLikeInterface)
        if target_like_iface is not None:
            c_interfaces.emit_target_like_descriptor(op, target_like_iface, lines)

        # Interface vtables.
        for spec in c_interfaces.INTERFACES:
            c_interfaces.emit_interface_vtable(op, spec, lines)

        # Symbol definition descriptor.
        if op.symbol_def is not None:
            attr_index = c_queries.resolve_attr_index(op, op.symbol_def.field, "symbol_def")
            retain_attr_index = _symbol_retain_attr_index(op)
            value_contract_indices = _symbol_value_contract_indices(op)
            kernel_contract_indices = _symbol_kernel_contract_indices(op)
            flags = c_symbols.symbol_interface_flags(op.symbol_def.interfaces)
            fact_domain = c_symbols.symbol_fact_domain_symbol(op)
            lines.append(f"static const loom_symbol_definition_descriptor_t {prefix}_symbol_def = {{")
            lines.append(f"    .name = {_bstring_expr(op.symbol_def.name)},")
            if attr_index != 0:
                lines.append(f"    .name_attr_index = {attr_index},")
            if retain_attr_index is not None:
                lines.append(f"    .retain_attr_index_plus_one = {retain_attr_index + 1},")
            definition_flags: list[str] = []
            if op.symbol_def.is_declaration:
                definition_flags.append("LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION")
            if op.symbol_def.is_test_only:
                definition_flags.append("LOOM_SYMBOL_DEFINITION_FLAG_TEST_ONLY")
            if definition_flags:
                lines.append(f"    .flags = {' | '.join(definition_flags)},")
            if value_contract_indices is not None:
                result_index, value_attr_index, predicates_attr_index = value_contract_indices
                lines.append(f"    .value_contract_result_index_plus_one = {result_index + 1},")
                if value_attr_index is not None:
                    lines.append(f"    .value_contract_value_attr_index_plus_one = {value_attr_index + 1},")
                if predicates_attr_index is not None:
                    lines.append(f"    .value_contract_predicates_attr_index_plus_one = {predicates_attr_index + 1},")
            if flags != "0":
                lines.append(f"    .interfaces = {flags},")
            if op.symbol_def.bytecode_kind != "LOOM_SYMBOL_NONE":
                lines.append(f"    .bytecode_kind = {op.symbol_def.bytecode_kind},")
            kernel_region_index, kernel_operand_index = kernel_contract_indices
            if kernel_region_index is not None:
                lines.append(f"    .kernel_workload_region_index_plus_one = {kernel_region_index + 1},")
            if kernel_operand_index is not None:
                lines.append(f"    .kernel_workload_operand_field_index_plus_one = {kernel_operand_index + 1},")
            if fact_domain:
                lines.append(f"    .fact_domain = &{fact_domain},")
            lines.append("};")

        # Structural placement descriptor.
        required_parent_kinds = c_traits.trait_op_kinds(op, ops_by_name, "HasParent")
        required_ancestor_kinds = c_traits.trait_op_kinds(op, ops_by_name, "HasAncestor")
        forbidden_ancestor_kinds = c_traits.trait_op_kinds(op, ops_by_name, "NoAncestor")
        if required_parent_kinds or required_ancestor_kinds or forbidden_ancestor_kinds:
            required_parent_ptr = "NULL"
            required_ptr = "NULL"
            forbidden_ptr = "NULL"
            if required_parent_kinds:
                required_parent_ptr = f"{prefix}_required_parents"
                lines.append(f"static const loom_op_kind_t {required_parent_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in required_parent_kinds)
                lines.append("};")
            if required_ancestor_kinds:
                required_ptr = f"{prefix}_required_ancestors"
                lines.append(f"static const loom_op_kind_t {required_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in required_ancestor_kinds)
                lines.append("};")
            if forbidden_ancestor_kinds:
                forbidden_ptr = f"{prefix}_forbidden_ancestors"
                lines.append(f"static const loom_op_kind_t {forbidden_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in forbidden_ancestor_kinds)
                lines.append("};")
            lines.append(f"static const loom_op_placement_descriptor_t {prefix}_placement = {{")
            if required_parent_ptr != "NULL":
                lines.append(f"    .required_parents = {required_parent_ptr},")
                lines.append(f"    .required_parent_count = IREE_ARRAYSIZE({required_parent_ptr}),")
            if required_ptr != "NULL":
                lines.append(f"    .required_ancestors = {required_ptr},")
                lines.append(f"    .required_ancestor_count = IREE_ARRAYSIZE({required_ptr}),")
            if forbidden_ptr != "NULL":
                lines.append(f"    .forbidden_ancestors = {forbidden_ptr},")
                lines.append(f"    .forbidden_ancestor_count = IREE_ARRAYSIZE({forbidden_ptr}),")
            lines.append("};")

        # Vtable.
        traits = c_traits.trait_flags(op)
        vtable_flag_bits: list[str] = []
        if layout.segmented_operands:
            vtable_flag_bits.append("LOOM_OP_VTABLE_SEGMENTED_OPERANDS")
        elif layout.variadic_operand:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_OPERANDS")
        if layout.variadic_result:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_RESULTS")
        if layout.variadic_region:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_REGIONS")
        if has_flags:
            vtable_flag_bits.append("LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS")
        if c_queries.op_has_type_propagation_candidate(op, layout):
            vtable_flag_bits.append("LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE")
        if any(kind == "LOOM_FORMAT_KIND_OPERAND_DICT" for kind, _, _ in elements):
            vtable_flag_bits.append("LOOM_OP_VTABLE_HAS_OPERAND_DICT")
        if op.keyed_module_record_attr is not None:
            vtable_flag_bits.append("LOOM_OP_VTABLE_KEYED_MODULE_RECORD")
        vtable_flags_str = " | ".join(vtable_flag_bits) if vtable_flag_bits else "0"

        sym_kind = _symbol_kind(op)
        canon = op.canonicalize or "NULL"
        infer_facts_fn = op.facts or "NULL"
        type_transfer_fn = op.type_transfer or "NULL"
        verify_fn = op.verify or "NULL"
        eff_traits = op.effective_traits or "NULL"
        interface_ptrs = {spec.vtable_field: c_interfaces.interface_vtable_ptr(op, spec) for spec in c_interfaces.INTERFACES}
        symbol_def_ptr = f"&{prefix}_symbol_def" if op.symbol_def is not None else "NULL"
        has_placement = any(trait.name in ("HasParent", "HasAncestor", "NoAncestor") for trait in op.traits)
        placement_ptr = f"&{prefix}_placement" if has_placement else "NULL"
        attr_desc_ptr = f"{prefix}_attr_desc" if non_flags else "NULL"
        operand_desc_ptr = f"{prefix}_operand_desc" if op.operands else "NULL"
        operand_descriptor_count = len(op.operands)
        successor_selector_operand_index = c_queries.resolve_successor_selector_operand_index(op)
        implied_operand_descriptor_count = layout.fixed_operand_count
        if layout.segmented_operands:
            implied_operand_descriptor_count = -1
        elif layout.variadic_operand:
            implied_operand_descriptor_count += 1
        result_desc_ptr = f"{prefix}_result_desc" if op.results else "NULL"
        region_desc_ptr = f"{prefix}_region_desc" if op.regions else "NULL"
        constraint_ptr = f"{prefix}_constraints" if op.constraints else "NULL"
        fmt_ptr = f"{prefix}_format" if elements else "NULL"

        vtable_storage = "const" if export_vtables else "static const"
        lines.append(f"{vtable_storage} loom_op_vtable_t {prefix}_vtable = {{")

        def append_nonzero(field_name: str, value: int | str) -> None:
            if value != 0 and value != "0":
                lines.append(f"    .{field_name} = {value},")

        def append_nonnull(field_name: str, value: str) -> None:
            if value != "NULL":
                lines.append(f"    .{field_name} = {value},")

        append_nonzero("traits", traits)
        append_nonzero("fixed_operand_count", layout.fixed_operand_count)
        if operand_desc_ptr != "NULL" and operand_descriptor_count != implied_operand_descriptor_count:
            lines.append(f"    .operand_descriptor_count = IREE_ARRAYSIZE({operand_desc_ptr}),")
        append_nonzero("fixed_result_count", layout.fixed_result_count)
        append_nonzero("vtable_flags", vtable_flags_str)
        operand_role_mask_parts = [OPERAND_ROLE_MASK_MAP[operand.role] for operand in op.operands if operand.role in OPERAND_ROLE_MASK_MAP]
        if operand_role_mask_parts:
            lines.append(f"    .operand_role_mask = {' | '.join(sorted(set(operand_role_mask_parts)))},")
        if successor_selector_operand_index is not None:
            lines.append("    .control_flow_flags = LOOM_OP_CONTROL_FLOW_HAS_SUCCESSOR_SELECTOR,")
            lines.append(f"    .successor_selector_operand_index = {successor_selector_operand_index},")
        if sym_kind != "LOOM_SYMBOL_NONE":
            lines.append(f"    .symbol_kind = {sym_kind},")
        append_nonnull("canonicalize", canon)
        append_nonnull("infer_facts", infer_facts_fn)
        append_nonnull("effective_traits", eff_traits)
        append_nonnull("type_transfer", type_transfer_fn)
        append_nonnull("verify", verify_fn)
        lines.append(f"    .name = {_op_name_expr(op.name)},")
        if attr_desc_ptr != "NULL":
            lines.append(f"    .attr_descriptors = {attr_desc_ptr},")
            lines.append(f"    .attribute_count = IREE_ARRAYSIZE({attr_desc_ptr}),")
        append_nonnull("operand_descriptors", operand_desc_ptr)
        append_nonnull("result_descriptors", result_desc_ptr)
        if region_desc_ptr != "NULL":
            lines.append(f"    .region_descriptors = {region_desc_ptr},")
            lines.append(f"    .region_count = IREE_ARRAYSIZE({region_desc_ptr}),")
        if constraint_ptr != "NULL":
            lines.append(f"    .constraints = {constraint_ptr},")
            lines.append(f"    .constraint_count = IREE_ARRAYSIZE({constraint_ptr}),")
        append_nonnull("format_elements", fmt_ptr)
        if elements:
            lines.append(f"    .format_element_count = IREE_ARRAYSIZE({fmt_ptr}),")
        if has_flags:
            lines.append(f"    .instance_flags_case_names = {prefix}_instance_flags_names,")
            lines.append(f"    .instance_flags_case_count = IREE_ARRAYSIZE({prefix}_instance_flags_names),")
        if op.keyed_module_record_attr is not None:
            key_attr_index = c_queries.resolve_attr_index(
                op,
                op.keyed_module_record_attr,
                "KeyedModuleRecord",
            )
            lines.append(f"    .module_record_key_attr_index = {key_attr_index},")
        for spec in c_interfaces.INTERFACES:
            interface_ptr = interface_ptrs[spec.vtable_field]
            if interface_ptr != "NULL":
                lines.append(f"    .{spec.vtable_field} = {interface_ptr},")
        if symbol_def_ptr != "NULL":
            lines.append(f"    .symbol_def = {symbol_def_ptr},")
        if placement_ptr != "NULL":
            lines.append(f"    .placement = {placement_ptr},")

        lines.append("};")
        lines.append("")

    # Parameterized attribute families share the ordinary attribute schema but
    # retain a distinct dialect-owned outer identity.
    encoding_enum_names = _emit_encoding_enum_case_names(lines, encoding_families)
    _emit_encoding_auxiliary_key_descriptors(lines, dialect_name, encoding_families)
    _emit_parameterized_attr_tables(lines, dialect_name, parameterized_attrs, encoding_enum_names)
    _emit_encoding_family_tables(lines, encoding_families, encoding_enum_names)

    lines.append("#undef _OP_NAME")
    lines.append("#undef _BSTRING")
    lines.append("")

    if not emit_registration:
        return "\n".join(lines)

    # Registration function.
    c_arrays.append_value_array(
        lines,
        "loom_op_vtable_t* const",
        f"loom_{dialect_name}_vtable_array",
        [f"&{_c_prefix(op)}_vtable" for op in ops],
    )
    condition_refinement_indexes = _emit_condition_refinement_table(lines, dialect_name, ops)
    c_arrays.append_struct_array(
        lines,
        "loom_op_semantics_t",
        f"loom_{dialect_name}_semantics_array",
        [_op_semantics_row(op, condition_refinement_indexes.get(op.name, 0)) for op in ops],
    )
    _emit_dialect_table_accessors(
        lines,
        dialect_name,
        parameterized_attrs,
        has_condition_refinements=bool(condition_refinement_indexes),
    )

    return "\n".join(lines)


def generate_tables_h(dialect_name: str, ops: Sequence[Op], *, include_path: str | None = None) -> str:
    """Generates private declarations shared by a sharded dialect table."""
    guard = f"LOOM_OPS_{dialect_name.upper()}_TABLES_H_"
    lines: list[str] = []

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/ops.h"')
    lines.append("")
    _emit_table_string_macros(lines, dialect_name)
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.extend(f"extern const loom_op_vtable_t {_c_prefix(op)}_vtable;" for op in ops)
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('}  // extern "C"')
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")
    return "\n".join(lines)


def generate_tables_aggregator_c(
    dialect_name: str,
    dialect_id: int,
    ops: Sequence[Op],
    parameterized_attrs: Sequence[ParameterizedAttrDef] = (),
    encoding_families: Sequence[EncodingFamilyDef] = (),
    *,
    include_path: str | None = None,
) -> str:
    """Generates a dialect table aggregator for sharded per-op vtable files."""
    lines: list[str] = []

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/tables.h"')
    lines.append("")

    encoding_enum_names = _emit_encoding_enum_case_names(lines, encoding_families)
    _emit_encoding_auxiliary_key_descriptors(lines, dialect_name, encoding_families)
    _emit_parameterized_attr_tables(lines, dialect_name, parameterized_attrs, encoding_enum_names)
    _emit_encoding_family_tables(lines, encoding_families, encoding_enum_names)

    c_arrays.append_value_array(
        lines,
        "loom_op_vtable_t* const",
        f"loom_{dialect_name}_vtable_array",
        [f"&{_c_prefix(op)}_vtable" for op in ops],
    )
    condition_refinement_indexes = _emit_condition_refinement_table(lines, dialect_name, ops)
    c_arrays.append_struct_array(
        lines,
        "loom_op_semantics_t",
        f"loom_{dialect_name}_semantics_array",
        [_op_semantics_row(op, condition_refinement_indexes.get(op.name, 0)) for op in ops],
    )
    _emit_dialect_table_accessors(
        lines,
        dialect_name,
        parameterized_attrs,
        has_condition_refinements=bool(condition_refinement_indexes),
    )

    return "\n".join(lines)


def generate_sharded_tables_c(
    dialect_name: str,
    dialect_id: int,
    category_groups: Sequence[tuple[Any, Sequence[Op]]],
    parameterized_attrs: Sequence[ParameterizedAttrDef] = (),
    encoding_families: Sequence[EncodingFamilyDef] = (),
    *,
    include_path: str | None = None,
) -> dict[str, str]:
    """Generates an aggregator plus category shards for one dialect."""
    all_ops: list[Op] = []
    table_files: dict[str, str] = {}
    for category, category_ops in category_groups:
        shard_ops = list(category_ops)
        all_ops.extend(shard_ops)
        if not shard_ops:
            continue
        category_key = category.key
        filename = f"tables/{_c_identifier(category_key)}.c"
        table_files[filename] = generate_tables_c(
            dialect_name,
            dialect_id,
            shard_ops,
            include_path=include_path,
            emit_registration=False,
            export_vtables=True,
            private_header=True,
        )
    table_files["tables.c"] = generate_tables_aggregator_c(
        dialect_name,
        dialect_id,
        all_ops,
        parameterized_attrs,
        encoding_families,
        include_path=include_path,
    )
    table_files["tables.h"] = generate_tables_h(dialect_name, all_ops, include_path=include_path)
    return table_files
