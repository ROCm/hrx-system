# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C row emitters for generated lower-rule tables."""

from __future__ import annotations

from collections.abc import Mapping

from loom.gen.support.c import c_i64_literal as _c_i64_literal
from loom.gen.support.string_pool import CStringPool
from loom.gen.target.contracts import lower_rule_spelling
from loom.target.contracts import (
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_RULE_FLAG_CONTRACT_ONLY,
    LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS,
    LOWER_SOURCE_MEMORY_NONE,
    CompiledLowerRuleSet,
    ContractFragment,
    GuardKind,
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerDiagnostic,
    LowerDiagnosticParam,
    LowerEmit,
    LowerEmitKind,
    LowerGuard,
    LowerRule,
    LowerRuleSpan,
    LowerSourceMemory,
    LowerTiedResult,
    LowerValueRef,
    TypePattern,
)
from loom.target.contracts.diagnostics import DiagnosticParamKind
from loom.target.low_descriptors import Descriptor

_GUARD_VALUE_REF_KINDS = frozenset(
    (
        GuardKind.VALUE_TYPE,
        GuardKind.VALUE_MATERIALIZABLE,
        GuardKind.LOW_VALUE_REGISTER_CLASS,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
        GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
        GuardKind.VALUE_EXACT_I64,
        GuardKind.VALUE_EXACT_POWER_OF_TWO_I64,
        GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD,
        GuardKind.VALUE_EXACT_FLOAT,
        GuardKind.VALUE_I64_RANGE,
        GuardKind.VALUE_I64_RANGE_LE,
        GuardKind.VALUE_I64_RANGE_GE,
        GuardKind.VALUE_FLOAT_EQUALS,
        GuardKind.VALUE_STORAGE_ELEMENT_FORMAT,
        GuardKind.VALUE_MEMORY_SPACE,
        GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
        GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
        GuardKind.VALUE_NO_USES,
        GuardKind.VECTOR_EXTRACT_SHAPE,
    )
)

_GUARD_OTHER_VALUE_REF_KINDS = frozenset(
    (
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
        GuardKind.VALUE_I64_RANGE_LE,
        GuardKind.VALUE_I64_RANGE_GE,
        GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
        GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
        GuardKind.VECTOR_EXTRACT_SHAPE,
    )
)

_ATTR_COPY_VALUE_REF_KINDS = frozenset(
    (
        LowerAttrCopyKind.VALUE_EXACT_I64,
        LowerAttrCopyKind.VALUE_EXACT_I64_NEGATE,
        LowerAttrCopyKind.VALUE_EXACT_I64_LOG2,
        LowerAttrCopyKind.VALUE_EXACT_I64_MINUS_ONE,
        LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_MULTIPLIER,
        LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_SHIFT,
        LowerAttrCopyKind.VALUE_I32_AS_U32_BITS,
        LowerAttrCopyKind.VALUE_FLOAT_AS_F16_BITS,
        LowerAttrCopyKind.VALUE_FLOAT_AS_BF16_BITS,
        LowerAttrCopyKind.VALUE_FLOAT_AS_F32_BITS,
        LowerAttrCopyKind.VALUE_FLOAT_AS_F64_BITS,
    )
)


def guard_uses_value_ref(kind: GuardKind) -> bool:
    return kind in _GUARD_VALUE_REF_KINDS


def guard_uses_other_value_ref(kind: GuardKind) -> bool:
    return kind in _GUARD_OTHER_VALUE_REF_KINDS


def attr_copy_uses_value_ref(kind: LowerAttrCopyKind) -> bool:
    return kind in _ATTR_COPY_VALUE_REF_KINDS


def diagnostic_has_implicit_target_context(row: LowerDiagnostic) -> bool:
    """Returns whether a diagnostic carries authored target context."""

    return row.target_context_param_count != 0


def diagnostic_stored_params(
    row: LowerDiagnostic,
) -> tuple[LowerDiagnosticParam, ...]:
    """Returns parameter rows that must be stored in the generated C table."""

    if not diagnostic_has_implicit_target_context(row):
        return row.params
    return row.params[row.target_context_param_count :]


def emit_optional_array(
    name: str,
    c_type: str,
    rows: list[list[str]],
) -> list[str]:
    if not rows:
        return []
    lines = [f"static const {c_type} {name}[] = {{"]
    for row in rows:
        lines.append("    {")
        lines.extend(f"        {field}," for field in row)
        lines.append("    },")
    lines.extend(["};", ""])
    return lines


def emit_optional_value_array(
    name: str,
    c_type: str,
    values: list[str],
) -> list[str]:
    if not values:
        return []
    lines = [f"static const {c_type} {name}[] = {{"]
    lines.extend(f"    {value}," for value in values)
    lines.extend(["};", ""])
    return lines


def _append_field(
    fields: list[str],
    name: str,
    value: str | int,
    *,
    always: bool = False,
    default: str = "0",
) -> None:
    value_string = str(value)
    if always or value_string != default:
        fields.append(f".{name} = {value_string}")


def value_ref_row(row: LowerValueRef) -> list[str]:
    fields: list[str] = []
    _append_field(
        fields,
        "kind",
        lower_rule_spelling.VALUE_REF_KIND_C_NAMES[row.kind],
        always=True,
    )
    _append_field(fields, "index", row.index, always=True)
    _append_field(fields, "element_index", row.element_index)
    _append_field(fields, "materializer_index", row.materializer_index)
    return fields


def source_memory_row(
    descriptor_refs: Mapping[str, int],
    row: LowerSourceMemory,
    *,
    byte_offset_immediate_string_offset: str | None = None,
    address_immediate_string_offset: str | None = None,
) -> list[str]:
    if row.byte_offset_materializer is not None and byte_offset_immediate_string_offset is None:
        raise ValueError("source-memory byte-offset materializer is missing its string offset")
    if row.address_materializer is not None and address_immediate_string_offset is None:
        raise ValueError("source-memory address materializer is missing its string offset")
    constraint = row.constraint
    fields: list[str] = []
    flags: list[str] = []
    if constraint.dynamic_byte_stride is None:
        flags.append("LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_BYTE_STRIDE_ANY")
    if constraint.allow_dynamic_stride_values:
        flags.append("LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_STRIDE_VALUES")
    if constraint.preserve_source_index:
        flags.append("LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_PRESERVE_SOURCE_INDEX")
    if flags:
        _append_field(fields, "flags", " | ".join(flags))
    _append_field(
        fields,
        "operation_kind",
        lower_rule_spelling.SOURCE_MEMORY_OPERATION_C_NAMES[constraint.operation],
        always=True,
    )
    _append_field(
        fields,
        "root_kind",
        lower_rule_spelling.SOURCE_MEMORY_ROOT_KIND_C_NAMES[constraint.root_kind],
        default="LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY",
    )
    _append_field(
        fields,
        "address_layout",
        lower_rule_spelling.SOURCE_MEMORY_ADDRESS_LAYOUT_C_NAMES[constraint.address_layout],
        default="LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_ANY",
    )
    _append_field(
        fields,
        "memory_space_mask",
        lower_rule_spelling.memory_space_mask(constraint.memory_spaces),
        always=True,
    )
    _append_field(
        fields,
        "element_byte_count",
        constraint.element_byte_count,
        always=True,
    )
    _append_field(
        fields,
        "vector_lane_count",
        constraint.vector_lane_count,
        always=True,
    )
    _append_field(
        fields,
        "vector_lane_byte_stride",
        _c_i64_literal(constraint.vector_lane_byte_stride),
        always=True,
    )
    _append_field(
        fields,
        "static_byte_offset_minimum",
        _c_i64_literal(constraint.static_byte_offset_minimum),
        always=True,
    )
    _append_field(
        fields,
        "static_byte_offset_maximum",
        _c_i64_literal(constraint.static_byte_offset_maximum),
        always=True,
    )
    _append_field(fields, "minimum_alignment", constraint.minimum_alignment)
    _append_field(
        fields,
        "dynamic_term_count",
        ("LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_TERM_COUNT_ANY" if constraint.dynamic_term_count is None else constraint.dynamic_term_count),
    )
    _append_field(
        fields,
        "dynamic_term_count_minimum",
        constraint.dynamic_term_count_minimum,
    )
    _append_field(
        fields,
        "dynamic_view_base_term_count",
        ("LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_VIEW_BASE_TERM_COUNT_ANY" if constraint.dynamic_view_base_term_count is None else constraint.dynamic_view_base_term_count),
        always=True,
    )
    _append_field(
        fields,
        "dynamic_index_source",
        lower_rule_spelling.SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_C_NAMES[constraint.dynamic_index_source],
        default="LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE",
    )
    if constraint.dynamic_byte_stride is not None:
        _append_field(
            fields,
            "dynamic_byte_stride",
            _c_i64_literal(constraint.dynamic_byte_stride),
        )
    _append_field(
        fields,
        "dynamic_offset_unsigned_bit_count",
        constraint.dynamic_offset_unsigned_bit_count,
    )
    _append_field(
        fields,
        "dynamic_offset_diagnostic_index",
        lower_rule_spelling.diagnostic_index(row.dynamic_offset_diagnostic_index),
        always=True,
    )
    _append_field(
        fields,
        "address_layout_diagnostic_index",
        lower_rule_spelling.diagnostic_index(row.address_layout_diagnostic_index),
        always=True,
    )
    _append_field(
        fields,
        "cache_policy_build_flags",
        constraint.cache_policy_build_flags,
    )
    _append_field(
        fields,
        "diagnostic_index",
        lower_rule_spelling.diagnostic_index(row.diagnostic_index),
        always=True,
    )
    _append_field(
        fields,
        "address_diagnostic_index",
        lower_rule_spelling.diagnostic_index(row.address_diagnostic_index),
        always=True,
    )
    if row.byte_offset_materializer is not None:
        materializer = row.byte_offset_materializer
        _append_field(
            fields,
            "byte_offset_const_i64_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.const_i64),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "byte_offset_const_i64_immediate_string_offset",
            byte_offset_immediate_string_offset,
            always=True,
        )
        _append_field(
            fields,
            "byte_offset_add_i64_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.add_i64),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "byte_offset_mul_i64_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.mul_i64),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "byte_offset_shl_i64_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.shl_i64),
            always=True,
            default="0xFFFF",
        )
    if row.address_materializer is not None:
        materializer = row.address_materializer
        _append_field(
            fields,
            "address_base_kind",
            lower_rule_spelling.SOURCE_MEMORY_ADDRESS_BASE_C_NAMES[materializer.base],
            always=True,
        )
        _append_field(
            fields,
            "address_coordinate_type",
            lower_rule_spelling.SOURCE_MEMORY_ADDRESS_COORDINATE_TYPE_C_NAMES[materializer.coordinate_type],
            always=True,
        )
        _append_field(
            fields,
            "address_coordinate_unit_byte_count",
            materializer.coordinate_unit_byte_count,
            always=True,
        )
        _append_field(
            fields,
            "address_coordinate_minimum",
            _c_i64_literal(materializer.coordinate_minimum),
            always=True,
        )
        _append_field(
            fields,
            "address_coordinate_maximum",
            _c_i64_literal(materializer.coordinate_maximum),
            always=True,
        )
        _append_field(
            fields,
            "address_const_coordinate_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.const_coordinate),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_const_coordinate_immediate_string_offset",
            address_immediate_string_offset,
            always=True,
        )
        _append_field(
            fields,
            "address_add_coordinate_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.add_coordinate),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_mul_coordinate_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.mul_coordinate),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_shl_coordinate_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.shl_coordinate),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_index_to_coordinate_input_descriptor_ref",
            _descriptor_ref_index(
                descriptor_refs,
                materializer.index_to_coordinate_input,
            ),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_index_to_coordinate_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.index_to_coordinate),
            always=True,
            default="0xFFFF",
        )
        _append_field(
            fields,
            "address_descriptor_ref",
            _descriptor_ref_index(descriptor_refs, materializer.address),
            always=True,
            default="0xFFFF",
        )
    return fields


def descriptor_ref_keys(table: CompiledLowerRuleSet, source_contract: ContractFragment) -> tuple[str, ...]:
    used_keys = {row.descriptor.key for row in table.guards if row.kind == GuardKind.DESCRIPTOR_AVAILABLE and row.descriptor is not None}
    used_keys.update(row.descriptor.key for row in table.emits)
    for row in table.source_memories:
        if row.byte_offset_materializer is not None:
            materializer = row.byte_offset_materializer
            used_keys.update(
                descriptor.key
                for descriptor in (
                    materializer.const_i64,
                    materializer.add_i64,
                    materializer.mul_i64,
                    materializer.shl_i64,
                )
                if descriptor is not None
            )
        if row.address_materializer is not None:
            materializer = row.address_materializer
            used_keys.update(
                descriptor.key
                for descriptor in (
                    materializer.const_coordinate,
                    materializer.add_coordinate,
                    materializer.mul_coordinate,
                    materializer.shl_coordinate,
                    materializer.index_to_coordinate_input,
                    materializer.index_to_coordinate,
                    materializer.address,
                )
                if descriptor is not None
            )
    return tuple(descriptor.key for descriptor in source_contract.descriptor_set.descriptors if descriptor.key in used_keys)


def descriptor_ref_row(key_string_offset: str) -> list[str]:
    return [f".key_string_offset = {key_string_offset}"]


def _descriptor_ref_index(descriptor_refs: Mapping[str, int], descriptor: Descriptor | None) -> int:
    if descriptor is None:
        return 0xFFFF
    return descriptor_refs[descriptor.key]


def _rule_flags_c_expression(flags: int) -> str:
    if flags == LOWER_RULE_FLAG_CONTRACT_ONLY:
        return "LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY"
    if flags == LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS:
        return "LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS"
    if flags == (LOWER_RULE_FLAG_CONTRACT_ONLY | LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS):
        return "LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY | LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS"
    return f"0x{flags:X}"


def guard_row(descriptor_refs: Mapping[str, int], row: LowerGuard) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "kind", lower_rule_spelling.GUARD_KIND_C_NAMES[row.kind], always=True)

    if guard_uses_value_ref(row.kind):
        _append_field(fields, "value_ref_index", row.value_ref_index, always=True)
    if guard_uses_other_value_ref(row.kind):
        _append_field(
            fields,
            "other_value_ref_index",
            row.other_value_ref_index,
            always=True,
        )

    if row.kind in (
        GuardKind.ATTR_KIND,
        GuardKind.ENUM_ATTR_EQUALS,
        GuardKind.I64_RANGE,
        GuardKind.OPERAND_SEGMENT_COUNT,
        GuardKind.I64_ARRAY_COUNT,
        GuardKind.I64_ARRAY_ELEMENT_RANGE,
        GuardKind.I64_ARRAY_ELEMENTS_RANGE,
        GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
        GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
        GuardKind.VECTOR_EXTRACT_SHAPE,
    ):
        _append_field(fields, "attr_index", row.attr_index, always=True)

    if row.kind == GuardKind.VALUE_TYPE:
        _append_field(
            fields,
            "index",
            f"{{.type_pattern_index = {row.type_pattern_index}}}",
            always=True,
        )
    elif row.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE:
        _append_field(
            fields,
            "index",
            f"{{.element_index = {row.u64}}}",
            always=True,
        )
    if row.diagnostic_index != 0xFFFF:
        _append_field(
            fields,
            "diagnostic_index",
            lower_rule_spelling.diagnostic_index(row.diagnostic_index),
            always=True,
        )
    if row.kind == GuardKind.ATTR_KIND:
        _append_field(fields, "attr_kind", lower_rule_spelling.attr_kind_c_name(row.attr_kind), always=True)
    u64_payload: str | None = None
    if row.kind in (
        GuardKind.ENUM_ATTR_EQUALS,
        GuardKind.OPERAND_SEGMENT_COUNT,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
        GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
        GuardKind.I64_ARRAY_COUNT,
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
        GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD,
        GuardKind.VALUE_FLOAT_EQUALS,
        GuardKind.INSTANCE_FLAGS_HAS_ALL,
        GuardKind.SOURCE_REPRESENTATION_GROUP,
    ):
        u64_payload = lower_rule_spelling.u64_c_literal(row.u64)
    elif row.kind == GuardKind.VALUE_STORAGE_ELEMENT_FORMAT:
        if row.u64_c_expression is None:
            raise ValueError("storage element-format guard is missing expression")
        u64_payload = row.u64_c_expression
    elif row.kind == GuardKind.VALUE_MEMORY_SPACE:
        u64_payload = lower_rule_spelling.memory_space_mask(row.memory_spaces)
    if u64_payload is not None:
        _append_field(
            fields,
            "payload",
            f"{{.u64 = {u64_payload}}}",
            always=True,
        )
    if row.kind == GuardKind.DESCRIPTOR_AVAILABLE:
        _append_field(
            fields,
            "descriptor_ref",
            _descriptor_ref_index(descriptor_refs, row.descriptor),
            always=True,
        )
    if row.kind == GuardKind.LOW_VALUE_REGISTER_CLASS:
        _append_field(fields, "register_class_id", row.register_class_id, always=True)
    if row.kind in (
        GuardKind.I64_RANGE,
        GuardKind.I64_ARRAY_ELEMENT_RANGE,
        GuardKind.I64_ARRAY_ELEMENTS_RANGE,
        GuardKind.VALUE_I64_RANGE,
    ):
        _append_field(
            fields,
            "payload",
            f"{{.i64_range = {{.minimum = {_c_i64_literal(row.minimum_i64)}, .maximum = {_c_i64_literal(row.maximum_i64)}}}}}",
            always=True,
        )
    elif row.kind in (
        GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
        GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
    ):
        _append_field(
            fields,
            "payload",
            f"{{.packed_integer = {{.storage_payload_multiple = UINT32_C({row.u64}), .storage_unit_bit_count = UINT32_C({row.minimum_i64}), .maximum_lane_count = UINT32_C({row.maximum_i64})}}}}",
            always=True,
        )
    return fields


def attr_copy_row(
    row: LowerAttrCopy,
    *,
    target_name_string_offset: str | None = None,
) -> list[str]:
    if target_name_string_offset is None:
        raise ValueError("attribute-copy row is missing its target-name string offset")
    fields: list[str] = []
    _append_field(fields, "kind", lower_rule_spelling.ATTR_COPY_KIND_C_NAMES[row.kind], always=True)
    _append_field(
        fields,
        "target_name_string_offset",
        target_name_string_offset,
        always=True,
    )
    if row.kind in (
        LowerAttrCopyKind.DIRECT,
        LowerAttrCopyKind.ENUM_ORDINAL,
        LowerAttrCopyKind.I64_ARRAY_ELEMENT,
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
        LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE,
        LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
        LowerAttrCopyKind.I64_LOW_BIT_MASK,
        LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_MASK,
        LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
        LowerAttrCopyKind.I64_LITERAL_MINUS_ATTR,
        LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS,
    ):
        _append_field(fields, "source_attr_index", row.source_attr_index, always=True)
    if row.kind in (
        LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_MASK,
        LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
        LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS,
    ):
        _append_field(
            fields,
            "other_source_attr_index",
            row.other_source_attr_index,
            always=True,
        )
    if row.kind in (
        LowerAttrCopyKind.I64_ARRAY_ELEMENT,
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
        LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
    ):
        _append_field(
            fields,
            "source_element_index",
            row.source_element_index,
            always=True,
        )
    if row.kind in (
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
        LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE,
        LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
    ):
        _append_field(
            fields,
            "source_element_count",
            row.source_element_count,
            always=True,
        )
    if row.kind in (
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
        LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE,
    ):
        _append_field(
            fields,
            "source_element_bit_width",
            row.source_element_bit_width,
            always=True,
        )
    _append_field(fields, "target_bit_offset", row.target_bit_offset)
    if attr_copy_uses_value_ref(row.kind):
        _append_field(fields, "value_ref_index", row.value_ref_index, always=True)
    if row.kind in (
        LowerAttrCopyKind.I64_LITERAL,
        LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
        LowerAttrCopyKind.I64_LITERAL_MINUS_ATTR,
        LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS,
        LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT,
        LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER,
    ):
        _append_field(
            fields,
            "literal_i64",
            _c_i64_literal(row.literal_i64),
            always=True,
        )
    if row.kind == LowerAttrCopyKind.SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE:
        _append_field(
            fields,
            "dynamic_term_index",
            row.dynamic_term_index,
            always=True,
        )
    return fields


def tied_result_row(row: LowerTiedResult) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "result_index", row.result_index, always=True)
    _append_field(fields, "operand_index", row.operand_index, always=True)
    if row.has_type_change:
        _append_field(fields, "has_type_change", lower_rule_spelling.c_bool(row.has_type_change))
    return fields


def emit_row(descriptor_refs: Mapping[str, int], row: LowerEmit) -> list[str]:
    fields: list[str] = []
    flags = lower_rule_spelling.emit_flags(row.flags)
    _append_field(fields, "kind", lower_rule_spelling.EMIT_KIND_C_NAMES[row.kind], always=True)
    _append_field(fields, "flags", flags)
    _append_field(
        fields,
        "descriptor_ref",
        _descriptor_ref_index(descriptor_refs, row.descriptor),
        always=True,
    )
    if row.operand_ref_count:
        _append_field(fields, "operand_ref_start", row.operand_ref_start, always=True)
        _append_field(fields, "operand_ref_count", row.operand_ref_count, always=True)
    _append_field(fields, "copy_operand_mask", row.copy_operand_mask)
    if row.kind == LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES:
        _append_field(
            fields,
            "accumulator_operand_index",
            row.accumulator_operand_index,
            always=True,
        )
    if row.result_ref_count:
        if row.flags & LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN:
            _append_field(
                fields,
                "result_type_pattern_start",
                row.result_type_pattern_start,
                always=True,
            )
        else:
            _append_field(fields, "result_ref_start", row.result_ref_start, always=True)
        _append_field(fields, "result_ref_count", row.result_ref_count, always=True)
    if row.flags & LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS:
        _append_field(
            fields,
            "result_bind_ref_start",
            row.result_bind_ref_start,
            always=True,
        )
    if row.attr_copy_count:
        _append_field(fields, "attr_copy_start", row.attr_copy_start, always=True)
        _append_field(fields, "attr_copy_count", row.attr_copy_count, always=True)
    if row.tied_result_count:
        _append_field(fields, "tied_result_start", row.tied_result_start, always=True)
        _append_field(fields, "tied_result_count", row.tied_result_count, always=True)
    if row.source_memory_ordinal != LOWER_SOURCE_MEMORY_NONE:
        _append_field(
            fields,
            "source_memory_ordinal",
            row.source_memory_ordinal,
            always=True,
        )
    return fields


def rule_row(
    row: LowerRule,
    report_key_ordinals: Mapping[str, int],
) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "source_op_kind", lower_rule_spelling.op_c_name(row.source_op), always=True)
    if row.flags:
        _append_field(fields, "flags", _rule_flags_c_expression(row.flags), always=True)
    if row.report_key:
        _append_field(
            fields,
            "report_key_ordinal",
            report_key_ordinals[row.report_key],
            always=True,
        )
    _append_field(fields, "temporary_count", row.temporary_count)
    if row.guard_count:
        _append_field(fields, "guard_start", row.guard_start, always=True)
        _append_field(fields, "guard_count", row.guard_count, always=True)
    if row.emit_count:
        _append_field(fields, "emit_start", row.emit_start, always=True)
        _append_field(fields, "emit_count", row.emit_count, always=True)
    if row.alias_ref_count:
        _append_field(fields, "alias_ref_start", row.alias_ref_start, always=True)
        _append_field(fields, "alias_ref_count", row.alias_ref_count, always=True)
    if row.elide_ref_count:
        _append_field(fields, "elide_ref_start", row.elide_ref_start, always=True)
        _append_field(fields, "elide_ref_count", row.elide_ref_count, always=True)
    return fields


def span_row(row: LowerRuleSpan) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "source_op_kind", lower_rule_spelling.op_c_name(row.source_op), always=True)
    _append_field(fields, "rule_start", row.rule_start, always=True)
    _append_field(fields, "rule_count", row.rule_count, always=True)
    return fields


def rule_set_row(
    *,
    table: CompiledLowerRuleSet,
    source_contract: ContractFragment,
    string_pool: CStringPool,
    string_data_name: str,
    spans_name: str,
    rules_name: str,
    report_keys: tuple[str, ...],
    report_keys_name: str,
    type_patterns_name: str,
    value_refs_name: str,
    materializers_name: str,
    source_memories_name: str,
    descriptor_ref_keys: tuple[str, ...],
    descriptor_refs_name: str,
    diagnostic_param_rows: tuple[tuple[LowerDiagnosticParam, int], ...],
    diagnostic_params_name: str,
    diagnostic_param_refs: tuple[int, ...],
    diagnostic_param_refs_name: str,
    guard_rows: tuple[LowerGuard, ...],
    guards_name: str,
    guard_refs: tuple[int, ...],
    guard_refs_name: str,
    attr_copies_name: str,
    tied_results_name: str,
    emits_name: str,
    diagnostics_name: str,
) -> list[str]:
    fields: list[str] = []
    if source_contract.target_contract_query:
        fields.append(".flags = LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY")
    if string_pool.entries:
        fields.append(f".string_table = {{.data = {string_data_name}, .data_length = sizeof({string_data_name}) - 1}}")
    _append_table_fields(fields, "spans", table.spans, spans_name)
    _append_table_fields(fields, "rules", table.rules, rules_name)
    _append_table_fields(
        fields,
        "report_key_string_offsets",
        report_keys,
        report_keys_name,
    )
    _append_table_fields(
        fields,
        "type_patterns",
        table.type_patterns,
        type_patterns_name,
    )
    _append_table_fields(fields, "value_refs", table.value_refs, value_refs_name)
    _append_table_fields(
        fields,
        "materializers",
        source_contract.materializers,
        materializers_name,
    )
    _append_table_fields(
        fields,
        "source_memories",
        table.source_memories,
        source_memories_name,
    )
    _append_table_fields(
        fields,
        "descriptor_refs",
        descriptor_ref_keys,
        descriptor_refs_name,
    )
    _append_table_fields(
        fields,
        "diagnostic_params",
        diagnostic_param_rows,
        diagnostic_params_name,
    )
    _append_table_fields(
        fields,
        "diagnostic_param_refs",
        diagnostic_param_refs,
        diagnostic_param_refs_name,
    )
    _append_table_fields(fields, "guards", guard_rows, guards_name)
    _append_table_fields(fields, "guard_refs", guard_refs, guard_refs_name)
    _append_table_fields(fields, "attr_copies", table.attr_copies, attr_copies_name)
    _append_table_fields(fields, "tied_results", table.tied_results, tied_results_name)
    _append_table_fields(fields, "emits", table.emits, emits_name)
    _append_table_fields(fields, "diagnostics", table.diagnostics, diagnostics_name)
    return fields


def _append_table_fields(
    fields: list[str],
    field_name: str,
    rows: tuple[object, ...],
    table_name: str,
) -> None:
    if not rows:
        return
    fields.append(f".{field_name} = {table_name}")
    fields.append(f".{_table_count_field_name(field_name)} = IREE_ARRAYSIZE({table_name})")


def _table_count_field_name(field_name: str) -> str:
    if field_name == "attr_copies":
        return "attr_copy_count"
    if field_name == "source_memories":
        return "source_memory_count"
    if field_name == "diagnostic_params":
        return "diagnostic_param_count"
    if field_name == "diagnostic_param_refs":
        return "diagnostic_param_ref_count"
    if field_name == "guard_refs":
        return "guard_ref_count"
    if field_name == "report_key_string_offsets":
        return "report_key_count"
    return f"{field_name[:-1]}_count"


def diagnostic_param_row(
    row: LowerDiagnosticParam,
    *,
    string_value_offset: str | None = None,
) -> list[str]:
    if row.kind == DiagnosticParamKind.STRING_LITERAL and string_value_offset is None:
        raise ValueError("diagnostic string literal is missing its string offset")
    fields: list[str] = []
    _append_field(
        fields,
        "kind",
        lower_rule_spelling.DIAGNOSTIC_PARAM_KIND_C_NAMES[row.kind],
        always=True,
    )
    if row.kind == DiagnosticParamKind.STRING_LITERAL:
        _append_field(
            fields,
            "value",
            f"{{.string_value_offset = {string_value_offset}}}",
            always=True,
        )
    if row.kind == DiagnosticParamKind.VALUE_TYPE:
        _append_field(
            fields,
            "value",
            f"{{.value_ref_index = {row.value_ref_index}}}",
            always=True,
        )
    if row.kind == DiagnosticParamKind.I64_LITERAL:
        _append_field(
            fields,
            "value",
            f"{{.i64_value = {_c_i64_literal(row.i64_value)}}}",
            always=True,
        )
    if row.kind == DiagnosticParamKind.U32_LITERAL:
        _append_field(
            fields,
            "value",
            f"{{.u32_value = {row.u32_value}}}",
            always=True,
        )
    if row.kind == DiagnosticParamKind.U64_LITERAL:
        _append_field(
            fields,
            "value",
            f"{{.u64_value = {lower_rule_spelling.u64_c_literal(row.u64_value)}}}",
            always=True,
        )
    if row.kind == DiagnosticParamKind.BOOL_LITERAL:
        _append_field(
            fields,
            "value",
            f"{{.bool_value = {str(row.bool_value).lower()}}}",
            always=True,
        )
    return fields


def type_pattern_row(type_pattern: TypePattern) -> list[str]:
    flags = [
        "LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND",
        "LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT",
    ]
    row = [
        ".flags = " + " | ".join(flags),
        f".type_kind = {lower_rule_spelling.type_kind_c_name(type_pattern)}",
        f".element_type_mask = {lower_rule_spelling.scalar_type_mask_c_expr(type_pattern.elements)}",
    ]
    if type_pattern.dims:
        row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK"
        row.extend(
            [
                f".rank = {len(type_pattern.dims)}",
                f".static_dim0 = {lower_rule_spelling.c_expression(type_pattern.dims[0])}",
            ]
        )
        row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0"
        if len(type_pattern.dims) >= 2:
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1"
            row.append(f".static_dim1 = {lower_rule_spelling.c_expression(type_pattern.dims[1])}")
    elif type_pattern.kind == "vector":
        if type_pattern.lanes is not None:
            row.append(".rank = 1")
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0"
            row.append(f".static_dim0 = {lower_rule_spelling.c_expression(type_pattern.lanes)}")
        elif type_pattern.minimum_lanes is not None and type_pattern.maximum_lanes is not None:
            row.append(".rank = 1")
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE"
            row.extend(
                [
                    f".static_dim0_min = {lower_rule_spelling.c_expression(type_pattern.minimum_lanes)}",
                    f".static_dim0_max = {lower_rule_spelling.c_expression(type_pattern.maximum_lanes)}",
                ]
            )
        elif type_pattern.minimum_static_elements is not None and type_pattern.maximum_static_elements is not None:
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE"
            row.extend(
                [
                    f".static_element_count_min = {lower_rule_spelling.c_expression(type_pattern.minimum_static_elements)}",
                    f".static_element_count_max = {lower_rule_spelling.c_expression(type_pattern.maximum_static_elements)}",
                ]
            )
        else:
            raise ValueError("generated vector type patterns require static shape")
    return row
