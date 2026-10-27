# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical diagnostics for generated target-Low rule guards."""

from __future__ import annotations

from loom.error.target import (
    ERR_TARGET_002,
    ERR_TARGET_003,
    ERR_TARGET_004,
    ERR_TARGET_005,
    ERR_TARGET_006,
    ERR_TARGET_007,
)
from loom.target.contracts.diagnostics import (
    DiagnosticRef,
    i64_param,
    string_param,
    target_diagnostic,
    u32_param,
    value_type_param,
)
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.lower_rule_bindings import _f64_bits
from loom.target.contracts.lower_rule_tables import SourceMemoryRejectionReason
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source_memory import (
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
)
from loom.target.low_descriptors import (
    Descriptor,
)


def _type_pattern_text(type_pattern: TypePattern) -> str:
    if type_pattern.kind == "buffer":
        return "buffer"
    element_text = _type_pattern_element_text(type_pattern)
    if type_pattern.kind == "scalar":
        return element_text
    if type_pattern.kind == "view":
        return f"view<{element_text}>"
    if type_pattern.dims:
        dims_text = "x".join(str(dim) for dim in type_pattern.dims)
        return f"vector<{dims_text}x{element_text}>"
    if type_pattern.lanes is not None:
        return f"vector<{type_pattern.lanes}x{element_text}>"
    return f"vector<{element_text}>"


def _type_pattern_element_text(type_pattern: TypePattern) -> str:
    if len(type_pattern.elements) == 1:
        return type_pattern.elements[0]
    return "{" + ", ".join(type_pattern.elements) + "}"


def _value_type_diagnostic(
    field: str,
    type_pattern: TypePattern,
    *,
    element: int = 0,
) -> DiagnosticRef:
    field_name = f"{field}[{element}]" if element else field
    return target_diagnostic(
        ERR_TARGET_002,
        string_param("field_name", field_name),
        value_type_param("actual_type", field, element=element),
        string_param("expected_type", _type_pattern_text(type_pattern)),
    )


def _enum_attr_diagnostic(field: str, enum_keyword: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"enum_case.{enum_keyword}",
    )


def _enum_attr_set_diagnostic(
    field: str,
    enum_keywords: tuple[str, ...],
) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        "enum_set." + ".".join(enum_keywords),
    )


def _i64_attr_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "field", field, "i64_attr_range", minimum, maximum
    )


def _i64_attrs_sum_diagnostic(
    field: str,
    other_field: str,
    value: int,
) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "fields",
        f"{field}+{other_field}",
        f"i64_attrs_sum_equals.{value}",
    )


def _descriptor_available_diagnostic(descriptor: Descriptor) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "descriptor",
        descriptor.key,
        "descriptor_available",
    )


def _materializer_diagnostic(field: str, materializer: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"materializer.{materializer}",
    )


def _register_class_diagnostic(field: str, register_class: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"low_register_class.{register_class}",
    )


def _static_dim0_multiple_diagnostic(field: str, multiple: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "field",
        field,
        "static_dim0_multiple",
        multiple,
    )


def _register_unit_count_diagnostic(
    field: str,
    other_field: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "field",
        field,
        other_field,
        "low_register_unit_count_eq",
    )


def _static_element_count_relation_diagnostic(
    field: str,
    other_field: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "field",
        field,
        other_field,
        "static_element_count_eq",
    )


def _register_unit_count_exact_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "field",
        field,
        "low_register_unit_count",
        count,
    )


def _operand_segment_count_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "operand_segment",
        field,
        "operand_segment_count",
        count,
    )


def _i64_array_count_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic("i64_array", field, "array_count", count)


def _i64_array_element_range_diagnostic(
    field: str,
    element: int,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _element_range_constraint_diagnostic(
        "i64_array",
        field,
        "element_range",
        element,
        minimum,
        maximum,
    )


def _i64_array_elements_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "i64_array",
        field,
        "elements_range",
        minimum,
        maximum,
    )


def _bounded_integer_diagnostic(field: str, guard: Guard) -> DiagnosticRef:
    signedness = (
        "signed" if guard.kind == GuardKind.VALUE_SIGNED_BIT_COUNT else "unsigned"
    )
    return _count_constraint_diagnostic(
        "value_fact",
        field,
        f"{signedness}_bit_count",
        guard.count or 0,
    )


def _exact_integer_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "exact_i64")


def _exact_power_of_two_integer_diagnostic(field: str, addend: int) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        "exact_power_of_two_i64"
        if addend == 0
        else f"exact_power_of_two_i64.addend({addend})",
    )


def _u32_divisor_magic_kind_diagnostic(field: str, kind: int) -> DiagnosticRef:
    suffix = ("multiply", "multiply_shift", "multiply_add_shift")[kind]
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        f"u32_divisor_magic.{suffix}",
    )


def _i32_divisor_magic_kind_diagnostic(field: str, kind: int) -> DiagnosticRef:
    suffix = (
        "multiply",
        "multiply_shift",
        "multiply_add",
        "multiply_add_shift",
        "multiply_subtract",
        "multiply_subtract_shift",
    )[kind]
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        f"i32_divisor_magic.{suffix}",
    )


def _exact_float_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "exact_float")


def _exact_float_type_diagnostic(field: str, type_field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact", field, f"finite_exactly_representable_as_{type_field}"
    )


def _exact_power_of_two_float_diagnostic(
    field: str,
    minimum_exponent: int,
    maximum_exponent: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "value_fact",
        field,
        "exact_power_of_two_float_exponent",
        minimum_exponent,
        maximum_exponent,
    )


def _not_nan_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "not_nan")


def _not_subnormal_or_flag_diagnostic(
    value_field: str,
    flags_field: str,
    flag_keyword: str,
) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact",
        value_field,
        f"not_subnormal_or_{flags_field}_{flag_keyword}",
    )


def _integer_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "value_fact", field, "i64_range", minimum, maximum
    )


def _integer_range_relation_diagnostic(
    field: str,
    other_field: str,
    relation: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "value_fact",
        field,
        other_field,
        f"i64_range_{relation}",
    )


def _float_equals_diagnostic(field: str, value: float) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact", field, f"float_equals.0x{_f64_bits(value):016x}"
    )


def _storage_element_format_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value", field, "storage_schema.element_format")


def _storage_operand_schema_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value", field, "storage_schema.encoded_operand"
    )


def _value_no_uses_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value", field, "no_ordinary_uses")


def _value_no_uses_after_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value", field, "no_ordinary_uses_after_source_op"
    )


def _instance_flags_diagnostic(
    field: str, enum_keyword: str, predicate: str
) -> DiagnosticRef:
    return _named_constraint_diagnostic("flags", field, f"{predicate}.{enum_keyword}")


def _named_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_003,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
    )


# The default projections depend only on the rejection reason, not the access.
_DEFAULT_SOURCE_MEMORY_DIAGNOSTICS = {
    reason: _named_constraint_diagnostic("source-memory", "access", reason.value)
    for reason in SourceMemoryRejectionReason
}


def _source_memory_diagnostic(
    constraint: SourceMemoryConstraint,
    reason: SourceMemoryRejectionReason,
    materializer: SourceMemoryAddressMaterializer | None,
) -> DiagnosticRef:
    override = None
    if reason == SourceMemoryRejectionReason.MINIMUM_ALIGNMENT:
        override = constraint.alignment_diagnostic
    elif reason == SourceMemoryRejectionReason.BYTE_OFFSET_WIDTH:
        override = constraint.byte_offset_diagnostic
    elif reason == SourceMemoryRejectionReason.ADDRESS_LAYOUT:
        override = constraint.address_layout_diagnostic
    elif reason == SourceMemoryRejectionReason.ADDRESS_MATERIALIZATION:
        override = materializer.diagnostic if materializer is not None else None
    if override is None:
        override = constraint.diagnostic
    if override is not None:
        ref = override.ref
        if ref is None:
            raise ValueError(
                f"source-memory {reason.value} diagnostic is missing an error ref"
            )
        return ref
    return _DEFAULT_SOURCE_MEMORY_DIAGNOSTICS[reason]


def _source_memory_diagnostics(
    constraint: SourceMemoryConstraint,
    materializer: SourceMemoryAddressMaterializer | None,
) -> tuple[DiagnosticRef, ...]:
    return tuple(
        _source_memory_diagnostic(constraint, reason, materializer)
        for reason in SourceMemoryRejectionReason
    )


def _attr_diagnostic(field: str, attr_type: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("field", field, f"attr_kind.{attr_type}")


def _count_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    expected_count: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_004,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
        u32_param("expected_count", expected_count),
    )


def _range_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_005,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
        i64_param("minimum", minimum),
        i64_param("maximum", maximum),
    )


def _element_range_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    element: int,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_006,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        u32_param("element", element),
        string_param("constraint_key", constraint_key),
        i64_param("minimum", minimum),
        i64_param("maximum", maximum),
    )


def _relation_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    other_subject_name: str,
    constraint_key: str,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_007,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("other_subject_name", other_subject_name),
        string_param("constraint_key", constraint_key),
    )


def _guard_diagnostic(
    guard: Guard,
    default_diagnostic: DiagnosticRef,
) -> DiagnosticRef:
    if guard.diagnostic is None:
        return default_diagnostic
    ref = guard.diagnostic.ref
    if ref is None:
        raise ValueError("guard diagnostic is missing an error ref")
    return ref
