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
    ERR_TARGET_008,
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
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source_memory import (
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
)
from loom.target.low_descriptors import (
    Descriptor,
)


def _type_pattern_text(type_pattern: TypePattern) -> str:
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


def _value_type_diagnostic(field: str, type_pattern: TypePattern) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_002,
        string_param("field_name", field),
        value_type_param("actual_type", field),
        string_param("expected_type", _type_pattern_text(type_pattern)),
    )


def _enum_attr_diagnostic(field: str, enum_keyword: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"enum_case.{enum_keyword}",
    )


def _i64_attr_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "field", field, "i64_attr_range", minimum, maximum
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


def _exact_power_of_two_integer_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        "exact_power_of_two_i64",
    )


def _u32_divisor_magic_is_add_diagnostic(field: str, *, is_add: bool) -> DiagnosticRef:
    suffix = "add" if is_add else "no_add"
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        f"u32_divisor_magic.{suffix}",
    )


def _exact_float_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "exact_float")


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


def _value_no_uses_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value", field, "no_ordinary_uses")


def _instance_flags_diagnostic(field: str, enum_keyword: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("flags", field, f"has_all.{enum_keyword}")


def _source_memory_diagnostic(
    constraint: SourceMemoryConstraint,
) -> DiagnosticRef:
    if constraint.diagnostic is not None:
        ref = constraint.diagnostic.ref
        if ref is None:
            raise ValueError("source-memory diagnostic is missing an error ref")
        return ref
    return target_diagnostic(
        ERR_TARGET_008,
        string_param("operation_kind", constraint.operation.value),
    )


def _source_memory_dynamic_offset_diagnostic(
    constraint: SourceMemoryConstraint,
) -> DiagnosticRef:
    if constraint.dynamic_offset_diagnostic is not None:
        ref = constraint.dynamic_offset_diagnostic.ref
        if ref is None:
            raise ValueError(
                "source-memory dynamic-offset diagnostic is missing an error ref"
            )
        return ref
    return _source_memory_diagnostic(constraint)


def _source_memory_address_layout_diagnostic(
    constraint: SourceMemoryConstraint,
) -> DiagnosticRef:
    if constraint.address_layout_diagnostic is not None:
        ref = constraint.address_layout_diagnostic.ref
        if ref is None:
            raise ValueError(
                "source-memory address-layout diagnostic is missing an error ref"
            )
        return ref
    return _source_memory_diagnostic(constraint)


def _source_memory_address_diagnostic(
    constraint: SourceMemoryConstraint,
    materializer: SourceMemoryAddressMaterializer | None,
) -> DiagnosticRef:
    if materializer is not None and materializer.diagnostic is not None:
        ref = materializer.diagnostic.ref
        if ref is None:
            raise ValueError("source-memory address diagnostic is missing an error ref")
        return ref
    return _source_memory_diagnostic(constraint)


def _attr_diagnostic(field: str, attr_type: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("field", field, f"attr_kind.{attr_type}")


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
