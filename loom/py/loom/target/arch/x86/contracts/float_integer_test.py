# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.float_integer import (
    avx2_scalar_float_integer_rules,
    avx2_vector_float_integer_rules,
    avx512_scalar_float_integer_rules,
    avx512_vector_float_integer_rules,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_INTEGER_I32_FORMS,
    AVX512_FLOAT_INTEGER_I32_FORMS,
    AVX512_FLOAT_INTEGER_I64_FORMS,
)
from loom.target.contracts import (
    DescriptorRule,
    GuardKind,
    Scalar,
    Vector,
    descriptor_by_key,
)


def _avx2_descriptor(key: str):
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


def _avx512_descriptor(key: str):
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _value_type(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def test_scalar_rules_cover_signed_unsigned_types_and_native_overlays() -> None:
    base_rules = avx2_scalar_float_integer_rules(_avx2_descriptor)
    native_rules = avx512_scalar_float_integer_rules(_avx512_descriptor)
    for rule in base_rules:
        rule.validate(X86_AVX2_DESCRIPTOR_SET)
    for rule in native_rules:
        rule.validate(X86_AVX512_CORE_DESCRIPTOR_SET)

    operations = (
        scalar_conversion.scalar_fptosi,
        scalar_conversion.scalar_fptoui,
    )
    assert {
        (rule.source_op, _value_type(rule, "input"), _value_type(rule, "result"))
        for rule in base_rules
    } == {
        (source_op, Scalar(f"f{source_width}"), Scalar(f"i{result_width}"))
        for source_op in operations
        for source_width in (32, 64)
        for result_width in (1, 8, 16, 32, 64)
    }
    assert {
        (rule.source_op, _value_type(rule, "input"), _value_type(rule, "result"))
        for rule in native_rules
    } == {
        (
            scalar_conversion.scalar_fptoui,
            Scalar(f"f{source_width}"),
            Scalar(f"i{result_width}"),
        )
        for source_width in (32, 64)
        for result_width in (32, 64)
    }


def test_vector_rules_cover_every_native_or_composed_packet_cell() -> None:
    avx2_rules = avx2_vector_float_integer_rules(_avx2_descriptor)
    avx512_rules = avx512_vector_float_integer_rules(_avx512_descriptor)
    for rule in avx2_rules:
        rule.validate(X86_AVX2_DESCRIPTOR_SET)
    for rule in avx512_rules:
        rule.validate(X86_AVX512_CORE_DESCRIPTOR_SET)

    operations = (vector.vector_fptosi, vector.vector_fptoui)
    assert {
        (rule.source_op, _value_type(rule, "input"), _value_type(rule, "result"))
        for rule in avx2_rules
    } == {
        (
            source_op,
            Vector(form.source.name, lanes=form.lane_count),
            Vector(f"i{result_width}", lanes=form.lane_count),
        )
        for form in AVX2_FLOAT_INTEGER_I32_FORMS
        for source_op in operations
        for result_width in (8, 16, 32)
    }

    actual_counts = Counter(
        (rule.source_op, _value_type(rule, "input"), _value_type(rule, "result"))
        for rule in avx512_rules
    )
    expected_counts = Counter()
    for form in AVX512_FLOAT_INTEGER_I32_FORMS:
        source_type = Vector(form.source.name, lanes=form.lane_count)
        expected_counts[
            (
                vector.vector_fptoui,
                source_type,
                Vector("i32", lanes=form.lane_count),
            )
        ] += 1
        if form in AVX512_FLOAT_INTEGER_I32_FORMS[-2:]:
            expected_counts[
                (
                    vector.vector_fptosi,
                    source_type,
                    Vector("i32", lanes=form.lane_count),
                )
            ] += 1
        for source_op in operations:
            for result_width in (8, 16):
                expected_counts[
                    (
                        source_op,
                        source_type,
                        Vector(f"i{result_width}", lanes=form.lane_count),
                    )
                ] += 1
    f64x8 = AVX512_FLOAT_INTEGER_I32_FORMS[-1]
    for source_op in operations:
        for result_width in (8, 16):
            expected_counts[
                (
                    source_op,
                    Vector("f64", lanes=f64x8.lane_count),
                    Vector(f"i{result_width}", lanes=f64x8.lane_count),
                )
            ] += 1
    for form in AVX512_FLOAT_INTEGER_I64_FORMS:
        for source_op in operations:
            expected_counts[
                (
                    source_op,
                    Vector(form.source.name, lanes=form.lane_count),
                    Vector("i64", lanes=form.lane_count),
                )
            ] += 1
    assert actual_counts == expected_counts
