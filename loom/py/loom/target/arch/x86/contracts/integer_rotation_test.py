# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise as scalar
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.integer_rotation import (
    avx512_vector_rotation_rules,
    scalar_integer_rotation_rules,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX512_CORE_DESCRIPTOR_SET,
    X86_SCALAR_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorRule,
    Guard,
    GuardKind,
    Scalar,
    Vector,
    descriptor_by_key,
)


def _scalar_descriptor(key: str):
    return descriptor_by_key(X86_SCALAR_DESCRIPTOR_SET, key)


def _avx512_descriptor(key: str):
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _guard(rule: DescriptorRule, kind: GuardKind, field: str) -> Guard:
    return next(
        guard for guard in rule.guards if guard.kind is kind and guard.field == field
    )


def test_scalar_rules_cover_source_types_and_keep_cl_constraints_use_local() -> None:
    rules = scalar_integer_rotation_rules(_scalar_descriptor)
    for rule in rules:
        rule.validate(X86_SCALAR_DESCRIPTOR_SET)

    source_types = (
        *((False, Scalar(f"i{bit_count}")) for bit_count in (8, 16, 32, 64)),
        (True, Scalar("index")),
    )
    expected = {
        (
            index_source_op if is_index else scalar_source_op,
            type_pattern,
            immediate,
        )
        for scalar_source_op, index_source_op in (
            (scalar.scalar_rotli, index.index_rotli),
            (scalar.scalar_rotri, index.index_rotri),
        )
        for is_index, type_pattern in source_types
        for immediate in (False, True)
    }
    actual = {
        (
            rule.source_op,
            _guard(rule, GuardKind.VALUE_TYPE, "result").type_pattern,
            any(guard.kind is GuardKind.VALUE_EXACT_I64 for guard in rule.guards),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)

    for rule in rules:
        (emit,) = rule.emit
        immediate = any(
            guard.kind is GuardKind.VALUE_EXACT_I64 for guard in rule.guards
        )
        if immediate:
            assert rule.priority == 2
            count_range = _guard(rule, GuardKind.VALUE_I64_RANGE, "rhs")
            assert (count_range.minimum, count_range.maximum) == (0, 255)
            assert not emit.copy_operands
        else:
            assert rule.priority == 1
            assert emit.copy_operands == ("rhs",)


def test_avx512_rules_cover_every_native_rotation_cell() -> None:
    rules = avx512_vector_rotation_rules(_avx512_descriptor)
    for rule in rules:
        rule.validate(X86_AVX512_CORE_DESCRIPTOR_SET)

    register_class_by_type = {
        Vector(element, lanes=vector_bit_width // element_bit_width): register_class
        for element, element_bit_width in (("i32", 32), ("i64", 64))
        for vector_bit_width, register_class in (
            (128, "x86.xmm"),
            (256, "x86.ymm"),
            (512, "x86.zmm"),
        )
    }
    expected = {
        (source_op, value_type, immediate)
        for source_op in (vector.vector_rotli, vector.vector_rotri)
        for value_type in register_class_by_type
        for immediate in (False, True)
    }
    actual = {
        (
            rule.source_op,
            _guard(rule, GuardKind.VALUE_TYPE, "result").type_pattern,
            any(guard.kind is GuardKind.VALUE_EXACT_I64 for guard in rule.guards),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)

    for rule in rules:
        assert rule.descriptor is not None
        value_type = _guard(rule, GuardKind.VALUE_TYPE, "result").type_pattern
        register_class = register_class_by_type[value_type]
        for field in ("lhs", "rhs", "result"):
            assert _guard(rule, GuardKind.VALUE_TYPE, field).type_pattern == value_type
        immediate = any(
            guard.kind is GuardKind.VALUE_EXACT_I64 for guard in rule.guards
        )
        if immediate:
            assert rule.priority == 2
            count_range = _guard(rule, GuardKind.VALUE_I64_RANGE, "rhs")
            assert (count_range.minimum, count_range.maximum) == (0, 255)
        else:
            assert rule.priority == 1
            for field in ("lhs", "rhs", "result"):
                assert (
                    _guard(
                        rule, GuardKind.LOW_VALUE_REGISTER_CLASS, field
                    ).register_class
                    == register_class
                )
        availability = tuple(
            guard.descriptor
            for guard in rule.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
        )
        assert availability == (
            (rule.descriptor,) if register_class != "x86.zmm" else ()
        )
