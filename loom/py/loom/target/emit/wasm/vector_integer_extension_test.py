# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import Guard, GuardKind, descriptor_by_key
from loom.target.emit.wasm.vector_integer_extension import integer_extension_rules


def _descriptor(key):
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_guard(field, type_pattern):
    return Guard.value_type(field, type_pattern)


def test_rules_cover_every_integer_widening_pair_and_signedness():
    rules = integer_extension_rules(_descriptor, _type_guard)

    expected_pairs = {
        (source, result, signedness)
        for source, results in (
            (8, (16, 32, 64)),
            (16, (32, 64)),
            (32, (64,)),
        )
        for result in results
        for signedness in ("s", "u")
    }
    actual_pairs = set()
    for rule in rules:
        type_guards = tuple(
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_TYPE
        )
        assert len(type_guards) == 2
        source_type = type_guards[0].type_pattern
        result_type = type_guards[1].type_pattern
        assert source_type is not None
        assert result_type is not None
        source = int(source_type.elements[0][1:])
        result = int(result_type.elements[0][1:])
        signedness = "s" if rule.source_op.name == "vector.extsi" else "u"
        actual_pairs.add((source, result, signedness))
        maximum_lanes = 128 // result
        assert source_type.minimum_lanes == result_type.minimum_lanes == 1
        assert source_type.maximum_lanes == result_type.maximum_lanes == maximum_lanes
        assert any(
            guard.kind is GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
            for guard in rule.guards
        )

    assert actual_pairs == expected_pairs
    assert len(rules) == len(expected_pairs) == 12


def test_rules_compose_one_low_half_extension_per_width_step():
    rules = integer_extension_rules(_descriptor, _type_guard)

    for rule in rules:
        type_guards = tuple(
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_TYPE
        )
        source_type = type_guards[0].type_pattern
        result_type = type_guards[1].type_pattern
        assert source_type is not None
        assert result_type is not None
        source = int(source_type.elements[0][1:])
        result = int(result_type.elements[0][1:])
        signedness = "s" if rule.source_op.name == "vector.extsi" else "u"

        expected_keys = []
        width = source
        while width < result:
            next_width = width * 2
            expected_keys.append(
                f"wasm.i{next_width}x{128 // next_width}.extend_low_"
                f"i{width}x{128 // width}_{signedness}"
            )
            width = next_width
        assert [emit.descriptor.key for emit in rule.emit] == expected_keys
        assert rule.report_key == (
            "wasm.integer_extension.native"
            if len(expected_keys) == 1
            else "wasm.integer_extension.composed"
        )
