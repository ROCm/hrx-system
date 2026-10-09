# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-level tests for Wasm SIMD predicate contracts."""

from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import DescriptorRule, Guard, GuardKind, TypePattern
from loom.target.emit.wasm.predicate import (
    FLOAT_PREDICATES,
    INTEGER_PREDICATES,
    NNAN_FLOAT_PREDICATES,
    PREDICATE_LANE_TYPE,
    PREDICATE_PAYLOAD_FAMILIES,
    PREDICATE_REPRESENTATIONS,
    PREDICATE_TYPE,
    _predicate_conversion_lanes,
    predicate_rules,
)
from loom.target.low_descriptors import Descriptor


def _descriptor(key: str) -> Descriptor:
    return next(
        descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
        if descriptor.key == key
    )


_RULES = predicate_rules(_descriptor, Guard.value_type)


def _rules_for(source_op):
    return tuple(
        rule
        for rule in _RULES
        if isinstance(rule, DescriptorRule) and rule.source_op is source_op
    )


def _type_guard(rule: DescriptorRule, field: str) -> TypePattern:
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def _representation_guard(rule: DescriptorRule, field: str) -> int:
    return next(
        guard.count
        for guard in rule.guards
        if guard.kind is GuardKind.LOW_VALUE_REPRESENTATION and guard.field == field
    )


def _predicate_guard(rule: DescriptorRule) -> str:
    return next(
        guard.enum_keyword
        for guard in rule.guards
        if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "predicate"
    )


def _has_nnan_guard(rule: DescriptorRule) -> bool:
    return any(
        guard.kind is GuardKind.INSTANCE_FLAGS_HAS_ALL and guard.field == "fastmath"
        for guard in rule.guards
    )


def test_conversion_shuffle_preserves_each_live_lane_across_every_width_pair():
    for source_width in PREDICATE_REPRESENTATIONS:
        for result_width in PREDICATE_REPRESENTATIONS:
            lanes = _predicate_conversion_lanes(source_width, result_width)
            assert len(lanes) == 16
            source_byte_count = source_width // 8
            result_byte_count = result_width // 8
            live_lane_count = min(128 // source_width, 128 // result_width)
            for lane in range(live_lane_count):
                assert (
                    lanes[lane * result_byte_count : (lane + 1) * result_byte_count]
                    == (lane * source_byte_count,) * result_byte_count
                )


def test_vector_select_covers_every_condition_and_payload_width_pair():
    rules = _rules_for(vector.vector_select)
    predicate_rules = [
        rule for rule in rules if _type_guard(rule, "result").elements == ("i1",)
    ]
    assert len(predicate_rules) == 1
    assert len(predicate_rules[0].emit) == 1

    actual = set()
    for rule in rules:
        result_type = _type_guard(rule, "result")
        if result_type.elements == ("i1",):
            continue
        payload_width = next(
            width
            for width, payload_type in PREDICATE_PAYLOAD_FAMILIES
            if payload_type == result_type
        )
        condition_width = _representation_guard(rule, "condition")
        actual.add((condition_width, payload_width))
        descriptor_keys = [emit.descriptor.key for emit in rule.emit]
        assert descriptor_keys[-1] == "wasm.v128.bitselect"
        assert descriptor_keys.count("wasm.i8x16.shuffle") == (
            condition_width != payload_width
        )
    assert actual == {
        (condition_width, payload_width)
        for condition_width in PREDICATE_REPRESENTATIONS
        for payload_width, _ in PREDICATE_PAYLOAD_FAMILIES
    }


def test_compare_rules_cover_every_type_predicate_and_result_representation():
    expected = {
        (element, predicate, representation)
        for element in ("i8", "i16", "i32", "i64")
        for predicate in INTEGER_PREDICATES
        for representation in PREDICATE_REPRESENTATIONS
    } | {
        (element, predicate, representation)
        for element in ("f32", "f64")
        for predicate in FLOAT_PREDICATES
        for representation in PREDICATE_REPRESENTATIONS
    }
    actual = set()
    nnan_actual = set()
    for source_op in (vector.vector_cmpi, vector.vector_cmpf):
        for rule in _rules_for(source_op):
            (element,) = _type_guard(rule, "lhs").elements
            representation = _representation_guard(rule, "result")
            cell = (element, _predicate_guard(rule), representation)
            if _has_nnan_guard(rule):
                nnan_actual.add(cell)
                assert rule.priority == 1
                assert rule.report_key.endswith(".nnan")
            else:
                actual.add(cell)
                assert rule.priority == 0
            descriptor_keys = [emit.descriptor.key for emit in rule.emit]
            native_width = int(element[1:])
            assert descriptor_keys.count("wasm.i8x16.shuffle") == (
                representation != native_width
            )
    assert actual == expected
    assert nnan_actual == {
        (element, predicate, representation)
        for element in ("f32", "f64")
        for predicate in NNAN_FLOAT_PREDICATES
        for representation in PREDICATE_REPRESENTATIONS
    }


def test_i64_unsigned_relations_bias_both_operands_before_signed_comparison():
    for rule in _rules_for(vector.vector_cmpi):
        if _type_guard(rule, "lhs").elements != ("i64",):
            continue
        predicate = _predicate_guard(rule)
        if not predicate.startswith("u"):
            continue
        assert [emit.descriptor.key for emit in rule.emit[:4]] == [
            "wasm.v128.const",
            "wasm.v128.xor",
            "wasm.v128.xor",
            f"wasm.i64x2.{predicate[1:]}_s",
        ]


def test_float_predicates_have_exact_ieee_compositions_and_nnan_fast_paths():
    general_operations = {
        "oeq": ("eq",),
        "ogt": ("gt",),
        "oge": ("ge",),
        "olt": ("lt",),
        "ole": ("le",),
        "une": ("ne",),
        "one": ("lt", "gt", "v128.or"),
        "ord": ("eq", "eq", "v128.and"),
        "ueq": ("lt", "gt", "v128.or", "v128.not"),
        "ugt": ("le", "v128.not"),
        "uge": ("lt", "v128.not"),
        "ult": ("ge", "v128.not"),
        "ule": ("gt", "v128.not"),
        "uno": ("eq", "eq", "v128.and", "v128.not"),
    }
    for element, native_width, shape in (
        ("f32", 32, "f32x4"),
        ("f64", 64, "f64x2"),
    ):
        rules = [
            rule
            for rule in _rules_for(vector.vector_cmpf)
            if _type_guard(rule, "lhs").elements == (element,)
            and _representation_guard(rule, "result") == native_width
        ]
        for predicate, operations in general_operations.items():
            rule = next(
                rule
                for rule in rules
                if _predicate_guard(rule) == predicate and not _has_nnan_guard(rule)
            )
            expected = tuple(
                f"wasm.{operation}"
                if operation.startswith("v128.")
                else f"wasm.{shape}.{operation}"
                for operation in operations
            )
            assert tuple(emit.descriptor.key for emit in rule.emit) == expected

        for predicate, native_predicate in NNAN_FLOAT_PREDICATES.items():
            rule = next(
                rule
                for rule in rules
                if _predicate_guard(rule) == predicate and _has_nnan_guard(rule)
            )
            operation = {
                "oeq": "eq",
                "ogt": "gt",
                "oge": "ge",
                "olt": "lt",
                "ole": "le",
                "une": "ne",
            }[native_predicate]
            assert [emit.descriptor.key for emit in rule.emit] == [
                f"wasm.{shape}.{operation}"
            ]


def test_lane_access_has_one_rule_for_each_physical_representation():
    for source_op, field, rules_per_representation in (
        (vector.vector_extract, "source", 1),
        (vector.vector_insert, "dest", 2),
    ):
        counts = {representation: 0 for representation in PREDICATE_REPRESENTATIONS}
        for rule in _rules_for(source_op):
            assert _type_guard(rule, field) == PREDICATE_LANE_TYPE
            representation = _representation_guard(rule, field)
            counts[representation] += 1
            if representation == 64:
                descriptor_keys = [emit.descriptor.key for emit in rule.emit]
                expected_conversion = (
                    "wasm.i32.wrap_i64"
                    if source_op is vector.vector_extract
                    else "wasm.i64.extend_i32_s"
                )
                assert expected_conversion in descriptor_keys
        assert counts == {
            representation: rules_per_representation
            for representation in PREDICATE_REPRESENTATIONS
        }


def test_representation_invariant_rules_are_single_shared_rows():
    for source_op in (
        vector.vector_constant,
        vector.vector_splat,
        vector.vector_andi,
        vector.vector_ori,
        vector.vector_xori,
        scf.scf_select,
    ):
        rules = _rules_for(source_op)
        assert len(rules) == 1
        assert all(
            _type_guard(rule, "result").elements == PREDICATE_TYPE.elements
            for rule in rules
        )
    assert PREDICATE_TYPE.minimum_static_elements == 1
    assert PREDICATE_TYPE.maximum_static_elements == 16
    assert PREDICATE_TYPE.minimum_lanes is None
