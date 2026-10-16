# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.predicate import (
    avx2_predicate_register_class,
    avx2_predicate_representation_bit_widths,
    avx2_predicate_representation_id,
)
from loom.target.arch.x86.contracts.vector_integer_count import (
    avx2_vector_integer_count_rules,
    avx512_vector_integer_count_rules,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractCase,
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    ReadOnlyDataLiteral,
    TypePattern,
    ValueAliasRule,
    descriptor_by_key,
)


def _avx2_descriptor(key: str):
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


def _avx512_descriptor(key: str):
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _guard(rule: DescriptorRule | ValueAliasRule, kind: GuardKind, field: str):
    return next(
        guard for guard in rule.guards if guard.kind is kind and guard.field == field
    )


def _numeric_rules(rules: tuple[ContractCase, ...]) -> tuple[DescriptorRule, ...]:
    return tuple(
        rule
        for rule in rules
        if isinstance(rule, DescriptorRule)
        and rule.report_key == "integer_bit_count.vector.composed"
    )


def _literal_payloads(rule: DescriptorRule) -> tuple[bytes, ...]:
    return tuple(
        literal.data
        for emit in rule.emit
        if isinstance(emit, EmitDescriptorOp)
        for literal in emit.immediates.values()
        if isinstance(literal, ReadOnlyDataLiteral)
    )


def _lane_bounds(type_pattern: TypePattern) -> tuple[int, int]:
    if type_pattern.lanes is not None:
        return type_pattern.lanes, type_pattern.lanes
    assert isinstance(type_pattern.minimum_lanes, int)
    assert isinstance(type_pattern.maximum_lanes, int)
    return (
        type_pattern.minimum_lanes,
        type_pattern.maximum_lanes,
    )


def test_evex_byte_shuffle_descriptor_matches_the_avx512bw_encoding() -> None:
    avx2 = _avx2_descriptor("x86.avx2.vpshufb.ymm")
    avx512 = _avx512_descriptor("x86.avx512.vpshufb.zmm")
    assert avx512.encoding_format_id == avx2.encoding_format_id
    assert avx512.encoding_id == 0xA600
    assert avx512.semantic_tag == "bits.shuffle.bytes.v512"


def test_numeric_rules_cover_every_operation_element_and_carrier() -> None:
    avx2_rules = avx2_vector_integer_count_rules(_avx2_descriptor)
    avx512_rules = avx512_vector_integer_count_rules(_avx512_descriptor)
    for rules, descriptor_set in (
        (avx2_rules, X86_AVX2_DESCRIPTOR_SET),
        (avx512_rules, X86_AVX512_CORE_DESCRIPTOR_SET),
    ):
        for rule in rules:
            rule.validate(descriptor_set)

    operations = (vector.vector_ctlzi, vector.vector_cttzi, vector.vector_ctpopi)
    elements = (8, 16, 32, 64)
    actual_avx2 = {
        (
            rule.source_op,
            _guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern.element,
            _lane_bounds(_guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern),
            _guard(rule, GuardKind.LOW_VALUE_REGISTER_CLASS, "input").register_class,
        )
        for rule in _numeric_rules(avx2_rules)
    }
    assert actual_avx2 == {
        (operation, f"i{element}", bounds, register_class)
        for operation in operations
        for element in elements
        for bounds, register_class in (
            ((1, 128 // element), "x86.xmm"),
            ((256 // element, 256 // element), "x86.ymm"),
        )
    }

    actual_avx512 = {
        (
            rule.source_op,
            _guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern.element,
            _lane_bounds(_guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern),
            _guard(rule, GuardKind.LOW_VALUE_REGISTER_CLASS, "input").register_class,
        )
        for rule in _numeric_rules(avx512_rules)
    }
    assert actual_avx512 == {
        (
            operation,
            f"i{element}",
            (512 // element, 512 // element),
            "x86.zmm",
        )
        for operation in operations
        for element in elements
    }


def test_predicate_rules_cover_every_legal_physical_representation() -> None:
    avx2_rules = avx2_vector_integer_count_rules(_avx2_descriptor)
    aliases = tuple(rule for rule in avx2_rules if isinstance(rule, ValueAliasRule))
    assert len(aliases) == 1
    assert aliases[0].source_op is vector.vector_ctpopi
    alias_type = _guard(aliases[0], GuardKind.VALUE_TYPE, "input").type_pattern
    assert (alias_type.minimum_static_elements, alias_type.maximum_static_elements) == (
        2,
        64,
    )

    complement_rules = tuple(
        rule
        for rule in avx2_rules
        if isinstance(rule, DescriptorRule)
        and rule.report_key == "integer_bit_count.predicate.complement"
    )
    actual = {
        (
            rule.source_op,
            _guard(
                rule, GuardKind.VALUE_TYPE, "input"
            ).type_pattern.minimum_static_elements,
            _guard(rule, GuardKind.LOW_VALUE_REGISTER_CLASS, "input").register_class,
            _guard(rule, GuardKind.LOW_VALUE_REPRESENTATION, "input").count,
        )
        for rule in complement_rules
    }
    assert actual == {
        (
            operation,
            lane_count,
            avx2_predicate_register_class(representation_bit_width),
            avx2_predicate_representation_id(lane_count, representation_bit_width),
        )
        for operation in (vector.vector_ctlzi, vector.vector_cttzi)
        for lane_count in (2, 4, 8, 16, 32)
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        )
    }

    avx512_rules = avx512_vector_integer_count_rules(_avx512_descriptor)
    avx512_complements = tuple(
        rule
        for rule in avx512_rules
        if isinstance(rule, DescriptorRule)
        and rule.report_key == "integer_bit_count.predicate.complement"
    )
    assert {
        (
            rule.source_op,
            _guard(rule, GuardKind.LOW_VALUE_REGISTER_CLASS, "input").register_class,
            _guard(rule, GuardKind.LOW_VALUE_REPRESENTATION, "input").count,
            _guard(
                rule, GuardKind.VALUE_TYPE, "input"
            ).type_pattern.maximum_static_elements,
        )
        for rule in avx512_complements
    } == {
        (operation, register_class, representation, maximum_elements)
        for operation in (vector.vector_ctlzi, vector.vector_cttzi)
        for register_class, representation, maximum_elements in (
            ("x86.k", 1, 64),
            ("x86.zmm", 8, 64),
        )
    }


def test_recipes_use_bounded_vector_sequences_and_intern_rule_literals() -> None:
    numeric_rules = (
        *_numeric_rules(avx2_vector_integer_count_rules(_avx2_descriptor)),
        *_numeric_rules(avx512_vector_integer_count_rules(_avx512_descriptor)),
    )
    expected_emit_counts = {
        vector.vector_ctlzi: {8: 21, 16: 22, 32: 26, 64: 30},
        vector.vector_cttzi: {8: 12, 16: 16, 32: 18, 64: 20},
        vector.vector_ctpopi: {8: 8, 16: 12, 32: 14, 64: 16},
    }
    for rule in numeric_rules:
        emits = tuple(emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp))
        assert all(not emit.descriptor.key.startswith("x86.scalar.") for emit in emits)
        element = _guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern.element
        assert len(emits) == expected_emit_counts[rule.source_op][int(element[1:])]

    avx2_rules = _numeric_rules(avx2_vector_integer_count_rules(_avx2_descriptor))

    def find(source_op, element: str, register_class: str) -> DescriptorRule:
        return next(
            rule
            for rule in avx2_rules
            if rule.source_op is source_op
            and _guard(rule, GuardKind.VALUE_TYPE, "input").type_pattern.element
            == element
            and _guard(rule, GuardKind.LOW_VALUE_REGISTER_CLASS, "input").register_class
            == register_class
        )

    nibble_table = bytes((0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)) * 2
    assert _literal_payloads(find(vector.vector_ctpopi, "i8", "x86.ymm")) == (
        nibble_table,
        bytes((0x0F,)) * 32,
    )
    assert _literal_payloads(find(vector.vector_ctlzi, "i8", "x86.ymm")) == (
        bytes((0x7F,)) * 32,
        bytes((0x3F,)) * 32,
        bytes((0x0F,)) * 32,
        nibble_table,
    )
