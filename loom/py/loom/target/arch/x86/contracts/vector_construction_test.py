# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    INTEGER_ELEMENTS,
    X86_VECTOR_STORAGE_FAMILIES,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    Scalar,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import EncodingFieldValue, OperandAddressMapKind


def _value_type_guard(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def test_avx2_iotas_cover_every_integer_element_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(
                element.name,
                lanes=vector_bit_width // element.bit_width,
            ),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS
        for priority in (0, 1)
        if vector_bit_width >= element.bit_width
        if vector_bit_width != element.bit_width or priority == 0
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_avx512_iotas_cover_every_zmm_integer_element() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(element.name, lanes=512 // element.bit_width),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for priority in (0, 1)
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_concat_covers_equal_packets_in_every_storage_family() -> None:
    for fragment, widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, (8, 16, 32, 64, 128)),
        (X86_AVX512_CONTRACT_FRAGMENT, (256,)),
    ):
        rules = [
            case
            for case in fragment.cases
            if isinstance(case, DescriptorRule)
            and case.source_op is vector.vector_concat
        ]
        assert {
            (_value_type_guard(rule, "inputs"), _value_type_guard(rule, "result"))
            for rule in rules
        } == {
            (
                Vector(family.element_names, lanes=width // family.element_bit_width),
                Vector(
                    family.element_names, lanes=2 * width // family.element_bit_width
                ),
            )
            for family in X86_VECTOR_STORAGE_FAMILIES
            for width in widths
            if width >= family.element_bit_width
        }
        for rule in rules:
            assert any(
                guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
                and guard.field == "inputs"
                and guard.count == 2
                for guard in rule.guards
            )
            assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
            assert len(rule.emit) == 1
            inputs = _value_type_guard(rule, "inputs")
            family = next(
                family
                for family in X86_VECTOR_STORAGE_FAMILIES
                if family.element_names == inputs.elements
            )
            input_bit_width = inputs.lanes * family.element_bit_width
            if input_bit_width < 128:
                continue
            assert rule.emit[0].operands == {
                "low": ValueRef.operand("inputs", element=0),
                "high": ValueRef.operand("inputs", element=1),
            }
            assert rule.emit[0].immediates == {}


def test_iota_construction_keeps_shared_halves_in_pure_descriptor_operations() -> None:
    for fragment in (X86_AVX2_CONTRACT_FRAGMENT, X86_AVX512_CONTRACT_FRAGMENT):
        for rule in fragment.cases:
            if (
                not isinstance(rule, DescriptorRule)
                or rule.source_op is not vector.vector_iota
            ):
                continue
            assert all(isinstance(emit, EmitDescriptorOp) for emit in rule.emit)


def test_half_descriptors_preserve_read_footprints_and_native_register_domains() -> (
    None
):
    for descriptors, prefix, mnemonic, wide, narrow, register_count in (
        (X86_AVX2_DESCRIPTOR_SET, "avx2", "vinsertf128", "ymm", "xmm", 16),
        (X86_AVX512_CORE_DESCRIPTOR_SET, "avx512", "vinserti64x4", "zmm", "ymm", 0),
    ):
        rows = {row.key: row for row in descriptors.descriptors}
        insert = rows[f"x86.{prefix}.{mnemonic}.{wide}.{narrow}"]
        concat = rows[f"x86.{prefix}.{mnemonic}.concat.{wide}"]
        assert concat.encoding_format_id == insert.encoding_format_id
        assert concat.encoding_id == insert.encoding_id
        assert concat.encoding_field_values == (EncodingFieldValue(1, 1),)
        assert concat.immediates == ()
        assert concat.feature_mask_words == insert.feature_mask_words == ()
        for row, classes in (
            (insert, (wide, wide, narrow)),
            (concat, (wide, narrow, narrow)),
        ):
            assert tuple(
                operand.reg_alts[0].reg_class for operand in row.operands
            ) == tuple("x86." + name for name in classes)
            assert all(
                operand.addressable_unit_count == register_count
                and operand.address_map_kind
                is (
                    OperandAddressMapKind.LOW_SUBSET
                    if register_count
                    else OperandAddressMapKind.DIRECT
                )
                for operand in row.operands
            )
