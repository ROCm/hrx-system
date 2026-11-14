# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.descriptors import X86_AVX512_CORE_DESCRIPTOR_SET
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VBMI, FEATURE_AVX512_VL
from loom.target.arch.x86.vector_families import X86_VECTOR_STORAGE_FAMILIES
from loom.target.contracts import DescriptorRule, GuardKind
from loom.target.low_descriptors import ConstraintKind


def _rules(fragment, source_op):
    return tuple(
        rule
        for rule in fragment.cases
        if isinstance(rule, DescriptorRule) and rule.source_op is source_op
    )


def _type(rule, field):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def test_native_width_matrix_preserves_exact_source_and_result_shapes() -> None:
    for fragment, widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, (16, 32, 64, 128, 256)),
        (X86_AVX512_CONTRACT_FRAGMENT, (16, 32, 64, 128, 256, 512)),
    ):
        for source_op, full_name in (
            (vector.vector_interleave, "result"),
            (vector.vector_deinterleave, "source"),
        ):
            expected = {
                (family.element_names, width // family.element_bit_width)
                for family in X86_VECTOR_STORAGE_FAMILIES
                for width in widths
                if width >= 2 * family.element_bit_width
                if not (
                    fragment is X86_AVX512_CONTRACT_FRAGMENT
                    and source_op is vector.vector_deinterleave
                    and family.element_bit_width == 64
                    and width < 512
                )
            }
            rules = _rules(fragment, source_op)
            assert {
                (_type(rule, full_name).elements, _type(rule, full_name).lanes)
                for rule in rules
            } == expected
            for rule in rules:
                full = _type(rule, full_name)
                for name in ("even", "odd"):
                    half = _type(rule, name)
                    assert half.elements == full.elements
                    assert half.lanes * 2 == full.lanes
                assert any(
                    guard.kind is GuardKind.I64_RANGE
                    and guard.field == "axis"
                    and guard.minimum == guard.maximum == 0
                    for guard in rule.guards
                )


def test_two_table_controls_never_read_the_unpopulated_upper_halves() -> None:
    count = 0
    for rule in _rules(X86_AVX512_CONTRACT_FRAGMENT, vector.vector_interleave):
        if not rule.descriptor.mnemonic.startswith("vpermt2"):
            continue
        pattern = _type(rule, "result")
        element_bits = next(
            family.element_bit_width
            for family in X86_VECTOR_STORAGE_FAMILIES
            if family.element_names == pattern.elements
        )
        literal = rule.emit[0].immediates["data"].data
        element_bytes = element_bits // 8
        indices = tuple(
            int.from_bytes(literal[start : start + element_bytes], "little")
            for start in range(0, len(literal), element_bytes)
        )
        assert indices == tuple(
            index
            for lane in range(pattern.lanes // 2)
            for index in (lane, pattern.lanes + lane)
        )
        count += 1
    assert count == 8


def test_byte_zip_keeps_a_base_feature_composition() -> None:
    rules = tuple(
        rule
        for rule in _rules(X86_AVX512_CONTRACT_FRAGMENT, vector.vector_interleave)
        if _type(rule, "result").elements[0] == "i8"
        and _type(rule, "result").lanes == 64
    )
    assert len(rules) == 2
    base, native = sorted(rules, key=lambda rule: rule.priority)
    assert all(not emit.descriptor.feature_mask_words for emit in base.emit)
    assert native.descriptor.feature_mask_words == (FEATURE_AVX512_VBMI,)


def test_native_permutation_domains_and_destructive_operands() -> None:
    count = 0
    for row in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors:
        if not row.mnemonic.startswith(("vpunpck", "vpermi2", "vpermt2")):
            continue
        evex = row.key.startswith("x86.avx512.")
        suffix = row.key.rsplit(".", 1)[1]
        expected_features = FEATURE_AVX512_VL if evex and suffix != "zmm" else 0
        if row.mnemonic in ("vpermi2b", "vpermt2b"):
            expected_features |= FEATURE_AVX512_VBMI
        assert row.feature_mask_words == (
            (expected_features,) if expected_features else ()
        )
        assert all(
            operand.addressable_unit_count == (0 if evex else 16)
            for operand in row.operands
        )
        if row.mnemonic.startswith("vperm"):
            assert tuple(operand.field_name for operand in row.operands) == (
                ("dst", "control", "lhs", "rhs")
                if row.mnemonic.startswith("vpermi")
                else ("dst", "lhs", "control", "rhs")
            )
            assert {
                (
                    constraint.kind,
                    constraint.lhs_operand_index,
                    constraint.rhs_operand_index,
                )
                for constraint in row.constraints
            } == {
                (ConstraintKind.TIED, 0, 1),
                (ConstraintKind.DESTRUCTIVE, 0, 1),
            }
        count += 1
    assert count == 64
