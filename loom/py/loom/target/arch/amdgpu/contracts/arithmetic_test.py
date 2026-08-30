# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU arithmetic contract source tables."""

from __future__ import annotations

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.arithmetic import (
    AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    LOWER_RULE_FLAG_CONTRACT_ONLY,
    CompiledLowerRuleSet,
    GuardKind,
    LowerRule,
    compile_lower_rule_set,
)


def _compiled_arithmetic_rules() -> CompiledLowerRuleSet:
    return compile_lower_rule_set(
        AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
        dialect_ops=AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    )


def _rules_for_source_op(
    compiled: CompiledLowerRuleSet,
    source_op: Op,
) -> tuple[LowerRule, ...]:
    rules: list[LowerRule] = []
    for span in compiled.spans:
        if span.source_op is source_op:
            rule_end = span.rule_start + span.rule_count
            rules.extend(compiled.rules[span.rule_start : rule_end])
    if not rules:
        raise AssertionError(f"no lower-rule span for {source_op.name}")
    return tuple(rules)


def _rule_descriptor_keys(
    compiled: CompiledLowerRuleSet,
    rule: LowerRule,
) -> tuple[str, ...]:
    return tuple(
        compiled.emits[emit_index].descriptor.key
        for emit_index in range(rule.emit_start, rule.emit_start + rule.emit_count)
    )


def _descriptor_sequence_positions(
    compiled: CompiledLowerRuleSet,
    source_op: Op,
) -> dict[tuple[str, ...], int]:
    positions: dict[tuple[str, ...], int] = {}
    for ordinal, rule in enumerate(_rules_for_source_op(compiled, source_op)):
        descriptor_keys = _rule_descriptor_keys(compiled, rule)
        if descriptor_keys:
            positions.setdefault(descriptor_keys, ordinal)
    return positions


def test_unsigned_bitfield_extract_rules_try_native_bfe_before_shift_mask() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_extractu,
    )

    assert (
        positions[("amdgpu.v_bfe_u32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshrrev_b32.src0_inline",
                "amdgpu.v_and_b32.src0_inline",
            )
        ]
    )
    assert (
        positions[("amdgpu.v_bfe_u32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshrrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
            )
        ]
    )


def test_signed_bitfield_extract_rules_try_native_bfe_before_shift_pair() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_extracts,
    )

    assert (
        positions[("amdgpu.v_bfe_i32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_ashrrev_i32.src0_inline",
            )
        ]
    )


def test_bitfield_insert_rules_try_native_bfi_before_mask_merge_fallback() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_insert,
    )

    native_shift_bfi = (
        "amdgpu.v_lshlrev_b32.src0_inline",
        "amdgpu.v_bfi_b32.src0_lit",
    )
    assert (
        positions[native_shift_bfi]
        < positions[
            (
                "amdgpu.v_and_b32.src0_inline",
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_or_b32",
            )
        ]
    )
    assert (
        positions[native_shift_bfi]
        < positions[
            (
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_or_b32",
            )
        ]
    )


def test_packed_i16_arithmetic_rules_try_native_pk_ops_before_word_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    arithmetic_cases = (
        (vector.vector_addi, "amdgpu.v_pk_add_u16", "amdgpu.v_add_u32"),
        (vector.vector_subi, "amdgpu.v_pk_sub_i16", "amdgpu.v_sub_u32"),
        (vector.vector_muli, "amdgpu.v_pk_mul_lo_u16", "amdgpu.v_mul_lo_u32"),
        (vector.vector_minsi, "amdgpu.v_pk_min_i16", "amdgpu.v_min_i32"),
        (vector.vector_maxsi, "amdgpu.v_pk_max_i16", "amdgpu.v_max_i32"),
        (vector.vector_minui, "amdgpu.v_pk_min_u16", "amdgpu.v_min_u32"),
        (vector.vector_maxui, "amdgpu.v_pk_max_u16", "amdgpu.v_max_u32"),
    )
    for source_op, packed_descriptor, word_descriptor in arithmetic_cases:
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[(packed_descriptor,)] < positions[(word_descriptor,)]

    shift_cases = (
        (vector.vector_shli, "amdgpu.v_pk_lshlrev_b16"),
        (vector.vector_shrsi, "amdgpu.v_pk_ashrrev_i16"),
        (vector.vector_shrui, "amdgpu.v_pk_lshrrev_b16"),
    )
    for source_op, packed_descriptor in shift_cases:
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[(packed_descriptor,)] == 0


def test_packed_bf16_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    add_positions = _descriptor_sequence_positions(compiled, vector.vector_addf)
    assert (
        add_positions[("amdgpu.v_pk_add_bf16",)] < add_positions[("amdgpu.v_add_f32",)]
    )

    mul_positions = _descriptor_sequence_positions(compiled, vector.vector_mulf)
    assert (
        mul_positions[("amdgpu.v_pk_mul_bf16",)] < mul_positions[("amdgpu.v_mul_f32",)]
    )

    fma_positions = _descriptor_sequence_positions(compiled, vector.vector_fmaf)
    assert (
        fma_positions[("amdgpu.v_pk_fma_bf16",)] < fma_positions[("amdgpu.v_fma_f32",)]
    )


def test_packed_f16_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    arithmetic_cases = (
        (vector.vector_addf, "amdgpu.v_pk_add_f16", "amdgpu.v_add_f32"),
        (vector.vector_mulf, "amdgpu.v_pk_mul_f16", "amdgpu.v_mul_f32"),
        (vector.vector_minnumf, "amdgpu.v_pk_minnum_f16", "amdgpu.v_min_f32"),
        (vector.vector_maxnumf, "amdgpu.v_pk_maxnum_f16", "amdgpu.v_max_f32"),
    )
    for source_op, packed_descriptor, scalar_descriptor in arithmetic_cases:
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[(packed_descriptor,)] < positions[(scalar_descriptor,)]

    minimum_positions = _descriptor_sequence_positions(compiled, vector.vector_minimumf)
    assert minimum_positions[("amdgpu.v_pk_minimum_f16",)] == 0

    maximum_positions = _descriptor_sequence_positions(compiled, vector.vector_maximumf)
    assert maximum_positions[("amdgpu.v_pk_maximum_f16",)] == 0


def test_ieee_minmax_rules_publish_direct_scalar_and_vector_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    for source_op, descriptors in (
        (
            scalar_arithmetic.scalar_minimumf,
            {
                "amdgpu.v_minimum_f16",
                "amdgpu.v_minimum_f32",
                "amdgpu.v_minimum_f64",
            },
        ),
        (
            scalar_arithmetic.scalar_maximumf,
            {
                "amdgpu.v_maximum_f16",
                "amdgpu.v_maximum_f32",
                "amdgpu.v_maximum_f64",
            },
        ),
        (
            vector.vector_minimumf,
            {
                "amdgpu.v_pk_minimum_f16",
                "amdgpu.v_minimum_f32",
                "amdgpu.v_minimum_f64",
            },
        ),
        (
            vector.vector_maximumf,
            {
                "amdgpu.v_pk_maximum_f16",
                "amdgpu.v_maximum_f32",
                "amdgpu.v_maximum_f64",
            },
        ),
    ):
        published_descriptors = {
            descriptor
            for sequence in _descriptor_sequence_positions(compiled, source_op)
            for descriptor in sequence
        }
        assert descriptors <= published_descriptors


def test_clamp_rules_publish_distinct_number_and_ieee_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    scalar_positions = _descriptor_sequence_positions(
        compiled, scalar_arithmetic.scalar_clampf
    )
    vector_positions = _descriptor_sequence_positions(compiled, vector.vector_clampf)
    assert ("amdgpu.v_maxmin_num_f16",) in scalar_positions
    assert ("amdgpu.v_maximumminimum_f16",) in scalar_positions
    for positions in (scalar_positions, vector_positions):
        assert ("amdgpu.v_maxmin_num_f32",) in positions
        assert ("amdgpu.v_maximumminimum_f32",) in positions
    assert (
        "amdgpu.v_pk_maxnum_f16",
        "amdgpu.v_pk_minnum_f16",
    ) in vector_positions
    assert (
        "amdgpu.v_pk_maximum_f16",
        "amdgpu.v_pk_minimum_f16",
    ) in vector_positions


def test_packed_f32_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _compiled_arithmetic_rules()

    for source_op, packed_descriptor, scalar_descriptor in (
        (vector.vector_addf, "amdgpu.v_pk_add_f32", "amdgpu.v_add_f32"),
        (vector.vector_mulf, "amdgpu.v_pk_mul_f32", "amdgpu.v_mul_f32"),
    ):
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[(packed_descriptor,)] < positions[(scalar_descriptor,)]


def test_vector_extract_rules_publish_contract_only_shape_rows() -> None:
    compiled = _compiled_arithmetic_rules()
    rules = _rules_for_source_op(compiled, vector.vector_extract)
    contract_rules = tuple(
        rule for rule in rules if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
    )

    assert len(contract_rules) == 14
    for rule in contract_rules:
        assert rule.emit_count == 0
        guard_kinds = tuple(
            compiled.guards[guard_index].kind
            for guard_index in range(
                rule.guard_start,
                rule.guard_start + rule.guard_count,
            )
        )
        assert GuardKind.VECTOR_EXTRACT_SHAPE in guard_kinds


def test_vector_construct_rules_publish_contract_only_storage_rows() -> None:
    compiled = _compiled_arithmetic_rules()

    expected_rule_counts = {
        vector.vector_from_elements: 12,
        vector.vector_iota: 2,
        vector.vector_insert: 6,
        vector.vector_splat: 11,
    }
    for source_op, expected_rule_count in expected_rule_counts.items():
        rules = _rules_for_source_op(compiled, source_op)
        contract_rules = tuple(
            rule for rule in rules if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
        )

        assert len(contract_rules) == expected_rule_count
        assert all(rule.emit_count == 0 for rule in contract_rules)


def test_vector_packed_float_conversion_rules_publish_contract_only_shape_rows() -> (
    None
):
    compiled = _compiled_arithmetic_rules()

    expected_rule_counts = {
        vector.vector_extf: 8,
        vector.vector_fptrunc: 8,
    }
    for source_op, expected_rule_count in expected_rule_counts.items():
        rules = _rules_for_source_op(compiled, source_op)
        contract_rules = tuple(
            rule for rule in rules if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
        )

        assert len(contract_rules) == expected_rule_count
        for rule in contract_rules:
            assert rule.emit_count == 0
            guard_kinds = tuple(
                compiled.guards[guard_index].kind
                for guard_index in range(
                    rule.guard_start,
                    rule.guard_start + rule.guard_count,
                )
            )
            assert GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ in guard_kinds
