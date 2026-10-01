# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU contract source tables."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.arithmetic import (
    AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
)
from loom.target.arch.amdgpu.contracts.integer import (
    AMDGPU_INTEGER_CONTRACT_DIALECT_OPS,
    AMDGPU_INTEGER_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    LOWER_RULE_FLAG_CONTRACT_ONLY,
    CompiledLowerRuleSet,
    GuardKind,
    LowerRule,
    SourceValueKind,
    TypePattern,
    compile_lower_rule_set,
)


def _compiled_arithmetic_rules() -> CompiledLowerRuleSet:
    return compile_lower_rule_set(
        AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
        dialect_ops=AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    )


def _compiled_integer_rules() -> CompiledLowerRuleSet:
    return compile_lower_rule_set(
        AMDGPU_INTEGER_CONTRACT_FRAGMENT,
        dialect_ops=AMDGPU_INTEGER_CONTRACT_DIALECT_OPS,
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


def _rule_type_patterns(
    compiled: CompiledLowerRuleSet,
    rule: LowerRule,
) -> tuple[TypePattern, ...]:
    return tuple(
        compiled.type_patterns[guard.type_pattern_index].type_pattern
        for guard in compiled.guards[
            rule.guard_start : rule.guard_start + rule.guard_count
        ]
        if guard.kind == GuardKind.VALUE_TYPE
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


def test_index_madd_rules_accept_wrapping_carrier_results() -> None:
    for compiled in (_compiled_arithmetic_rules(), _compiled_integer_rules()):
        for rule in _rules_for_source_op(compiled, index.index_madd):
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            for guard in guards:
                if guard.kind not in (
                    GuardKind.VALUE_SIGNED_BIT_COUNT,
                    GuardKind.VALUE_UNSIGNED_BIT_COUNT,
                ):
                    continue
                value_ref = compiled.value_refs[guard.value_ref_index]
                assert not (
                    value_ref.kind == SourceValueKind.RESULT
                    and value_ref.name == "result"
                )


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


def test_f32_copysign_rules_try_literal_bfi_before_register_mask() -> None:
    compiled = _compiled_arithmetic_rules()

    for source_op in (
        scalar_arithmetic.scalar_copysignf,
        vector.vector_copysignf,
    ):
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert (
            positions[("amdgpu.v_bfi_b32.src0_lit",)]
            < positions[("amdgpu.s_mov_b32", "amdgpu.v_bfi_b32")]
        )


def test_integer_extrema_rules_prefer_encoded_constants() -> None:
    compiled = _compiled_integer_rules()

    for source_op, suffix in (
        (scalar_arithmetic.scalar_minsi, "min_i32"),
        (scalar_arithmetic.scalar_maxsi, "max_i32"),
        (scalar_arithmetic.scalar_minui, "min_u32"),
        (scalar_arithmetic.scalar_maxui, "max_u32"),
    ):
        descriptor_sequences = tuple(
            _rule_descriptor_keys(compiled, rule)
            for rule in _rules_for_source_op(compiled, source_op)
        )
        assert descriptor_sequences == (
            (f"amdgpu.s_{suffix}.rhs_inline",),
            (f"amdgpu.s_{suffix}.rhs_inline",),
            (f"amdgpu.s_{suffix}.lit",),
            (f"amdgpu.s_{suffix}.lit",),
            (f"amdgpu.v_{suffix}.src0_inline",),
            (f"amdgpu.v_{suffix}.src0_inline",),
            (f"amdgpu.v_{suffix}.lit",),
            (f"amdgpu.v_{suffix}.lit",),
            (f"amdgpu.s_{suffix}",),
            (f"amdgpu.v_{suffix}",),
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


def test_32bit_vector_shape_contracts_match_lane_semantics() -> None:
    compiled = _compiled_arithmetic_rules()
    rank1_i32_source_ops: set[Op] = set()
    rank1_f32_descriptors: set[str] = set()
    static_i32_source_ops: set[Op] = set()
    static_f32_source_ops: set[Op] = set()

    for rule in compiled.rules:
        for type_pattern in set(_rule_type_patterns(compiled, rule)):
            if type_pattern.kind != "vector":
                continue
            if type_pattern.minimum_lanes is not None:
                if type_pattern.element == "i32":
                    rank1_i32_source_ops.add(rule.source_op)
                    assert rule.emit_count == 0
                elif type_pattern.element == "f32":
                    descriptor_keys = _rule_descriptor_keys(compiled, rule)
                    assert descriptor_keys
                    rank1_f32_descriptors.update(descriptor_keys)
            elif type_pattern.minimum_static_elements is not None:
                if type_pattern.element == "i32":
                    static_i32_source_ops.add(rule.source_op)
                elif type_pattern.element == "f32":
                    static_f32_source_ops.add(rule.source_op)

    assert rank1_i32_source_ops == {
        vector.vector_bitpack,
        vector.vector_bitunpacks,
        vector.vector_bitunpacku,
    }
    assert rank1_f32_descriptors == {
        "amdgpu.v_pk_add_f32",
        "amdgpu.v_pk_fma_f32",
        "amdgpu.v_pk_mul_f32",
    }
    assert {
        vector.vector_addi,
        vector.vector_bitfield_extractu,
        vector.vector_fptosi,
        vector.vector_sitofp,
    } <= static_i32_source_ops
    assert {
        vector.vector_addf,
        vector.vector_clampf,
        vector.vector_divf,
        vector.vector_exp2f,
        vector.vector_fmaf,
        vector.vector_mulf,
    } <= static_f32_source_ops


def test_vector_extract_rules_publish_contract_only_shape_rows() -> None:
    compiled = _compiled_arithmetic_rules()
    rules = _rules_for_source_op(compiled, vector.vector_extract)
    contract_rules = tuple(
        rule for rule in rules if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
    )

    assert len(contract_rules) == 16
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
        vector.vector_insert: 8,
        vector.vector_splat: 11,
    }
    for source_op, expected_rule_count in expected_rule_counts.items():
        rules = _rules_for_source_op(compiled, source_op)
        contract_rules = tuple(
            rule for rule in rules if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
        )

        assert len(contract_rules) == expected_rule_count
        assert all(rule.emit_count == 0 for rule in contract_rules)

    for source_op in (vector.vector_from_elements, vector.vector_splat):
        for rule in _rules_for_source_op(compiled, source_op):
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            assert len(guards) == 1
            assert guards[0].kind == GuardKind.VALUE_TYPE
            value_ref = compiled.value_refs[guards[0].value_ref_index]
            assert value_ref.kind == SourceValueKind.RESULT


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


def test_vector_integer_conversion_contracts_preserve_storage_and_lane_counts() -> None:
    compiled = _compiled_arithmetic_rules()
    widening = {("i8", "i16"), ("i8", "i32"), ("i16", "i32")}
    narrowing = {
        ("i16", "i8"),
        ("i32", "i8"),
        ("i32", "i16"),
        ("i64", "i8"),
        ("i64", "i16"),
        ("i64", "i32"),
    }
    to_float = {("i8", "f32"), ("i16", "f32"), ("i32", "f32")}
    from_float = {(result, source) for source, result in to_float}
    for source_op, expected_pairs in (
        (vector.vector_extsi, widening),
        (vector.vector_extui, widening),
        (vector.vector_trunci, narrowing),
        (vector.vector_sitofp, to_float),
        (vector.vector_uitofp, to_float),
        (vector.vector_fptosi, from_float),
        (vector.vector_fptoui, from_float),
    ):
        pairs = set()
        for rule in _rules_for_source_op(compiled, source_op):
            if not rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY:
                continue
            assert rule.emit_count == 0
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            assert any(
                guard.kind == GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
                for guard in guards
            )
            types = tuple(
                compiled.type_patterns[guard.type_pattern_index].type_pattern
                for guard in guards
                if guard.kind == GuardKind.VALUE_TYPE
            )
            assert len(types) == 2
            for type_pattern in types:
                assert type_pattern.kind == "vector"
                assert (
                    type_pattern.minimum_lanes == 1
                    or type_pattern.minimum_static_elements == 1
                )
            pairs.add(tuple(type_pattern.element for type_pattern in types))
        assert pairs == expected_pairs


def test_bitunpack_contract_preserves_packed_result_capacity() -> None:
    compiled = _compiled_arithmetic_rules()

    for source_op in (vector.vector_bitunpacku, vector.vector_bitunpacks):
        maximum_lane_counts: list[int] = []
        for rule in _rules_for_source_op(compiled, source_op):
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            maximum_lane_counts.extend(
                guard.maximum_i64
                for guard in guards
                if guard.kind == GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD
            )
        assert sorted(maximum_lane_counts) == [32, 64]
