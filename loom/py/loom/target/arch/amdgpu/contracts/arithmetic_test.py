# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU contract source tables."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import defs as scalar_defs
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.scalar_type import ScalarTypeKind, scalar_type_name
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
    DescriptorOperandMaterialization,
    GuardKind,
    LowerAttrCopyKind,
    LowerEmitKind,
    LowerRule,
    Scalar,
    SourceValueKind,
    TypePattern,
    UnsignedDivisorMagicKind,
    compile_lower_rule_set,
)

_AFN_FASTMATH_FLAG = scalar_defs.FastMathFlags.case("afn").value


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


def test_index_bit_operations_cover_scalar_and_vector_registers() -> None:
    compiled = _compiled_integer_rules()
    for source_op, scalar_sequence, vector_sequence in (
        (
            index.index_ctlzi,
            ("s_clz_i32_u32", "s_mov_b32", "s_min_u32"),
            ("v_clz_i32_u32", "v_mov_b32", "v_min_u32"),
        ),
        (
            index.index_cttzi,
            ("s_ctz_i32_b32", "s_mov_b32", "s_min_u32"),
            ("v_ctz_i32_b32", "v_mov_b32", "v_min_u32"),
        ),
        (index.index_ctpopi, ("s_bcnt1_i32_b32",), ("v_bcnt_u32_b32.src1_zero",)),
        (
            index.index_rotli,
            ("s_mov_b32", "s_sub_u32", "s_lshl_b32", "s_lshr_b32", "s_or_b32"),
            ("v_mov_b32", "v_sub_u32", "v_alignbit_b32"),
        ),
        (
            index.index_rotri,
            ("s_mov_b32", "s_sub_u32", "s_lshr_b32", "s_lshl_b32", "s_or_b32"),
            ("v_alignbit_b32",),
        ),
    ):
        rules = _rules_for_source_op(compiled, source_op)
        assert tuple(_rule_descriptor_keys(compiled, rule) for rule in rules) == (
            tuple(f"amdgpu.{key}" for key in scalar_sequence),
            tuple(f"amdgpu.{key}" for key in vector_sequence),
        )
        for rule in rules:
            assert set(_rule_type_patterns(compiled, rule)) == {Scalar("index")}


def test_constant_unsigned_quotients_cover_both_register_classes() -> None:
    compiled = _compiled_integer_rules()
    scalar_rules = _rules_for_source_op(compiled, scalar_arithmetic.scalar_divui)
    scalar_sequences = tuple(
        _rule_descriptor_keys(compiled, rule) for rule in scalar_rules
    )
    index_sequences = tuple(
        _rule_descriptor_keys(compiled, rule)
        for rule in _rules_for_source_op(compiled, index.index_div)
        if any(
            key.endswith("mul_hi_u32") for key in _rule_descriptor_keys(compiled, rule)
        )
    )
    # Every reciprocal shape shares the index implementation, with scalar i32
    # materialization instead of nonnegative address-value materialization.
    magic_count = 2 * len(UnsignedDivisorMagicKind)
    assert len(index_sequences) == magic_count
    assert scalar_sequences[:magic_count] == index_sequences
    assert index_sequences[:2] == (
        ("amdgpu.s_mov_b32", "amdgpu.s_mul_hi_u32"),
        ("amdgpu.v_mov_b32", "amdgpu.v_mul_hi_u32"),
    )
    assert scalar_sequences[magic_count:] == (
        (
            "amdgpu.s_cmp_ge_u32",
            "amdgpu.s_mov_b32",
            "amdgpu.s_mov_b32",
            "amdgpu.s_cselect_b32",
        ),
        (
            "amdgpu.v_cmp_uge_u32",
            "amdgpu.v_mov_b32",
            "amdgpu.v_mov_b32",
            "amdgpu.v_cndmask_b32",
        ),
    )
    for rule in scalar_rules:
        assert set(_rule_type_patterns(compiled, rule)) == {Scalar("i32")}


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


def test_scalar_bitfield_controls_use_isa_offset_and_width_fields() -> None:
    compiled = _compiled_integer_rules()
    for source_op, descriptor_key in (
        (scalar_bitwise.scalar_bitfield_extractu, "amdgpu.s_bfe_u32.lit"),
        (scalar_bitwise.scalar_bitfield_extracts, "amdgpu.s_bfe_i32.lit"),
    ):
        matching_rules = [
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if _rule_descriptor_keys(compiled, rule) == (descriptor_key,)
        ]
        assert len(matching_rules) == 1
        emit = compiled.emits[matching_rules[0].emit_start]
        assert emit.attr_copy_count == 1
        control = compiled.attr_copies[emit.attr_copy_start]
        assert control.kind == LowerAttrCopyKind.ATTRS_PACK_CONSECUTIVE
        assert control.source_attr_index == 0
        assert control.source_element_count == 2
        assert control.source_element_bit_width == 16


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


def test_scalar_i32_bitwise_rules_prefer_encoded_constants() -> None:
    compiled = _compiled_integer_rules()

    for source_op, suffix in (
        (scalar_bitwise.scalar_andi, "and_b32"),
        (scalar_bitwise.scalar_ori, "or_b32"),
        (scalar_bitwise.scalar_xori, "xor_b32"),
    ):
        descriptor_sequences = tuple(
            _rule_descriptor_keys(compiled, rule)
            for rule in _rules_for_source_op(compiled, source_op)
            if set(_rule_type_patterns(compiled, rule)) == {Scalar("i32")}
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


def test_mixed_scalar_bool_mask_rules_precede_mask_materialization() -> None:
    compiled = _compiled_integer_rules()

    for source_op, direct_sequences, fallback in (
        (
            scalar_bitwise.scalar_andi,
            (
                ("amdgpu.s_mov_b32", "amdgpu.s_cselect_b32"),
                (
                    "amdgpu.s_mov_b32",
                    "amdgpu.s_cmp_lg_i32.src1_inline",
                    "amdgpu.s_cselect_b32",
                ),
            ),
            ("amdgpu.s_and_b64",),
        ),
        (
            scalar_bitwise.scalar_ori,
            (
                ("amdgpu.s_mov_b64_exec_read", "amdgpu.s_cselect_b32"),
                (
                    "amdgpu.s_mov_b64_exec_read",
                    "amdgpu.s_cmp_lg_i32.src1_inline",
                    "amdgpu.s_cselect_b32",
                ),
            ),
            ("amdgpu.s_or_b64",),
        ),
        (
            scalar_bitwise.scalar_xori,
            (
                (
                    "amdgpu.s_mov_b64_exec_read",
                    "amdgpu.s_xor_b64",
                    "amdgpu.s_cmp_lg_i32.src1_inline",
                    "amdgpu.s_cselect_b32",
                ),
            ),
            ("amdgpu.s_xor_b64",),
        ),
    ):
        sequences = tuple(
            _rule_descriptor_keys(compiled, rule)
            for rule in _rules_for_source_op(compiled, source_op)
        )
        fallback_position = sequences.index(fallback)
        for sequence in direct_sequences:
            matching_positions = tuple(
                index
                for index, candidate in enumerate(sequences)
                if candidate == sequence
            )
            assert len(matching_positions) == 2
            assert all(index < fallback_position for index in matching_positions)


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
        assert positions[(packed_descriptor,)] == 3


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


def test_f32_literal_fma_product_flushing_forms_require_afn() -> None:
    compiled = _compiled_arithmetic_rules()
    product_flushing_key = "amdgpu.v_fmamk_f32.flush_product"

    for source_op in (scalar_math.scalar_fmaf, vector.vector_fmaf):
        rules = _rules_for_source_op(compiled, source_op)
        product_flushing_rules = tuple(
            rule
            for rule in rules
            if product_flushing_key in _rule_descriptor_keys(compiled, rule)
        )
        assert len(product_flushing_rules) == 4
        for rule in product_flushing_rules:
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            assert any(
                guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
                and guard.u64 == _AFN_FASTMATH_FLAG
                for guard in guards
            )

        exact_fmamk_rules = tuple(
            rule
            for rule in rules
            if _rule_descriptor_keys(compiled, rule) == ("amdgpu.v_fmamk_f32",)
        )
        assert len(exact_fmamk_rules) == 4
        for rule in exact_fmamk_rules:
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            assert all(
                guard.kind != GuardKind.INSTANCE_FLAGS_HAS_ALL for guard in guards
            )

        fmaak_rules = tuple(
            rule
            for rule in rules
            if _rule_descriptor_keys(compiled, rule) == ("amdgpu.v_fmaak_f32",)
        )
        assert len(fmaak_rules) == 4
        exact_fmaak_rule_count = 0
        relaxed_fmaak_rule_count = 0
        for rule in fmaak_rules:
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            requires_afn = any(
                guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
                and guard.u64 == _AFN_FASTMATH_FLAG
                for guard in guards
            )
            if requires_afn:
                relaxed_fmaak_rule_count += 1
                continue
            exact_fmaak_rule_count += 1
            assert any(
                guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
                and guard.descriptor is not None
                and guard.descriptor.key == "amdgpu.v_fmamk_f32"
                for guard in guards
            )
        assert exact_fmaak_rule_count == 2
        assert relaxed_fmaak_rule_count == 2

        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[("amdgpu.v_fmaak_f32",)] < positions[("amdgpu.v_fma_f32",)]
        assert (
            positions[("amdgpu.v_fmamk_f32.flush_product",)]
            < positions[("amdgpu.v_fma_f32",)]
        )

        general_rules = tuple(
            rule
            for rule in rules
            if _rule_descriptor_keys(compiled, rule) == ("amdgpu.v_fma_f32",)
        )
        assert len(general_rules) == 1
        general_emit = compiled.emits[general_rules[0].emit_start]
        assert (
            general_emit.operand_materialization
            is DescriptorOperandMaterialization.TARGET
        )
        general_guards = compiled.guards[
            general_rules[0].guard_start : (
                general_rules[0].guard_start + general_rules[0].guard_count
            )
        ]
        assert all(
            guard.kind
            not in (
                GuardKind.LOW_VALUE_REGISTER_CLASS,
                GuardKind.VALUE_MATERIALIZABLE,
            )
            for guard in general_guards
        )


def test_exact_f32_division_uses_scaled_fmaak_product() -> None:
    compiled = _compiled_arithmetic_rules()

    for source_op in (scalar_arithmetic.scalar_divf, vector.vector_divf):
        exact_rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if "amdgpu.v_div_fixup_f32" in _rule_descriptor_keys(compiled, rule)
        )
        assert len(exact_rules) == 1
        descriptor_keys = _rule_descriptor_keys(compiled, exact_rules[0])
        assert "amdgpu.v_fmaak_f32" in descriptor_keys


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
    assert minimum_positions[("amdgpu.v_pk_minimum_f16.broadcast_lhs_rhs",)] == 0
    assert minimum_positions[("amdgpu.v_pk_minimum_f16",)] == 3

    maximum_positions = _descriptor_sequence_positions(compiled, vector.vector_maximumf)
    assert maximum_positions[("amdgpu.v_pk_maximum_f16.broadcast_lhs_rhs",)] == 0
    assert maximum_positions[("amdgpu.v_pk_maximum_f16",)] == 3


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
        for suffix in ("lhs", "rhs", "lhs_rhs"):
            assert (
                positions[(f"{packed_descriptor}.broadcast_{suffix}",)]
                < positions[(packed_descriptor,)]
            )
        assert positions[(packed_descriptor,)] < positions[(scalar_descriptor,)]


def test_packed_f32_broadcast_rules_materialize_before_packet_repetition() -> None:
    compiled = _compiled_arithmetic_rules()
    for source_op in (vector.vector_addf, vector.vector_mulf):
        rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if "_f32.broadcast_" in _rule_descriptor_keys(compiled, rule)[0]
        )
        assert len(rules) == 3
        for rule in rules:
            emit = compiled.emits[rule.emit_start]
            assert emit.kind is LowerEmitKind.DESCRIPTOR_OP_PER_LANE
            assert rule.emit_count == 1
            assert emit.operand_ref_count == 2
            for operand_index, name in enumerate(("lhs", "rhs")):
                reference = compiled.value_refs[emit.operand_ref_start + operand_index]
                broadcast = name in emit.descriptor.key.split(".broadcast_")[1]
                assert reference.kind is (
                    SourceValueKind.UNIFORM_ELEMENT_ORIGIN_OPERAND
                    if broadcast
                    else SourceValueKind.OPERAND
                )
                assert bool(reference.materializer_index) == (operand_index == 1)


def test_packed_narrow_binary_broadcasts_cover_operations_and_source_roles() -> None:
    compiled = _compiled_arithmetic_rules()
    families = (
        (vector.vector_addf, "add_f16"),
        (vector.vector_subf, "add_f16"),
        (vector.vector_mulf, "mul_f16"),
        (vector.vector_minnumf, "minnum_f16"),
        (vector.vector_maxnumf, "maxnum_f16"),
        (vector.vector_minimumf, "minimum_f16"),
        (vector.vector_maximumf, "maximum_f16"),
        (vector.vector_addf, "add_bf16"),
        (vector.vector_subf, "add_bf16"),
        (vector.vector_mulf, "mul_bf16"),
        (vector.vector_addi, "add_u16"),
        (vector.vector_subi, "sub_i16"),
        (vector.vector_muli, "mul_lo_u16"),
        (vector.vector_minsi, "min_i16"),
        (vector.vector_maxsi, "max_i16"),
        (vector.vector_minui, "min_u16"),
        (vector.vector_maxui, "max_u16"),
        (vector.vector_shli, "lshlrev_b16"),
        (vector.vector_shrsi, "ashrrev_i16"),
        (vector.vector_shrui, "lshrrev_b16"),
    )
    for source_op, descriptor_name in families:
        masks = set()
        for rule in _rules_for_source_op(compiled, source_op):
            keys = _rule_descriptor_keys(compiled, rule)
            if (
                len(keys) != 1
                or keys[0].split(".broadcast_")[0] != f"amdgpu.v_pk_{descriptor_name}"
            ):
                continue
            emit = compiled.emits[rule.emit_start]
            assert emit.kind is LowerEmitKind.DESCRIPTOR_OP_PER_LANE
            attributes = {
                attribute.target_name: attribute.literal_i64
                for attribute in compiled.attr_copies[
                    emit.attr_copy_start : emit.attr_copy_start + emit.attr_copy_count
                ]
            }
            source_names = (
                ("shift", "value") if "rev_" in descriptor_name else ("lhs", "rhs")
            )
            suffix = keys[0].partition(".broadcast_")[2].split("_")
            mask = sum(
                1 << bit for bit, name in enumerate(source_names) if name in suffix
            )
            masks.add(mask)
            if source_op is vector.vector_subf:
                assert attributes["neg_lo"] == attributes["neg_hi"] == 2
            for position in range(2):
                reference = compiled.value_refs[emit.operand_ref_start + position]
                assert reference.kind is (
                    SourceValueKind.UNIFORM_ELEMENT_ORIGIN_OPERAND
                    if mask & (1 << position)
                    else SourceValueKind.OPERAND
                )
                # Reverse shifts encode the authored count before the value.
                assert reference.index == (
                    1 - position if "rev_" in descriptor_name else position
                )
                assert bool(reference.materializer_index) == (
                    bool(mask) and position == 1
                )
        assert masks == {0, 1, 2, 3}, descriptor_name


def test_packed_narrow_clamp_broadcasts_compose_both_selectors() -> None:
    compiled = _compiled_arithmetic_rules()
    for maximum, minimum in (("maxnum", "minnum"), ("maximum", "minimum")):
        masks = set()
        for rule in _rules_for_source_op(compiled, vector.vector_clampf):
            keys = _rule_descriptor_keys(compiled, rule)
            if tuple(key.split(".broadcast_")[0] for key in keys) != (
                f"amdgpu.v_pk_{maximum}_f16",
                f"amdgpu.v_pk_{minimum}_f16",
            ):
                continue
            first, second = compiled.emits[rule.emit_start : rule.emit_start + 2]
            references = (
                compiled.value_refs[first.operand_ref_start],
                compiled.value_refs[first.operand_ref_start + 1],
                compiled.value_refs[second.operand_ref_start + 1],
            )
            mask = sum(
                1 << position
                for position, reference in enumerate(references)
                if reference.kind is SourceValueKind.UNIFORM_ELEMENT_ORIGIN_OPERAND
            )
            masks.add(mask)
            for key, emit, operand_mask in (
                (keys[0], first, mask & 3),
                (keys[1], second, (mask & 4) >> 1),
            ):
                attributes = compiled.attr_copies[
                    emit.attr_copy_start : emit.attr_copy_start + emit.attr_copy_count
                ]
                assert not attributes
                suffix = "_".join(
                    name
                    for bit, name in enumerate(("lhs", "rhs"))
                    if operand_mask & (1 << bit)
                )
                assert key.partition(".broadcast_")[2] == suffix
        assert masks == set(range(8)), (maximum, minimum)


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
        *(
            f"amdgpu.v_pk_{operation}_f32.broadcast_{sources}"
            for operation in ("add", "mul")
            for sources in ("lhs", "rhs", "lhs_rhs")
        ),
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
        vector.vector_insert: 11,
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


def test_insertion_uses_the_complete_construction_storage_family() -> None:
    compiled = _compiled_arithmetic_rules()
    insertion = {
        pattern.elements[0]: pattern
        for rule in _rules_for_source_op(compiled, vector.vector_insert)
        for pattern in _rule_type_patterns(compiled, rule)
        if pattern.kind == "vector"
    }
    assert set(insertion) == {
        scalar_type_name(kind)
        for kind in ScalarTypeKind
        if kind not in (ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET)
    }
    for source_op in (
        vector.vector_from_elements,
        vector.vector_splat,
        vector.vector_extract,
    ):
        shapes = {
            pattern
            for rule in _rules_for_source_op(compiled, source_op)
            for pattern in _rule_type_patterns(compiled, rule)
            if pattern.kind == "vector"
        }
        assert set(insertion.values()) <= shapes, source_op.name
    for rule in _rules_for_source_op(compiled, vector.vector_insert):
        scalar, destination, result = _rule_type_patterns(compiled, rule)
        assert scalar == Scalar(destination.elements)
        assert destination == result


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
    widening = {
        ("i8", "i16"),
        ("i8", "i32"),
        ("i8", "i64"),
        ("i16", "i32"),
        ("i16", "i64"),
        ("i32", "i64"),
    }
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
