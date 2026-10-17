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
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import defs as scalar_defs
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import defs as scf
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
    Buffer,
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
    View,
    compile_lower_rule_set,
)

_AFN_FASTMATH_FLAG = scalar_defs.FastMathFlags.case("afn").value
_NNAN_FASTMATH_FLAG = scalar_defs.FastMathFlags.case("nnan").value
_NSZ_FASTMATH_FLAG = scalar_defs.FastMathFlags.case("nsz").value


_COMPILED_ARITHMETIC_RULES = compile_lower_rule_set(
    AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
    dialect_ops=AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
)

_COMPILED_INTEGER_RULES = compile_lower_rule_set(
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
        if compiled.emits[emit_index].descriptor is not None
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


def _assert_address_operands_use_materializers(
    compiled: CompiledLowerRuleSet,
) -> None:
    address_patterns = {Scalar("index"), Scalar("offset")}
    direct_low_guard_kinds = {
        GuardKind.LOW_VALUE_REGISTER_CLASS,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
    }
    for rule in compiled.rules:
        if not address_patterns.intersection(_rule_type_patterns(compiled, rule)):
            continue
        guards = compiled.guards[rule.guard_start : rule.guard_start + rule.guard_count]
        materializable_operands = {
            (
                value_ref.source_node_index,
                value_ref.index,
                value_ref.element_index,
            )
            for guard in guards
            if guard.kind is GuardKind.VALUE_MATERIALIZABLE
            for value_ref in (compiled.value_refs[guard.value_ref_index],)
        }
        for guard in guards:
            if guard.kind not in direct_low_guard_kinds:
                continue
            value_ref = compiled.value_refs[guard.value_ref_index]
            if (
                guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS
                and value_ref.kind is SourceValueKind.OPERAND
            ):
                assert (
                    value_ref.source_node_index,
                    value_ref.index,
                    value_ref.element_index,
                ) in materializable_operands
            else:
                assert value_ref.kind is not SourceValueKind.OPERAND
            if guard.kind is GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ:
                other_value_ref = compiled.value_refs[guard.other_value_ref_index]
                assert other_value_ref.kind is not SourceValueKind.OPERAND
        for emit in compiled.emits[rule.emit_start : rule.emit_start + rule.emit_count]:
            value_refs = compiled.value_refs[
                emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
            ]
            for value_ref in value_refs:
                if value_ref.kind is SourceValueKind.OPERAND:
                    assert value_ref.materializer_index != 0


def _operand_width_guards(
    compiled: CompiledLowerRuleSet,
    rule: LowerRule,
) -> set[tuple[int, GuardKind, int]]:
    width_guard_kinds = {
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
    }
    result: set[tuple[int, GuardKind, int]] = set()
    for guard in compiled.guards[
        rule.guard_start : rule.guard_start + rule.guard_count
    ]:
        if guard.kind not in width_guard_kinds:
            continue
        value_ref = compiled.value_refs[guard.value_ref_index]
        if value_ref.kind is SourceValueKind.OPERAND:
            result.add((value_ref.index, guard.kind, guard.u64))
    return result


def _lhs_range_guards(
    compiled: CompiledLowerRuleSet,
    rule: LowerRule,
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for guard in compiled.guards[
        rule.guard_start : rule.guard_start + rule.guard_count
    ]:
        if guard.kind is not GuardKind.VALUE_I64_RANGE:
            continue
        value_ref = compiled.value_refs[guard.value_ref_index]
        if value_ref.kind is SourceValueKind.OPERAND and value_ref.index == 0:
            result.add((value_ref.index, guard.minimum_i64, guard.maximum_i64))
    return result


def test_whole_value_select_publishes_complete_recipe_family() -> None:
    compiled = _COMPILED_INTEGER_RULES
    rules = _rules_for_source_op(compiled, scf.scf_select)

    payload_types = (
        Scalar(
            (
                "i1",
                "index",
                "offset",
                "i8",
                "i16",
                "i32",
                "i64",
                "f8E4M3",
                "f8E5M2",
                "f16",
                "bf16",
                "f32",
                "f64",
            )
        ),
        Buffer(),
        View(
            (
                "index",
                "offset",
                "i8",
                "i16",
                "i32",
                "i64",
                "f8E4M3",
                "f8E5M2",
                "f16",
                "bf16",
                "f32",
                "f64",
            )
        ),
        TypePattern.vector(
            "i1",
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
        ),
        TypePattern.vector(
            ("i8", "f8E4M3", "f8E5M2"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_I8_LANES",
        ),
        TypePattern.vector(
            "i16",
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_I16_LANES",
        ),
        TypePattern.vector(
            ("f16", "bf16"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES",
        ),
        TypePattern.vector(
            ("i32", "f32"),
            minimum_static_elements=1,
            maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
        ),
        TypePattern.vector(
            ("i64", "f64"),
            minimum_static_elements=1,
            maximum_static_elements=("(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)"),
        ),
    )
    assert len(rules) == len(payload_types)
    assert all(rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY for rule in rules)
    assert all(rule.emit_count == 0 for rule in rules)
    assert tuple(_rule_type_patterns(compiled, rule) for rule in rules) == tuple(
        (Scalar("i1"), payload_type, payload_type, payload_type)
        for payload_type in payload_types
    )


def test_index_madd_rules_accept_wrapping_carrier_results() -> None:
    for compiled in (_COMPILED_ARITHMETIC_RULES, _COMPILED_INTEGER_RULES):
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


def test_narrow_address_rules_materialize_source_carriers() -> None:
    _assert_address_operands_use_materializers(_COMPILED_ARITHMETIC_RULES)
    _assert_address_operands_use_materializers(_COMPILED_INTEGER_RULES)


def test_nonhomomorphic_index_rules_guard_projected_source_widths() -> None:
    compiled = _COMPILED_INTEGER_RULES
    signed32 = GuardKind.VALUE_SIGNED_BIT_COUNT
    for source_op in (index.index_min, index.index_max):
        for rule in _rules_for_source_op(compiled, source_op):
            assert _operand_width_guards(compiled, rule) == {
                (0, signed32, 32),
                (1, signed32, 32),
            }
    for source_op, expected_guards in (
        (index.index_shli, set()),
        (index.index_shrsi, {(0, signed32, 32)}),
    ):
        for rule in _rules_for_source_op(compiled, source_op):
            assert _operand_width_guards(compiled, rule) == expected_guards
            assert _lhs_range_guards(compiled, rule) == set()
    for rule in _rules_for_source_op(compiled, index.index_shrui):
        assert _operand_width_guards(compiled, rule) == set()
        assert _lhs_range_guards(compiled, rule) == {
            (0, -(2**31), (2**32) - 1),
        }


def test_index_bit_operations_cover_scalar_and_vector_registers() -> None:
    compiled = _COMPILED_INTEGER_RULES
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
    compiled = _COMPILED_INTEGER_RULES
    scalar_rules = tuple(
        rule
        for rule in _rules_for_source_op(compiled, scalar_arithmetic.scalar_divui)
        if not rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
    )
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


def test_dynamic_integer_division_recipes_cover_banks_and_result_forms() -> None:
    compiled = _COMPILED_INTEGER_RULES
    common_descriptors = {
        "amdgpu.v_cvt_f32_u32",
        "amdgpu.v_rcp_f32",
        "amdgpu.v_mul_f32.lit",
        "amdgpu.v_cvt_u32_f32",
    }
    scalar_descriptors = common_descriptors | {
        "amdgpu.v_readfirstlane_b32",
        "amdgpu.s_mov_b32",
        "amdgpu.s_mul_i32",
        "amdgpu.s_mul_hi_u32",
        "amdgpu.s_add_u32",
        "amdgpu.s_sub_u32",
        "amdgpu.s_sub_co_u32",
        "amdgpu.s_cselect_b32",
    }
    vector_descriptors = common_descriptors | {
        "amdgpu.v_mov_b32",
        "amdgpu.v_mul_lo_u32",
        "amdgpu.v_mul_hi_u32",
        "amdgpu.v_add_u32",
        "amdgpu.v_sub_u32",
        "amdgpu.v_sub_co_u32",
        "amdgpu.v_cndmask_b32",
    }
    combined_ops = {
        index.index_divrem,
        scalar_arithmetic.scalar_divremsi,
        scalar_arithmetic.scalar_divremui,
    }

    for source_op, type_pattern, signed in (
        (index.index_div, Scalar("index"), False),
        (index.index_rem, Scalar("index"), False),
        (index.index_divrem, Scalar("index"), False),
        (scalar_arithmetic.scalar_divui, Scalar("i32"), False),
        (scalar_arithmetic.scalar_remui, Scalar("i32"), False),
        (scalar_arithmetic.scalar_divremui, Scalar("i32"), False),
        (scalar_arithmetic.scalar_divsi, Scalar("i32"), True),
        (scalar_arithmetic.scalar_remsi, Scalar("i32"), True),
        (scalar_arithmetic.scalar_divremsi, Scalar("i32"), True),
    ):
        recipes = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY
        )
        assert len(recipes) == 2
        expected_scalar_descriptors = set(scalar_descriptors)
        expected_vector_descriptors = set(vector_descriptors)
        if signed:
            expected_scalar_descriptors.update(
                ("amdgpu.s_xor_b32", "amdgpu.s_ashr_i32")
            )
            expected_vector_descriptors.update(
                ("amdgpu.v_xor_b32", "amdgpu.v_ashrrev_i32.src0_inline")
            )
        descriptor_sets = set()
        for recipe in recipes:
            assert recipe.emit_count == 0
            guards = compiled.guards[
                recipe.guard_start : recipe.guard_start + recipe.guard_count
            ]
            assert (
                sum(guard.kind == GuardKind.VALUE_NOT_EXACT_I64 for guard in guards)
                == 1
            )
            descriptor_sets.add(
                frozenset(
                    guard.descriptor.key
                    for guard in guards
                    if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
                    and guard.descriptor is not None
                )
            )
            expected_type_count = 4 if source_op in combined_ops else 3
            assert _rule_type_patterns(compiled, recipe) == (type_pattern,) * (
                expected_type_count
            )
        assert descriptor_sets == {
            frozenset(expected_scalar_descriptors),
            frozenset(expected_vector_descriptors),
        }


def test_unsigned_bitfield_extract_rules_try_native_bfe_before_shift_mask() -> None:
    positions = _descriptor_sequence_positions(
        _COMPILED_ARITHMETIC_RULES,
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
        _COMPILED_ARITHMETIC_RULES,
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
    compiled = _COMPILED_INTEGER_RULES
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
        _COMPILED_ARITHMETIC_RULES,
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
    compiled = _COMPILED_ARITHMETIC_RULES

    for source_op in (
        scalar_arithmetic.scalar_copysignf,
        vector.vector_copysignf,
    ):
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert (
            positions[("amdgpu.v_bfi_b32.src0_lit",)]
            < positions[("amdgpu.s_mov_b32", "amdgpu.v_bfi_b32")]
        )

    scalar_f64_sequences = tuple(
        _rule_descriptor_keys(compiled, rule)
        for rule in _rules_for_source_op(compiled, scalar_arithmetic.scalar_copysignf)
        if set(_rule_type_patterns(compiled, rule)) == {Scalar("f64")}
    )
    assert scalar_f64_sequences == (
        ("amdgpu.v_bfi_b32.src0_lit",),
        ("amdgpu.s_mov_b32", "amdgpu.v_bfi_b32"),
    )


def test_i64_to_f32_rules_preserve_single_rounding() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    expected_sequences = {
        scalar_conversion.scalar_sitofp: (
            "amdgpu.v_xor_b32",
            "amdgpu.v_ashrrev_i32.src0_inline",
            "amdgpu.v_add_u32.src0_inline",
            "amdgpu.v_cls_i32",
            "amdgpu.v_add_u32.lit",
            "amdgpu.v_min_u32",
            "amdgpu.v_lshlrev_b64",
            "amdgpu.v_min_u32.src0_inline",
            "amdgpu.v_or_b32",
            "amdgpu.v_cvt_f32_i32",
            "amdgpu.v_sub_u32.src0_inline",
            "amdgpu.v_ldexp_f32",
        ),
        scalar_conversion.scalar_uitofp: (
            "amdgpu.v_clz_i32_u32",
            "amdgpu.v_min_u32.src0_inline",
            "amdgpu.v_lshlrev_b64",
            "amdgpu.v_min_u32.src0_inline",
            "amdgpu.v_or_b32",
            "amdgpu.v_cvt_f32_u32",
            "amdgpu.v_sub_u32.src0_inline",
            "amdgpu.v_ldexp_f32",
        ),
    }
    for source_op, expected_sequence in expected_sequences.items():
        matching_rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if set(_rule_type_patterns(compiled, rule))
            == {Scalar("i64"), Scalar("f32")}
        )
        assert len(matching_rules) == 1
        assert _rule_descriptor_keys(compiled, matching_rules[0]) == expected_sequence


def test_integer_extrema_rules_prefer_encoded_constants() -> None:
    compiled = _COMPILED_INTEGER_RULES

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
    compiled = _COMPILED_INTEGER_RULES

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
    compiled = _COMPILED_INTEGER_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES

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
        (
            vector.vector_shli,
            "amdgpu.v_pk_lshlrev_b16",
            "amdgpu.v_lshlrev_b32",
        ),
        (
            vector.vector_shrsi,
            "amdgpu.v_pk_ashrrev_i16",
            "amdgpu.v_ashrrev_i32",
        ),
        (
            vector.vector_shrui,
            "amdgpu.v_pk_lshrrev_b16",
            "amdgpu.v_lshrrev_b32",
        ),
    )
    for source_op, packed_descriptor, word_descriptor in shift_cases:
        positions = _descriptor_sequence_positions(compiled, source_op)
        assert positions[(packed_descriptor,)] < positions[(word_descriptor,)]


def test_i64_vector_shifts_bind_exact_i32_count_origins() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    i32_vector = TypePattern.vector(
        "i32",
        minimum_static_elements=1,
        maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
    )
    i64_vector = TypePattern.vector(
        "i64",
        minimum_static_elements=1,
        maximum_static_elements="(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)",
    )

    for source_op, descriptor_key in (
        (vector.vector_shli, "amdgpu.v_lshlrev_b64"),
        (vector.vector_shrsi, "amdgpu.v_ashrrev_i64"),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b64"),
    ):
        matches: dict[str, tuple[int, LowerRule]] = {}
        for ordinal, rule in enumerate(_rules_for_source_op(compiled, source_op)):
            descriptor_keys = _rule_descriptor_keys(compiled, rule)
            if descriptor_keys in (
                (descriptor_key,),
                (f"{descriptor_key}.src0_inline",),
            ):
                assert descriptor_keys[0] not in matches
                matches[descriptor_keys[0]] = (ordinal, rule)

        assert set(matches) == {descriptor_key, f"{descriptor_key}.src0_inline"}
        inline_ordinal, inline_rule = matches[f"{descriptor_key}.src0_inline"]
        varying_ordinal, varying_rule = matches[descriptor_key]
        assert inline_ordinal < varying_ordinal
        assert _rule_type_patterns(compiled, inline_rule) == (
            i64_vector,
            i64_vector,
            i64_vector,
        )

        inline_guards = compiled.guards[
            inline_rule.guard_start : inline_rule.guard_start + inline_rule.guard_count
        ]
        exact_guard = next(
            guard for guard in inline_guards if guard.kind is GuardKind.VALUE_EXACT_I64
        )
        range_guard = next(
            guard for guard in inline_guards if guard.kind is GuardKind.VALUE_I64_RANGE
        )
        for guard in (exact_guard, range_guard):
            count_ref = compiled.value_refs[guard.value_ref_index]
            assert count_ref.kind is SourceValueKind.OPERAND
            assert count_ref.index == 1
        assert (range_guard.minimum_i64, range_guard.maximum_i64) == (0, 63)

        origin_guards = tuple(
            guard
            for guard in compiled.guards[
                varying_rule.guard_start : varying_rule.guard_start
                + varying_rule.guard_count
            ]
            if guard.kind is GuardKind.VALUE_TYPE
            and compiled.value_refs[guard.value_ref_index].kind
            is SourceValueKind.EXACT_LANE_ORIGIN_OPERAND
        )
        assert len(origin_guards) == 1
        origin_guard = origin_guards[0]
        assert (
            compiled.type_patterns[origin_guard.type_pattern_index].type_pattern
            == i32_vector
        )
        origin_guard_ref = compiled.value_refs[origin_guard.value_ref_index]
        assert origin_guard_ref.index == 1

        emit = compiled.emits[varying_rule.emit_start]
        assert emit.kind is LowerEmitKind.DESCRIPTOR_OP_PER_LANE
        assert emit.operand_ref_count == 2
        count_ref, value_ref = compiled.value_refs[
            emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
        ]
        assert count_ref.kind is SourceValueKind.EXACT_LANE_ORIGIN_OPERAND
        assert count_ref.index == 1
        assert value_ref.kind is SourceValueKind.OPERAND
        assert value_ref.index == 0


def test_vector_vop2_rules_materialize_vsrc1_as_vgpr() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    register_cases = (
        (vector.vector_addi, "amdgpu.v_add_u32", 1),
        (vector.vector_subi, "amdgpu.v_sub_u32", 1),
        (vector.vector_muli, "amdgpu.v_mul_lo_u32", 1),
        (vector.vector_minsi, "amdgpu.v_min_i32", 1),
        (vector.vector_maxsi, "amdgpu.v_max_i32", 1),
        (vector.vector_minui, "amdgpu.v_min_u32", 1),
        (vector.vector_maxui, "amdgpu.v_max_u32", 1),
        (vector.vector_andi, "amdgpu.v_and_b32", 4),
        (vector.vector_ori, "amdgpu.v_or_b32", 4),
        (vector.vector_xori, "amdgpu.v_xor_b32", 4),
        (vector.vector_shli, "amdgpu.v_lshlrev_b32", 1),
        (vector.vector_shrsi, "amdgpu.v_ashrrev_i32", 1),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b32", 1),
    )
    for source_op, descriptor_key, expected_rule_count in register_cases:
        rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if _rule_descriptor_keys(compiled, rule) == (descriptor_key,)
        )
        assert len(rules) == expected_rule_count
        for rule in rules:
            emit = compiled.emits[rule.emit_start]
            assert emit.operand_ref_count == 2
            lhs = compiled.value_refs[emit.operand_ref_start]
            rhs = compiled.value_refs[emit.operand_ref_start + 1]
            assert lhs.materializer_index == 0
            assert rhs.materializer_index != 0

    literal_cases = (
        (vector.vector_addi, "amdgpu.v_add_u32.lit", 2),
        (vector.vector_andi, "amdgpu.v_and_b32.lit", 2),
        (vector.vector_ori, "amdgpu.v_or_b32.lit", 2),
        (vector.vector_xori, "amdgpu.v_xor_b32.lit", 2),
        (vector.vector_shli, "amdgpu.v_lshlrev_b32.src0_inline", 1),
        (vector.vector_shrsi, "amdgpu.v_ashrrev_i32.src0_inline", 1),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b32.src0_inline", 1),
    )
    for source_op, descriptor_key, expected_rule_count in literal_cases:
        rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if rule.emit_count == 1
            and _rule_descriptor_keys(compiled, rule) == (descriptor_key,)
        )
        assert len(rules) == expected_rule_count
        for rule in rules:
            emit = compiled.emits[rule.emit_start]
            assert emit.operand_ref_count == 1
            source = compiled.value_refs[emit.operand_ref_start]
            assert source.materializer_index != 0


def test_packed_bf16_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES

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


def test_native_reciprocal_root_rules_require_approximate_functions() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    expected_descriptors = {
        scalar_math.scalar_rsqrtf: {
            ("amdgpu.v_rsq_f32",),
            ("amdgpu.v_rsq_f64",),
        },
        vector.vector_rsqrtf: {("amdgpu.v_rsq_f32",)},
    }
    for source_op, expected in expected_descriptors.items():
        rules = _rules_for_source_op(compiled, source_op)
        assert {_rule_descriptor_keys(compiled, rule) for rule in rules} == expected
        for rule in rules:
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            assert any(
                guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
                and guard.u64 == _AFN_FASTMATH_FLAG
                for guard in guards
            )


def test_f32_literal_fma_product_flushing_forms_require_afn() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
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


def test_fp64_binary_rules_legalize_packet_sources() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    for source_op, descriptor_key in (
        (scalar_arithmetic.scalar_addf, "amdgpu.v_add_f64"),
        (scalar_arithmetic.scalar_subf, "amdgpu.v_sub_f64"),
        (scalar_arithmetic.scalar_mulf, "amdgpu.v_mul_f64"),
        (scalar_arithmetic.scalar_minimumf, "amdgpu.v_minimum_f64"),
        (scalar_arithmetic.scalar_maximumf, "amdgpu.v_maximum_f64"),
        (scalar_arithmetic.scalar_minnumf, "amdgpu.v_min_f64"),
        (scalar_arithmetic.scalar_maxnumf, "amdgpu.v_max_f64"),
        (vector.vector_minimumf, "amdgpu.v_minimum_f64"),
        (vector.vector_maximumf, "amdgpu.v_maximum_f64"),
        (vector.vector_minnumf, "amdgpu.v_min_f64"),
        (vector.vector_maxnumf, "amdgpu.v_max_f64"),
    ):
        direct_rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if _rule_descriptor_keys(compiled, rule) == (descriptor_key,)
        )
        assert direct_rules, source_op.name
        for rule in direct_rules:
            emit = compiled.emits[rule.emit_start]
            assert (
                emit.operand_materialization is DescriptorOperandMaterialization.TARGET
            ), (source_op.name, descriptor_key)


def test_exact_f32_division_uses_negated_fma_residuals() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES

    for source_op in (scalar_arithmetic.scalar_divf, vector.vector_divf):
        exact_rules = tuple(
            rule
            for rule in _rules_for_source_op(compiled, source_op)
            if "amdgpu.v_div_fixup_f32" in _rule_descriptor_keys(compiled, rule)
        )
        assert len(exact_rules) == 1
        descriptor_keys = _rule_descriptor_keys(compiled, exact_rules[0])
        assert descriptor_keys == (
            "amdgpu.v_div_scale_f32",
            "amdgpu.v_rcp_f32",
            "amdgpu.v_fma_f32.neg_a_one",
            "amdgpu.v_fma_f32",
            "amdgpu.v_div_scale_f32",
            "amdgpu.v_mul_f32",
            "amdgpu.v_fma_f32.neg_a",
            "amdgpu.v_fma_f32",
            "amdgpu.v_fma_f32.neg_a",
            "amdgpu.v_div_fmas_f32",
            "amdgpu.v_div_fixup_f32",
        )


def test_packed_f16_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES

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


def test_clamp_rules_publish_existing_f32_fallback_recipes() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES

    expected_descriptors = {
        0: {
            "amdgpu.v_cmp_olt_f32",
            "amdgpu.v_cmp_ogt_f32",
            "amdgpu.v_cndmask_b32",
        },
        1: {"amdgpu.v_max_f32", "amdgpu.v_min_f32"},
    }
    for source_op, expected_type in (
        (scalar_arithmetic.scalar_clampf, Scalar("f32")),
        (
            vector.vector_clampf,
            TypePattern.vector(
                "f32",
                minimum_static_elements=1,
                maximum_static_elements="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
            ),
        ),
    ):
        recipes = {}
        required_fastmath = {}
        for rule in _rules_for_source_op(compiled, source_op):
            if not rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY:
                continue
            guards = compiled.guards[
                rule.guard_start : rule.guard_start + rule.guard_count
            ]
            mode_guards = tuple(
                guard for guard in guards if guard.kind == GuardKind.ENUM_ATTR_EQUALS
            )
            assert len(mode_guards) == 1
            recipes[mode_guards[0].u64] = {
                guard.descriptor.key
                for guard in guards
                if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
            }
            required_fastmath[mode_guards[0].u64] = {
                guard.u64
                for guard in guards
                if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
            }
            assert set(_rule_type_patterns(compiled, rule)) == {expected_type}

        assert recipes == expected_descriptors
        assert required_fastmath == {
            0: set(),
            1: {_NNAN_FASTMATH_FLAG, _NSZ_FASTMATH_FLAG},
        }


def test_packed_f32_arithmetic_rules_publish_native_pk_ops() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES
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
            assert (
                emit.operand_materialization is DescriptorOperandMaterialization.TARGET
            )
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
    compiled = _COMPILED_ARITHMETIC_RULES
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
            assert (
                emit.operand_materialization is DescriptorOperandMaterialization.TARGET
            )
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
    compiled = _COMPILED_ARITHMETIC_RULES
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
    compiled = _COMPILED_ARITHMETIC_RULES
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
    compiled = _COMPILED_ARITHMETIC_RULES
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
    compiled = _COMPILED_ARITHMETIC_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES
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
    compiled = _COMPILED_ARITHMETIC_RULES

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
    compiled = _COMPILED_ARITHMETIC_RULES
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
    compiled = _COMPILED_ARITHMETIC_RULES

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
        assert sorted(maximum_lane_counts) == [32, 32, 64]


def test_bitpack_contract_covers_word_and_predicate_lanes() -> None:
    compiled = _COMPILED_ARITHMETIC_RULES
    cases: set[tuple[str, str, int]] = set()

    for rule in _rules_for_source_op(compiled, vector.vector_bitpack):
        guards = compiled.guards[rule.guard_start : rule.guard_start + rule.guard_count]
        types = tuple(
            compiled.type_patterns[guard.type_pattern_index].type_pattern
            for guard in guards
            if guard.kind == GuardKind.VALUE_TYPE
        )
        width_guard = next(
            guard for guard in guards if guard.kind == GuardKind.I64_RANGE
        )
        assert any(
            guard.kind == GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES
            for guard in guards
        )
        cases.add((types[0].element, types[1].element, width_guard.maximum_i64))

    assert cases == {("i32", "i8", 8), ("i1", "i8", 1)}
