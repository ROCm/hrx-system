# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU comparison contract source tables."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.scalar import comparison as scalar
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.compare import (
    AMDGPU_COMPARE_CONTRACT_DIALECT_OPS,
    AMDGPU_COMPARE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    CompiledLowerRuleSet,
    GuardKind,
    LowerRule,
    Scalar,
    SourceValueKind,
    compile_lower_rule_set,
)

_COMPILED_RULES = compile_lower_rule_set(
    AMDGPU_COMPARE_CONTRACT_FRAGMENT,
    dialect_ops=AMDGPU_COMPARE_CONTRACT_DIALECT_OPS,
)


def _rules_for_source_op(
    compiled: CompiledLowerRuleSet, source_op: Op
) -> tuple[LowerRule, ...]:
    for span in compiled.spans:
        if span.source_op is source_op:
            return compiled.rules[span.rule_start : span.rule_start + span.rule_count]
    raise AssertionError(f"no lower-rule span for {source_op.name}")


def _descriptor_sequences(
    compiled: CompiledLowerRuleSet, source_op: Op
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            compiled.emits[emit_index].descriptor.key
            for emit_index in range(rule.emit_start, rule.emit_start + rule.emit_count)
        )
        for rule in _rules_for_source_op(compiled, source_op)
    )


def test_scalar_float_classification_selects_cheapest_constant_form() -> None:
    compiled = _COMPILED_RULES
    assert _descriptor_sequences(compiled, scalar.scalar_isnanf) == tuple(
        (f"amdgpu.v_cmp_class_f{bit_width}.classes_inline",)
        for bit_width in (16, 32, 64)
    )
    for source_op in (scalar.scalar_isinff, scalar.scalar_isfinitef):
        assert _descriptor_sequences(compiled, source_op) == tuple(
            sequence
            for bit_width in (16, 32, 64)
            for sequence in (
                (f"amdgpu.v_cmp_class_f{bit_width}.classes_lit",),
                ("amdgpu.s_mov_b32", f"amdgpu.v_cmp_class_f{bit_width}"),
            )
        )


def test_vector_float_classification_publishes_all_native_widths() -> None:
    compiled = _COMPILED_RULES
    for source_op in (
        vector.vector_isnanf,
        vector.vector_isinff,
        vector.vector_isfinitef,
    ):
        rules = _rules_for_source_op(compiled, source_op)
        assert len(rules) == 3
        assert all(rule.emit_count == 0 for rule in rules)


def test_scalar_address_comparisons_materialize_source_carriers() -> None:
    compiled = _COMPILED_RULES
    address_patterns = {Scalar("index"), Scalar("offset")}
    address_sgpr_materializer_index = next(
        materializer_index
        for materializer_index, materializer in enumerate(
            AMDGPU_COMPARE_CONTRACT_FRAGMENT.materializers, start=1
        )
        if materializer.name == "address_sgpr"
    )
    direct_low_guard_kinds = {
        GuardKind.LOW_VALUE_REGISTER_CLASS,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
    }
    address_rule_count = 0
    scalar_rule_count = 0
    for rule in _rules_for_source_op(compiled, index.index_cmp):
        guards = compiled.guards[rule.guard_start : rule.guard_start + rule.guard_count]
        type_patterns = {
            compiled.type_patterns[guard.type_pattern_index].type_pattern
            for guard in guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        if not address_patterns.intersection(type_patterns):
            continue
        address_rule_count += 1
        uses_address_sgpr_materializer = any(
            guard.kind is GuardKind.VALUE_MATERIALIZABLE
            and compiled.value_refs[guard.value_ref_index].materializer_index
            == address_sgpr_materializer_index
            for guard in guards
        )
        if uses_address_sgpr_materializer:
            scalar_rule_count += 1
        for guard in guards:
            if guard.kind not in direct_low_guard_kinds:
                continue
            value_ref = compiled.value_refs[guard.value_ref_index]
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
    assert address_rule_count != 0
    assert scalar_rule_count != 0
