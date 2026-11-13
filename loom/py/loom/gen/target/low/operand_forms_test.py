# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.gen.target.low import compiler
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    DescriptorOpKind,
    MemorySpace,
    OperandForm,
    OperandFormImmediateAction,
    OperandFormMatch,
    OperandFormMatchKind,
    RegClassAlt,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_INCREMENT_I32_DESCRIPTOR,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_NATIVE_COND_BR_I32_DESCRIPTOR,
)


def test_operand_form_can_replace_register_input_with_constant_packet() -> None:
    replacement = TEST_LOW_CONST_I32_DESCRIPTOR
    source = replace(
        TEST_LOW_INCREMENT_I32_DESCRIPTOR,
        immediates=(),
        asm_forms=(),
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement.key,
                matches=(OperandFormMatch("value", OperandFormMatchKind.ALL_EQUAL_EXACT_I64),),
                immediate_action=OperandFormImmediateAction.SET_MATCHED_I64,
                immediate_field="i32_value",
                immediate_source_operand="value",
            ),
        ),
    )
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))
    (form,) = compiled.operand_forms
    assert compiled.descriptors[form.replacement_descriptor_ordinal].op_kind is DescriptorOpKind.CONST
    assert form.operand_map_count == 0
    assert form.immediate_action is OperandFormImmediateAction.SET_MATCHED_I64


def test_operand_form_cannot_replace_structural_branch_packet() -> None:
    replacement = TEST_LOW_NATIVE_COND_BR_I32_DESCRIPTOR
    source = replace(
        replacement,
        key="test.branch.source",
        operand_forms=(OperandForm(replacement_descriptor=replacement.key, matches=(OperandFormMatch("lhs", OperandFormMatchKind.ALL_EQUAL_I64, 0),)),),
    )
    with pytest.raises(ValueError, match=r"must replace low\.op with low\.op or low\.const"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))


def _commutative_form_descriptors(*, narrow: bool = False):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    left = base.operands[1]
    if narrow:
        left = replace(left, reg_alts=(*left.reg_alts, RegClassAlt("test.f32")))
    replacement = replace(
        TEST_LOW_INCREMENT_I32_DESCRIPTOR,
        encoding_byte_count=4,
        operands=(base.operands[0], base.operands[2]),
        asm_forms=(),
    )
    source = replace(
        base,
        encoding_byte_count=4,
        operands=(base.operands[0], left, base.operands[2]),
        constraints=(Constraint(ConstraintKind.COMMUTABLE, 1, 2),),
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement.key,
                matches=(OperandFormMatch("lhs", OperandFormMatchKind.ALL_EQUAL_EXACT_I64),),
                immediate_action=OperandFormImmediateAction.SET_MATCHED_I64,
                immediate_field="i32_value",
                immediate_source_operand="lhs",
            ),
        ),
    )
    return source, replacement


@pytest.mark.parametrize("narrow", [False, True])
def test_commutation_retains_source_map_and_only_needed_class_guards(narrow: bool) -> None:
    source, replacement = _commutative_form_descriptors(narrow=narrow)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))
    original, commuted = compiled.operand_forms
    assert compiled.descriptor_rows[0]["operand_form_count"] == 2
    assert original.match_count == 1
    assert commuted.match_count == (2 if narrow else 1)
    assert compiled.operand_form_operand_indices[original.operand_map_start] == 1
    assert compiled.operand_form_operand_indices[commuted.operand_map_start] == 0
    matches = compiled.operand_form_matches[commuted.match_start : commuted.match_start + commuted.match_count]
    assert matches[0].source_operand_index == 2
    assert matches[0].source_packet_operand_index == 1
    assert matches[0].match_kind is OperandFormMatchKind.ALL_EQUAL_EXACT_I64
    assert commuted.immediate_match_index == 0
    if narrow:
        assert matches[1].source_operand_index == 1
        assert matches[1].source_packet_operand_index == 0
        assert matches[1].match_kind is OperandFormMatchKind.ACCEPTED_REGISTER_CLASS
        assert matches[1].match_i64 == 1


def test_commutation_keeps_form_priority_before_next_encoding() -> None:
    source, replacement = _commutative_form_descriptors()
    literal = replace(replacement, key="test.literal", mnemonic="test.literal")
    source = replace(source, operand_forms=(*source.operand_forms, replace(source.operand_forms[0], replacement_descriptor=literal.key)))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement, literal)))
    assert [form.replacement_descriptor_ordinal for form in compiled.operand_forms] == [1, 1, 2, 2]


@pytest.mark.parametrize(("source_size", "replacement_size", "form_count"), [(4, 4, 2), (8, 4, 2), (4, 8, 1), (None, 4, 1), (4, None, 1)])
def test_commutation_requires_non_increasing_known_encoding_size(source_size, replacement_size, form_count: int) -> None:
    source, replacement = _commutative_form_descriptors()
    source = replace(source, encoding_byte_count=source_size)
    replacement = replace(replacement, encoding_byte_count=replacement_size)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))
    # The explicitly authored form remains available in every case.
    assert len(compiled.operand_forms) == form_count


@pytest.mark.parametrize("size", [0, -1, 2**32])
def test_encoding_size_rejects_invalid_cost_facts(size: int) -> None:
    source, replacement = _commutative_form_descriptors()
    source = replace(source, encoding_byte_count=size)
    with pytest.raises(ValueError, match="encoding byte count"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))


@pytest.mark.parametrize("constraint", [None, Constraint(ConstraintKind.COMMUTABLE, 2, 3)])
def test_only_commutations_reaching_a_matched_operand_add_forms(constraint) -> None:
    source, replacement = _commutative_form_descriptors()
    extra = replace(source.operands[-1], field_name="extra")
    source = replace(source, operands=(*source.operands, extra), constraints=() if constraint is None else (constraint,), asm_forms=())
    replacement = replace(replacement, operands=(*replacement.operands, extra))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement)))
    assert len(compiled.operand_forms) == 1


@pytest.mark.parametrize(("change", "diagnostic"), [("units", "unit count"), ("class", "compatible register class"), ("part", "register part")])
@pytest.mark.parametrize("commuted", [False, True])
def test_operand_form_rejects_incompatible_receiving_storage(change: str, diagnostic: str, commuted: bool) -> None:
    source, replacement = _commutative_form_descriptors()
    left = source.operands[1]
    if change == "units":
        left = replace(left, unit_count=2)
    elif change == "class":
        left = replace(left, reg_alts=(RegClassAlt("test.f32"),))
    else:
        left = replace(left, reg_alts=(replace(left.reg_alts[0], register_part="test.i32.low16"),))
    operands = (source.operands[0], left, source.operands[2]) if commuted else (source.operands[0], source.operands[1], replace(left, field_name="rhs"))
    source = replace(source, operands=operands)
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement))
    if commuted:
        # Raw value commutativity does not require both slots to accept every
        # source type. An impossible exchange contributes no candidate.
        assert len(compiler.compile_descriptor_set(descriptor_set).operand_forms) == 1
    else:
        with pytest.raises(ValueError, match=diagnostic):
            compiler.compile_descriptor_set(descriptor_set)


@pytest.mark.parametrize("change", ["none", "value", "kind", "source", "map_order"])
def test_operand_form_interns_exact_predicates_and_maps(change: str) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operands = (*base.operands, replace(base.operands[-1], field_name="extra"))
    replacements = []
    sources = []
    for ordinal in range(2):
        changed = ordinal == 1
        matched_index = 2 if changed and change == "source" else 3
        retained = [index for index in range(1, len(operands)) if index != matched_index]
        if changed and change == "map_order":
            retained.reverse()
        replacement = replace(
            base,
            key=f"test.replacement.{ordinal}",
            mnemonic=f"test.replacement.{ordinal}",
            operands=(operands[0], *(operands[index] for index in retained)),
            asm_forms=(),
        )
        match = OperandFormMatch(
            source_operand=operands[matched_index].field_name,
            match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64 if changed and change == "kind" else OperandFormMatchKind.ALL_EQUAL_I64,
            match_i64=17 if changed and change == "value" else 0,
        )
        sources.append(
            replace(
                base,
                key=f"test.source.{ordinal}",
                mnemonic=f"test.source.{ordinal}",
                operands=operands,
                asm_forms=(),
                operand_forms=(OperandForm(replacement_descriptor=replacement.key, matches=(match,)),),
            )
        )
        replacements.append(replacement)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(*sources, *replacements)))
    first, second = compiled.operand_forms
    same_matches = change in ("none", "map_order")
    same_map = change not in ("source", "map_order")
    assert (first.match_start == second.match_start) == same_matches
    assert (first.operand_map_start == second.operand_map_start) == same_map
    assert len(compiled.operand_form_matches) == (1 if same_matches else 2)
    assert len(compiled.operand_form_operand_indices) == (2 if same_map else 4)
    for source, replacement, form in zip(sources, replacements, compiled.operand_forms, strict=True):
        assert compiled.descriptors[form.replacement_descriptor_ordinal].key == replacement.key
        assert form.match_count == 1
        match = compiled.operand_form_matches[form.match_start]
        authored_match = source.operand_forms[0].matches[0]
        source_index = next(index for index, operand in enumerate(source.operands) if operand.field_name == authored_match.source_operand)
        assert match.source_operand_index == source_index
        assert match.source_packet_operand_index == source_index - 1
        assert match.match_kind == authored_match.match_kind
        assert match.match_i64 == authored_match.match_i64
        expected_map = [next(index for index, operand in enumerate(source.operands[1:]) if operand.field_name == retained.field_name) for retained in replacement.operands[1:]]
        assert compiled.operand_form_operand_indices[form.operand_map_start : form.operand_map_start + form.operand_map_count] == expected_map


@pytest.mark.parametrize("change", ["none", "width", "space", "order"])
def test_operand_form_preserves_effect_identity(change: str) -> None:
    base = TEST_LOW_LOAD_V4I32_DESCRIPTOR
    effects = (base.effects[0], replace(base.effects[0], memory_space=MemorySpace.WORKGROUP))
    replacement_effects = effects
    if change == "width":
        replacement_effects = (replace(effects[0], width_bits=256), effects[1])
    elif change == "space":
        replacement_effects = (replace(effects[0], memory_space=MemorySpace.GLOBAL), effects[1])
    elif change == "order":
        replacement_effects = tuple(reversed(effects))
    replacement = replace(base, key="test.load.variant", mnemonic="test.load.variant", operands=base.operands[:1], asm_forms=(), effects=replacement_effects)
    source = replace(
        base,
        effects=effects,
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement.key,
                matches=(OperandFormMatch(source_operand="address", match_kind=OperandFormMatchKind.ALL_EQUAL_I64, match_i64=0),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement))
    if change == "none":
        compiler.compile_descriptor_set(descriptor_set)
    else:
        with pytest.raises(ValueError, match="must preserve semantic effect ordinals"):
            compiler.compile_descriptor_set(descriptor_set)
