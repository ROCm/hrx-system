# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from itertools import permutations

import pytest

from loom.gen.target.low import compiler, views
from loom.gen.target.low.compiled import CompiledAsmOperandType, CompiledAsmTiedResult
from loom.gen.target.low.low_descriptors import generate_descriptor_set
from loom.ir import ScalarTypeKind
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    AsmResultValueType,
    Constraint,
    ConstraintKind,
    ImmediateFlag,
    NativeAsmValue,
    NativeAsmValueKind,
    OperandFlag,
    OperandRole,
)
from loom.target.test.descriptors import TEST_LOW_ADD_I32_DESCRIPTOR, TEST_LOW_COLLECT_FLAGS_DESCRIPTOR, TEST_LOW_CORE_DESCRIPTOR_SET


def _compile(descriptor):
    return compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


@pytest.mark.parametrize("results", [("carry", "dst"), ("dst",), ()])
def test_results_require_complete_canonical_order(results):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base,
        operands=(base.operands[0], replace(base.operands[0], field_name="carry"), *base.operands[1:]),
        asm_forms=(AsmForm(results=results, operands=("lhs", "rhs")),),
    )
    with pytest.raises(ValueError, match="results must enumerate canonical packet order"):
        _compile(descriptor)


@pytest.mark.parametrize("operands", [("rhs", "lhs"), ("lhs",), ()])
@pytest.mark.parametrize("segmented", [False, True])
def test_operands_require_complete_canonical_order(operands, segmented):
    form = AsmForm(results=("dst",), operands=operands)
    if segmented:
        form = replace(form, operands=(), operand_segments=(AsmOperandSegment(AsmOperandSegmentDelimiter.PAREN, operands),))
    with pytest.raises(ValueError, match="operands must enumerate canonical packet order"):
        _compile(replace(TEST_LOW_ADD_I32_DESCRIPTOR, asm_forms=(form,)))


@pytest.mark.parametrize("role", [OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE])
def test_packet_order_includes_every_input_role_but_not_implicit_state(role):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base,
        operands=(
            base.operands[0],
            base.operands[1],
            replace(base.operands[1], field_name="state", role=OperandRole.IMPLICIT, flags=(OperandFlag.IMPLICIT,)),
            replace(base.operands[2], role=role),
        ),
    )
    compiled = _compile(descriptor)
    assert compiled.operand_source_value_indices == [0, 0, None, 1]


def test_segments_preserve_canonical_order_with_variadic_suffix():
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base,
        operands=(*base.operands[:2], replace(base.operands[2], flags=(OperandFlag.VARIADIC,))),
        asm_forms=(
            AsmForm(
                results=("dst",),
                operand_segments=(
                    AsmOperandSegment(AsmOperandSegmentDelimiter.PAREN, ("lhs",)),
                    AsmOperandSegment(AsmOperandSegmentDelimiter.SQUARE, ("rhs",)),
                ),
            ),
        ),
    )
    compiled = _compile(descriptor)
    assert [segment.has_variadic_operand for segment in compiled.asm_forms[0].operand_segments] == [False, True]


@pytest.mark.parametrize("named_suffix", [False, True])
def test_prefix_fields_preserve_independent_suffix_syntax(named_suffix):
    descriptor = TEST_LOW_COLLECT_FLAGS_DESCRIPTOR
    form = descriptor.asm_forms[0]
    if named_suffix:
        form = replace(form, immediates=(*form.immediates[:-1], AsmImmediate("literal", name="value")))
    compiled = _compile(replace(descriptor, asm_forms=(form,)))
    layout = compiled.asm_table_storage.layouts[compiled.asm_forms[0].layout_index]
    assert layout.prefix_immediate_count == 4
    assert layout.explicit_immediate_count == 5
    assert layout.has_named_immediates == named_suffix
    assert layout.requires_named_immediates == named_suffix


@pytest.mark.parametrize("mutation", ["after_trailing", "flags_not_first", "flags_named", "flags_not_mask", "flags_default", "duplicate_keyword"])
def test_prefix_grammar_is_validated_at_generation(mutation):
    descriptor = TEST_LOW_COLLECT_FLAGS_DESCRIPTOR
    fields = list(descriptor.asm_forms[0].immediates)
    expected = "flags must be the first"
    if mutation == "after_trailing":
        fields[-2:] = reversed(fields[-2:])
        expected = "prefix immediates must precede"
    elif mutation == "flags_not_first":
        fields[:2] = reversed(fields[:2])
    elif mutation == "flags_named":
        fields[0] = replace(fields[0], name="flags")
    elif mutation in ("flags_not_mask", "flags_default"):
        flags = (
            replace(descriptor.immediates[0], enum_domain=descriptor.immediates[2].enum_domain)
            if mutation == "flags_not_mask"
            else replace(descriptor.immediates[0], flags=(ImmediateFlag.DEFAULT_VALUE,), default_value=1)
        )
        descriptor = replace(descriptor, immediates=(flags, *descriptor.immediates[1:]))
    else:
        fields[2] = replace(fields[2], name="access")
        expected = "uses immediate name 'access' more than once"
    descriptor = replace(descriptor, asm_forms=(replace(descriptor.asm_forms[0], immediates=tuple(fields)),))
    with pytest.raises(ValueError, match=expected):
        _compile(descriptor)


def test_native_assembly_order_is_independent_of_low_packet_order():
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    form = replace(
        base.asm_forms[0],
        native_assembly_values=(
            NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="rhs"),
            NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
        ),
    )
    compiled = _compile(replace(base, asm_forms=(form,)))
    assert [value.index for value in compiled.asm_forms[0].native_assembly_values] == [2, 0]


def test_view_rejects_noncanonical_packet_order():
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    compiled = _compile(base)
    descriptor = replace(base, asm_forms=(replace(base.asm_forms[0], operands=("rhs", "lhs")),))
    view = replace(TEST_LOW_CORE_DESCRIPTOR_SET, key="test.low.view", descriptors=(descriptor,))
    with pytest.raises(ValueError, match="operands must enumerate canonical packet order"):
        views.descriptor_set_view_for_spec(compiled, view)


@pytest.mark.parametrize(
    "kinds",
    [
        (ConstraintKind.TIED,),
        (ConstraintKind.PREFER_TIED,),
        (ConstraintKind.DESTRUCTIVE,),
        (ConstraintKind.TIED, ConstraintKind.DESTRUCTIVE),
        (ConstraintKind.DESTRUCTIVE, ConstraintKind.TIED),
    ],
)
def test_operand_type_inheritance_distinguishes_ownership(kinds):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base,
        operands=(
            base.operands[0],
            replace(base.operands[1], field_name="state", role=OperandRole.IMPLICIT, flags=(OperandFlag.IMPLICIT,)),
            *base.operands[1:],
        ),
        constraints=tuple(Constraint(kind, 0, 3) for kind in kinds),
    )
    form = _compile(descriptor).asm_forms[0]
    owns_operand = any(kind in (ConstraintKind.TIED, ConstraintKind.PREFER_TIED) for kind in kinds)
    assert form.result_types == (CompiledAsmOperandType(1, tied=owns_operand),)
    assert form.tied_results == ((CompiledAsmTiedResult(0, 1),) if owns_operand else ())
    generated = generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))
    kind = "TIED_OPERAND" if owns_operand else "OPERAND"
    assert f".kind = LOOM_LOW_ASM_RESULT_TYPE_KIND_{kind}," in generated.source
    assert ("static const loom_tied_result_t" in generated.source) == owns_operand


@pytest.mark.parametrize("result_order", permutations(range(3)))
def test_default_ties_keep_constraint_order_and_result_indexed_types(result_order):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    result_names = ("first", "second", "third")
    operand_names = ("left", "middle", "right")
    descriptor = replace(
        base,
        operands=tuple(replace(base.operands[0], field_name=name) for name in result_names) + tuple(replace(base.operands[1], field_name=name) for name in operand_names),
        constraints=tuple(Constraint(ConstraintKind.TIED, result, 3 + (2 - result)) for result in result_order),
        asm_forms=(AsmForm(results=result_names, operands=operand_names),),
    )
    form = _compile(descriptor).asm_forms[0]
    assert form.result_types == tuple(CompiledAsmOperandType(2 - result, tied=True) for result in range(3))
    assert form.tied_results == tuple(CompiledAsmTiedResult(result, 2 - result) for result in result_order)


def test_semantic_and_inherited_recipes_share_one_result_span():
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    payload = AsmResultValueType(ScalarTypeKind.I32, vector_lane_count=4)
    descriptor = replace(
        base,
        operands=(base.operands[0], replace(base.operands[0], field_name="carry"), *base.operands[1:]),
        constraints=(Constraint(ConstraintKind.TIED, 1, 3),),
        asm_forms=(AsmForm(results=("dst", "carry"), operands=("lhs", "rhs"), result_value_types=(payload, None)),),
    )
    compiled = _compile(descriptor)
    assert compiled.asm_table_storage.result_types == [payload, CompiledAsmOperandType(1, tied=True)]
    assert compiled.asm_table_storage.tied_results == [CompiledAsmTiedResult(1, 1)]


@pytest.mark.parametrize("constraint", [Constraint(ConstraintKind.TIED, 0, 1), Constraint(ConstraintKind.TIED, 0, 2), Constraint(ConstraintKind.DESTRUCTIVE, 0, 1)])
def test_layout_sharing_requires_the_same_inheritance_and_ownership(constraint):
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    first = replace(base, constraints=(Constraint(ConstraintKind.TIED, 0, 1),))
    second = replace(base, key="test.other", mnemonic="test.other", constraints=(constraint,))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(first, second)))
    assert (compiled.asm_forms[0].layout_index == compiled.asm_forms[1].layout_index) == (constraint == first.constraints[0])
