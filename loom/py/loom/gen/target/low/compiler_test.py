# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from itertools import permutations

import pytest

from loom.gen.target.low import compiler
from loom.target.low_descriptors import EncodingFieldValue
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


@pytest.mark.parametrize("candidate_names", permutations(("test.r0", "test.r1", "test.r2", "test.r3")))
def test_physical_packing_order_preserves_pairs_and_semantic_ordinals(candidate_names) -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=tuple(replace(reg_class, physical_registers=candidate_names) if reg_class.name == "test.explicit32" else reg_class for reg_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes),
    )
    compiled = compiler.compile_descriptor_set(descriptor_set)
    class_id = compiled.reg_class_ids["test.explicit32"]
    start = compiled.physical_register_candidate_starts[class_id]
    semantic_ids = compiled.physical_register_candidate_ids[start : start + 4]
    assert tuple(compiled.physical_registers[index].name for index in semantic_ids) == candidate_names
    allocation_order = compiled.physical_register_allocation_ordinals[start : start + 4]
    assert sorted(allocation_order) == list(range(4))
    packed_names = [candidate_names[ordinal] for ordinal in allocation_order]
    assert {frozenset(packed_names[:2]), frozenset(packed_names[2:])} == {frozenset(("test.r0", "test.r2")), frozenset(("test.r1", "test.r3"))}
    ranks = {ordinal: rank for rank, ordinal in enumerate(allocation_order)}
    for view in compiled.physical_register_views:
        if view.reg_class_id != class_id:
            continue
        units = compiled.physical_register_view_unit_candidate_ordinals[view.unit_candidate_ordinal_start : view.unit_candidate_ordinal_start + view.unit_count]
        assert view.packing_rank == min(ranks[ordinal] for ordinal in units)


def test_physical_packing_order_is_independent_of_view_declaration_order() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    reversed_views = compiler.compile_descriptor_set(
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            physical_register_views=tuple(reversed(TEST_LOW_CORE_DESCRIPTOR_SET.physical_register_views)),
        )
    )
    assert compiled.physical_register_allocation_ordinals == reversed_views.physical_register_allocation_ordinals
    assert compiled.physical_register_views == reversed_views.physical_register_views
    for class_id, reg_class in enumerate(compiled.reg_classes):
        start = compiled.physical_register_candidate_starts[class_id]
        count = len(reg_class.physical_registers)
        order = compiled.physical_register_allocation_ordinals[start : start + count]
        assert sorted(order) == list(range(count))
        if reg_class.name == "test.packed.narrow":
            # No aggregate spans this class's two candidates. Its source
            # preference order is already the packing order.
            assert order == [0, 1]


def test_compiler_interns_exact_descriptor_and_asm_spans() -> None:
    add_copy = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        key="test.add.copy.i32",
        mnemonic="test.add.copy.i32",
        semantic_tag="integer.add.copy.i32",
    )
    const = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        feature_mask_words=(0x5,),
        encoding_field_values=(EncodingFieldValue(7, 11),),
    )
    const_copy = replace(
        const,
        key="test.const.copy.i32",
        mnemonic="test.const.copy.i32",
        semantic_tag="integer.const.copy.i32",
    )
    materialized_const = replace(
        const,
        key="test.const.materialized.i32",
        mnemonic="test.const.materialized.i32",
        semantic_tag="integer.const.materialized.i32",
        constraints=(),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            TEST_LOW_ADD_I32_DESCRIPTOR,
            add_copy,
            const,
            const_copy,
            materialized_const,
        ),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)
    rows_by_key = {
        descriptor.key: row
        for descriptor, row in zip(
            compiled.descriptors,
            compiled.descriptor_rows,
            strict=True,
        )
    }
    add_row = rows_by_key[TEST_LOW_ADD_I32_DESCRIPTOR.key]
    add_copy_row = rows_by_key[add_copy.key]
    const_row = rows_by_key[const.key]
    const_copy_row = rows_by_key[const_copy.key]
    materialized_const_row = rows_by_key[materialized_const.key]

    assert add_row["operand_start"] == add_copy_row["operand_start"]
    assert const_row["operand_start"] == const_copy_row["operand_start"]
    assert const_row["operand_start"] != materialized_const_row["operand_start"]
    assert const_row["immediate_start"] == const_copy_row["immediate_start"]
    assert const_row["immediate_start"] == materialized_const_row["immediate_start"]
    assert const_row["constraint_start"] == const_copy_row["constraint_start"]
    assert const_row["feature_mask_word_start"] == const_copy_row["feature_mask_word_start"]
    assert const_row["encoding_field_value_start"] == const_copy_row["encoding_field_value_start"]
    assert len(compiled.operands) == 5
    assert len(compiled.immediates) == 1
    assert len(compiled.constraints) == 1
    assert compiled.feature_mask_words == [0x5]
    assert compiled.encoding_field_values == [EncodingFieldValue(7, 11)]

    asm_forms_by_descriptor = {form.descriptor_ordinal: form for form in compiled.asm_forms}
    add_form = asm_forms_by_descriptor[0]
    add_copy_form = asm_forms_by_descriptor[1]
    const_form = asm_forms_by_descriptor[2]
    const_copy_form = asm_forms_by_descriptor[3]
    assert add_form.result_index_start == add_copy_form.result_index_start
    assert add_form.operand_index_start == add_copy_form.operand_index_start
    assert const_form.result_index_start == const_copy_form.result_index_start
    assert const_form.immediate_start == const_copy_form.immediate_start
    assert compiled.asm_table_storage.operand_indices == [0, 1, 2]
    assert len(compiled.asm_table_storage.immediates) == 1
