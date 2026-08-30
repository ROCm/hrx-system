# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.gen.target.low import compiler
from loom.target.low_descriptors import EncodingFieldValue
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


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
