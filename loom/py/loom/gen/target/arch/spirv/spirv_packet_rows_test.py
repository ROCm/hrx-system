# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from loom.gen.target.arch.spirv.spirv_packet_rows import (
    _PACKETLESS_DESCRIPTOR_KEYS,
    _descriptor_ref_constant_name,
    _packet_rows,
    _PacketRow,
    _validate_rows,
    generate_tables,
)
from loom.target.arch.spirv.builtins import BUILTIN_DIMENSIONS, BUILTIN_INDEX_QUERIES
from loom.target.arch.spirv.cooperative_matrix import cooperative_matrix_descriptor_key
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_INSTRUCTIONS,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentKind,
    OrdinaryVectorInstruction,
    OrdinaryVectorInstructionType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
)
from loom.target.arch.spirv.scalar_constant import FLOAT_CONSTANT_TYPES
from loom.target.arch.spirv.scalar_memory import (
    RAW_STORAGE_BUFFER_BYTE,
    STORAGE_BUFFER_SCALARS,
)
from loom.target.low_descriptors import Descriptor


def _expect_row_validation_error(rows: tuple[_PacketRow, ...], expected_message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        _validate_rows(rows)


def _row_index(rows: tuple[_PacketRow, ...], descriptor_key: str) -> int:
    for index, row in enumerate(rows):
        if row.descriptor_key == descriptor_key:
            return index
    raise AssertionError(f"missing packet-row fixture '{descriptor_key}'")


def test_validation_rejects_duplicate_packet_descriptor_keys() -> None:
    rows = _packet_rows()
    i32_constant_index = _row_index(rows, "spirv.op_constant.i32")

    _expect_row_validation_error(
        (*rows, rows[i32_constant_index]),
        "SPIR-V packet descriptor keys must be unique: spirv.op_constant.i32",
    )


def test_validation_rejects_packet_rows_without_descriptors() -> None:
    rows = _packet_rows()
    rows_with_unknown_descriptor = (
        replace(rows[0], descriptor_key="spirv.op_missing.test"),
        *rows[1:],
    )

    _expect_row_validation_error(
        rows_with_unknown_descriptor,
        "SPIR-V packet rows reference missing descriptors: spirv.op_missing.test",
    )


def test_validation_rejects_descriptors_without_packet_rows() -> None:
    rows = _packet_rows()
    i32_constant_index = _row_index(rows, "spirv.op_constant.i32")
    rows_without_i32_constant = (
        *rows[:i32_constant_index],
        *rows[i32_constant_index + 1 :],
    )

    _expect_row_validation_error(
        rows_without_i32_constant,
        "SPIR-V descriptors are missing packet rows: spirv.op_constant.i32",
    )


def test_validation_preserves_the_deliberate_packetless_descriptor() -> None:
    assert _PACKETLESS_DESCRIPTOR_KEYS == frozenset({"spirv.op_variable.function.ptr"})
    rows = _packet_rows()
    rows_with_packetless_descriptor = (
        replace(
            rows[0],
            descriptor_key="spirv.op_variable.function.ptr",
        ),
        *rows[1:],
    )

    _expect_row_validation_error(
        rows_with_packetless_descriptor,
        "SPIR-V packetless descriptors must not have packet rows: spirv.op_variable.function.ptr",
    )


def test_validation_rejects_rows_exceeding_packet_operand_maximum() -> None:
    rows = _packet_rows()
    rows_with_five_operands = (
        replace(rows[0], operand_types=("value",) * 5),
        *rows[1:],
    )

    _expect_row_validation_error(
        rows_with_five_operands,
        "SPIR-V packet rows exceed the 4-operand maximum: spirv.op_constant_false.bool",
    )


def test_validation_rejects_heterogeneous_four_operand_types() -> None:
    rows = _packet_rows()
    rows_with_unencodable_types = (
        replace(rows[0], operand_types=("first", "second", "third", "fourth")),
        *rows[1:],
    )

    _expect_row_validation_error(
        rows_with_unencodable_types,
        "SPIR-V packet rows exceed the operand-type capacity without a repeated type: spirv.op_constant_false.bool",
    )


def _generated_row(tables: str, descriptor_key: str) -> str:
    marker = f"[{_descriptor_ref_constant_name(descriptor_key)}] ="
    start = tables.index(marker)
    end = tables.index("\n        },", start)
    return tables[start:end]


def _descriptor(descriptor_key: str) -> Descriptor:
    for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors:
        if descriptor.key == descriptor_key:
            return descriptor
    raise AssertionError(f"missing descriptor fixture '{descriptor_key}'")


def _cooperative_matrix_descriptor(
    op_name: str,
    *,
    role: str | None,
    element: str,
    k_size: int,
    accumulator: str,
    layout: str | None = None,
) -> str:
    return cooperative_matrix_descriptor_key(
        op_name,
        role=role,
        element=element,
        m_size=16,
        n_size=16,
        k_size=k_size,
        accumulator=accumulator,
        scope="subgroup",
        layout=layout,
    )


def test_generation_emits_scalar_memory_packet_rows() -> None:
    tables = generate_tables()

    for scalar in STORAGE_BUFFER_SCALARS:
        suffix = scalar.suffix.upper()
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_{suffix}_BYTE_OFFSET" in tables
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_STORAGE_BUFFER_{suffix}" in tables
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_STORE_STORAGE_BUFFER_{suffix}" in tables
        assert scalar.scalar_enum in tables
        assert f".memory_alignment = {scalar.byte_width}" in tables

    assert "LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS" in tables
    assert "LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER" in tables


def test_generation_emits_raw_storage_byte_bridge_rows() -> None:
    suffix = RAW_STORAGE_BUFFER_BYTE.suffix
    tables = generate_tables()
    load_row = _generated_row(
        tables,
        f"spirv.op_load.storage_buffer.{suffix}",
    )
    assert "LOOM_SPIRV_SCALAR_TYPE_U8" in load_row
    assert ".memory_alignment = 1" in load_row

    extend_row = _generated_row(tables, f"spirv.op_uconvert.{suffix}.u32")
    assert "LOOM_SPIRV_SCALAR_TYPE_U8" in extend_row
    assert "LOOM_SPIRV_SCALAR_TYPE_U32" in extend_row

    narrow_row = _generated_row(tables, f"spirv.op_uconvert.u32.{suffix}")
    assert "LOOM_SPIRV_SCALAR_TYPE_U32" in narrow_row
    assert "LOOM_SPIRV_SCALAR_TYPE_U8" in narrow_row


def test_generation_emits_complete_float_constant_packet_rows() -> None:
    tables = generate_tables()

    for scalar in FLOAT_CONSTANT_TYPES:
        row = _generated_row(tables, f"spirv.op_constant.{scalar.suffix}")
        assert "LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT" in row
        assert scalar.scalar_enum in row
        assert f".literal_word_count = {scalar.literal_word_count}" in row

        descriptor = _descriptor(f"spirv.op_constant.{scalar.suffix}")
        assert descriptor.feature_mask_words == ((scalar.feature_bits,) if scalar.feature_bits else ())


def test_generation_uses_byte_strides_for_cooperative_matrix_rows() -> None:
    tables = generate_tables()

    f16_lhs = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="lhs",
            element="f16",
            k_size=16,
            accumulator="f32",
            layout="row_major",
        ),
    )
    f16_init = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="init",
            element="f16",
            k_size=16,
            accumulator="f32",
            layout="row_major",
        ),
    )
    f16_store = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_store_khr",
            role="result",
            element="f16",
            k_size=16,
            accumulator="f32",
            layout="row_major",
        ),
    )
    assert ".cooperative_matrix_stride = 32" in f16_lhs
    assert ".cooperative_matrix_stride = 64" in f16_init
    assert ".cooperative_matrix_stride = 64" in f16_store

    bf16_rhs = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="rhs",
            element="bf16",
            k_size=16,
            accumulator="f32",
            layout="row_major",
        ),
    )
    assert ".cooperative_matrix_stride = 32" in bf16_rhs

    s8_lhs = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="lhs",
            element="s8",
            k_size=32,
            accumulator="s32",
            layout="row_major",
        ),
    )
    s8_rhs = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="rhs",
            element="s8",
            k_size=32,
            accumulator="s32",
            layout="row_major",
        ),
    )
    s8_store = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_store_khr",
            role="result",
            element="s8",
            k_size=32,
            accumulator="s32",
            layout="row_major",
        ),
    )
    assert ".cooperative_matrix_stride = 32" in s8_lhs
    assert ".cooperative_matrix_stride = 16" in s8_rhs
    assert ".cooperative_matrix_stride = 64" in s8_store

    u8_lhs = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_load_khr",
            role="lhs",
            element="u8",
            k_size=32,
            accumulator="u32",
            layout="row_major",
        ),
    )
    u8_store = _generated_row(
        tables,
        _cooperative_matrix_descriptor(
            "op_cooperative_matrix_store_khr",
            role="result",
            element="u8",
            k_size=32,
            accumulator="u32",
            layout="row_major",
        ),
    )
    assert "LOOM_SPIRV_SCALAR_TYPE_U8" in u8_lhs
    assert "LOOM_SPIRV_SCALAR_TYPE_U32" in u8_store
    assert ".cooperative_matrix_stride = 32" in u8_lhs
    assert ".cooperative_matrix_stride = 64" in u8_store


def test_generation_keeps_storage_buffer_address_untyped_until_access_chain() -> None:
    tables = generate_tables()

    access_row_start = tables.index("SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_F32_BYTE_OFFSET")
    access_row = tables[access_row_start : access_row_start + 800]
    assert "LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS" in access_row
    assert "LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER" in access_row
    assert "LOOM_SPIRV_SCALAR_TYPE_F32" in access_row


def test_generation_emits_coordinate_arithmetic_packet_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ISUB_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_FADD_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_FMUL_F64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SDIV_I16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SREM_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_UDIV_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_UMOD_U64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IMUL_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IMUL_ADD_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SHIFT_LEFT_LOGICAL_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ISUB_OFFSET64" in tables
    assert "LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD" in tables


def test_generation_emits_complete_integer_boolean_packet_matrix() -> None:
    tables = generate_tables()
    assert tuple(
        (
            row.source_type,
            row.bit_width,
            row.signed.suffix,
            row.signed.scalar_enum,
            row.signed.feature_atoms,
            row.unsigned.suffix,
            row.unsigned.scalar_enum,
            row.unsigned.feature_atoms,
        )
        for row in INTEGER_SCALAR_ALU_TYPE_PAIRS
    ) == (
        ("i8", 8, "i8", "LOOM_SPIRV_SCALAR_TYPE_S8", ("int8",), "u8", "LOOM_SPIRV_SCALAR_TYPE_U8", ("int8",)),
        ("i16", 16, "i16", "LOOM_SPIRV_SCALAR_TYPE_S16", ("int16",), "u16", "LOOM_SPIRV_SCALAR_TYPE_U16", ("int16",)),
        ("i32", 32, "i32", "LOOM_SPIRV_SCALAR_TYPE_S32", (), "u32", "LOOM_SPIRV_SCALAR_TYPE_U32", ()),
        ("i64", 64, "i64", "LOOM_SPIRV_SCALAR_TYPE_S64", ("int64",), "u64", "LOOM_SPIRV_SCALAR_TYPE_U64", ("int64",)),
    )
    assert tuple(
        (
            row.source_op_key,
            row.descriptor_suffix,
            row.mnemonic,
            row.opcode,
        )
        for row in INTEGER_BITWISE_BINARY_OPERATIONS
    ) == (
        ("andi", "bitwise_and", "OpBitwiseAnd", "LOOM_SPIRV_OP_BITWISE_AND"),
        ("ori", "bitwise_or", "OpBitwiseOr", "LOOM_SPIRV_OP_BITWISE_OR"),
        ("xori", "bitwise_xor", "OpBitwiseXor", "LOOM_SPIRV_OP_BITWISE_XOR"),
        ("shli", "shift_left_logical", "OpShiftLeftLogical", "LOOM_SPIRV_OP_SHIFT_LEFT_LOGICAL"),
        ("shrsi", "shift_right_arithmetic", "OpShiftRightArithmetic", "LOOM_SPIRV_OP_SHIFT_RIGHT_ARITHMETIC"),
        ("shrui", "shift_right_logical", "OpShiftRightLogical", "LOOM_SPIRV_OP_SHIFT_RIGHT_LOGICAL"),
    )
    assert tuple(
        (
            row.source_op_key,
            row.descriptor_suffix,
            row.mnemonic,
            row.opcode,
        )
        for row in BOOLEAN_BINARY_OPERATIONS
    ) == (
        ("andi", "logical_and", "OpLogicalAnd", "LOOM_SPIRV_OP_LOGICAL_AND"),
        ("ori", "logical_or", "OpLogicalOr", "LOOM_SPIRV_OP_LOGICAL_OR"),
        ("xori", "logical_not_equal", "OpLogicalNotEqual", "LOOM_SPIRV_OP_LOGICAL_NOT_EQUAL"),
    )
    assert tuple((row.value, row.descriptor_suffix, row.mnemonic, row.opcode) for row in BOOLEAN_CONSTANTS) == (
        (0, "false", "OpConstantFalse", "LOOM_SPIRV_OP_CONSTANT_FALSE"),
        (1, "true", "OpConstantTrue", "LOOM_SPIRV_OP_CONSTANT_TRUE"),
    )

    for constant in BOOLEAN_CONSTANTS:
        descriptor_key = f"spirv.op_constant_{constant.descriptor_suffix}.bool"
        row = _generated_row(
            tables,
            descriptor_key,
        )
        assert constant.opcode in row
        assert "LOOM_SPIRV_PACKET_FORM_BOOLEAN_CONSTANT" in row
        assert "LOOM_SPIRV_VALUE_CLASS_BOOL" in row
        descriptor = _descriptor(descriptor_key)
        assert descriptor.mnemonic == f"{constant.mnemonic}.bool"
        assert descriptor.feature_mask_words == ()

    for operation in BOOLEAN_BINARY_OPERATIONS:
        descriptor_key = f"spirv.op_{operation.descriptor_suffix}.bool"
        row = _generated_row(
            tables,
            descriptor_key,
        )
        assert operation.opcode in row
        assert row.count("LOOM_SPIRV_VALUE_CLASS_BOOL") == 3
        descriptor = _descriptor(descriptor_key)
        assert descriptor.mnemonic == f"{operation.mnemonic}.bool"
        assert descriptor.feature_mask_words == ()

    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        signed = scalar_pair.signed
        expected_feature_mask = (signed.feature_bits,) if signed.feature_bits else ()
        constant_descriptor_key = f"spirv.op_constant.{signed.suffix}"
        constant_row = _generated_row(
            tables,
            constant_descriptor_key,
        )
        assert signed.scalar_enum in constant_row
        assert f".literal_word_count = {scalar_pair.literal_word_count}" in constant_row
        constant_descriptor = _descriptor(constant_descriptor_key)
        assert constant_descriptor.mnemonic == f"OpConstant.{signed.suffix}"
        assert constant_descriptor.feature_mask_words == expected_feature_mask
        assert len(constant_descriptor.immediates) == 1
        constant_immediate = constant_descriptor.immediates[0]
        assert constant_immediate.field_name == f"{scalar_pair.source_type}_value"
        assert constant_immediate.bit_width == scalar_pair.bit_width
        assert constant_immediate.signed_min == scalar_pair.signed_minimum
        assert constant_immediate.unsigned_max == scalar_pair.signed_maximum

        for operation in INTEGER_BITWISE_BINARY_OPERATIONS:
            descriptor_key = f"spirv.op_{operation.descriptor_suffix}.{signed.suffix}"
            operation_row = _generated_row(
                tables,
                descriptor_key,
            )
            assert operation.opcode in operation_row
            assert operation_row.count(signed.scalar_enum) == 3
            descriptor = _descriptor(descriptor_key)
            expected_mnemonic = operation.mnemonic if signed.suffix == "i32" else f"{operation.mnemonic}.{signed.suffix}"
            assert descriptor.mnemonic == expected_mnemonic
            assert descriptor.feature_mask_words == expected_feature_mask

        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES:
            descriptor_key = f"spirv.op_{predicate.descriptor_suffix}.{signed.suffix}"
            compare_row = _generated_row(
                tables,
                descriptor_key,
            )
            assert predicate.opcode in compare_row
            assert compare_row.count(signed.scalar_enum) == 2
            descriptor = _descriptor(descriptor_key)
            assert descriptor.feature_mask_words == expected_feature_mask

        for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES:
            descriptor_key = f"spirv.op_{predicate.descriptor_suffix}.{scalar_pair.unsigned.suffix}"
            compare_row = _generated_row(
                tables,
                descriptor_key,
            )
            assert predicate.opcode in compare_row
            assert compare_row.count(scalar_pair.unsigned.scalar_enum) == 2
            descriptor = _descriptor(descriptor_key)
            assert descriptor.feature_mask_words == expected_feature_mask

        select_descriptor_key = f"spirv.op_select.{signed.suffix}"
        select_row = _generated_row(
            tables,
            select_descriptor_key,
        )
        assert select_row.count(signed.scalar_enum) == 3
        select_descriptor = _descriptor(select_descriptor_key)
        assert select_descriptor.feature_mask_words == expected_feature_mask

    bool_select_row = _generated_row(tables, "spirv.op_select.bool")
    assert bool_select_row.count("LOOM_SPIRV_VALUE_CLASS_BOOL") == 4
    assert _descriptor("spirv.op_select.bool").feature_mask_words == ()


def test_generation_emits_integer_compare_and_select_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_LESS_THAN_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_GREATER_THAN_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_LESS_THAN_U8" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_LESS_THAN_EQUAL_OFFSET64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_OFFSET64" in tables
    assert "LOOM_SPIRV_VALUE_CLASS_BOOL" in tables
    assert "LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE" in tables
    assert "LOOM_SPIRV_PACKET_FORM_SELECT" in tables


def test_generation_emits_complete_index_numeric_primitives() -> None:
    tables = generate_tables()

    coordinate_copy = _generated_row(tables, "spirv.op_copy_object.i32")
    assert "LOOM_SPIRV_OP_COPY_OBJECT" in coordinate_copy
    assert coordinate_copy.count("LOOM_SPIRV_SCALAR_TYPE_S32") == 2

    offset_multiply = _generated_row(tables, "spirv.op_imul.offset64")
    assert "LOOM_SPIRV_OP_I_MUL" in offset_multiply
    assert offset_multiply.count("LOOM_SPIRV_SCALAR_TYPE_U64") == 3

    bit_count = _generated_row(tables, "spirv.op_bit_count.i32")
    assert "LOOM_SPIRV_OP_BIT_COUNT" in bit_count
    assert bit_count.count("LOOM_SPIRV_SCALAR_TYPE_S32") == 2


def test_generation_emits_scalar_conversion_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I8_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I64_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_S_TO_F_I16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_F_TO_S_F32_I16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_CONVERT_U8_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_U_TO_F_U16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_F_TO_U_F32_U16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_I32_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_U32_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_F_CONVERT_F16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_F_CONVERT_F32_F16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_I32_F32" in tables
    assert "LOOM_SPIRV_PACKET_FORM_UNARY_TYPED" in tables


def test_generation_emits_complete_ordinary_vector_structural_matrix() -> None:
    tables = generate_tables()
    packet_rows_by_key = {row.descriptor_key: row for row in _packet_rows() if row.descriptor_key in {instruction.key for instruction in ORDINARY_VECTOR_INSTRUCTIONS}}

    assert len(ORDINARY_VECTOR_TYPES) == 30
    assert len(ORDINARY_VECTOR_INSTRUCTIONS) == 120
    assert len(packet_rows_by_key) == len(ORDINARY_VECTOR_INSTRUCTIONS)
    assert ('static_assert(LOOM_SPIRV_PACKET_MAX_OPERAND_COUNT == 4, "generated packet operand maximum");') in tables
    assert ('static_assert(LOOM_SPIRV_PACKET_OPERAND_TYPE_CAPACITY == 3, "generated packet operand-type capacity");') in tables
    assert "operand_type_count" not in tables

    for vector_type in ORDINARY_VECTOR_TYPES:
        component_type = vector_type.component_type
        construct_key = f"spirv.op_composite_construct.{vector_type.suffix}"
        extract_key = f"spirv.op_composite_extract.{vector_type.suffix}.{component_type.suffix}"
        insert_key = f"spirv.op_composite_insert.{component_type.suffix}.{vector_type.suffix}"
        select_key = f"spirv.op_select.{vector_type.suffix}"

        construct = _generated_row(tables, construct_key)
        assert "LOOM_SPIRV_OP_COMPOSITE_CONSTRUCT" in construct
        assert "LOOM_SPIRV_PACKET_FORM_COMPOSITE_CONSTRUCT" in construct
        assert component_type.vector_value_class in construct
        assert component_type.scalar_value_class in construct
        assert f".lane_count = {vector_type.lane_count}" in construct
        assert f".operand_count = {vector_type.lane_count}" in construct

        extract = _generated_row(tables, extract_key)
        assert "LOOM_SPIRV_OP_COMPOSITE_EXTRACT" in extract
        assert "LOOM_SPIRV_PACKET_FORM_COMPOSITE_EXTRACT" in extract
        assert ".operand_count = 1" in extract
        assert ".immediate_index = 0" in extract

        insert = _generated_row(tables, insert_key)
        assert "LOOM_SPIRV_OP_COMPOSITE_INSERT" in insert
        assert "LOOM_SPIRV_PACKET_FORM_COMPOSITE_INSERT" in insert
        assert ".operand_count = 2" in insert
        assert ".immediate_index = 0" in insert

        select = _generated_row(tables, select_key)
        assert "LOOM_SPIRV_OP_SELECT" in select
        assert "LOOM_SPIRV_PACKET_FORM_SELECT" in select
        assert "LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR" in select
        assert ".operand_count = 3" in select

        expected_features = (component_type.feature_bits,) if component_type.feature_bits else ()
        for descriptor_key in (construct_key, extract_key, insert_key, select_key):
            assert _descriptor(descriptor_key).feature_mask_words == expected_features

        component_index_maximum = vector_type.lane_count - 1
        for descriptor_key in (extract_key, insert_key):
            descriptor = _descriptor(descriptor_key)
            assert len(descriptor.immediates) == 1
            assert descriptor.immediates[0].unsigned_max == component_index_maximum

        if component_type.kind == OrdinaryVectorComponentKind.OFFSET:
            construct_registers = tuple(operand.reg_alts[0].reg_class for operand in _descriptor(construct_key).operands)
            extract_registers = tuple(operand.reg_alts[0].reg_class for operand in _descriptor(extract_key).operands)
            insert_registers = tuple(operand.reg_alts[0].reg_class for operand in _descriptor(insert_key).operands)
            assert construct_registers == (
                "spirv.id",
                *(("spirv.offset64",) * vector_type.lane_count),
            )
            assert extract_registers == ("spirv.offset64", "spirv.id")
            assert insert_registers == (
                "spirv.id",
                "spirv.offset64",
                "spirv.id",
            )


def _expected_ordinary_vector_value(
    value_type: OrdinaryVectorInstructionType,
) -> str:
    if isinstance(value_type, OrdinaryVectorType):
        component_type = value_type.component_type
        return f"{{.value_class = {component_type.vector_value_class}, .scalar_type = {component_type.scalar_enum}, .vector = {{.lane_count = {value_type.lane_count}}}}}"
    return f"{{.value_class = {value_type.scalar_value_class}, .scalar_type = {value_type.scalar_enum}}}"


def _assert_generated_ordinary_vector_instructions(
    instructions: tuple[OrdinaryVectorInstruction, ...],
) -> None:
    instruction_keys = {instruction.key for instruction in instructions}
    packet_rows_by_key = {row.descriptor_key: row for row in _packet_rows() if row.descriptor_key in instruction_keys}
    descriptors_by_key = {descriptor.key: descriptor for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors if descriptor.key in instruction_keys}

    assert packet_rows_by_key.keys() == instruction_keys
    assert descriptors_by_key.keys() == instruction_keys

    for instruction in instructions:
        row = packet_rows_by_key[instruction.key]
        assert row.opcode == instruction.opcode
        assert row.form == instruction.packet_form
        assert row.result_type == _expected_ordinary_vector_value(instruction.result_type)
        assert row.operand_types == tuple(_expected_ordinary_vector_value(operand_type) for operand_type in instruction.operand_types)
        assert row.result_count == 1
        assert row.immediate_index is None

        descriptor = descriptors_by_key[instruction.key]
        assert descriptor.mnemonic == instruction.mnemonic
        assert descriptor.semantic_tag == instruction.key
        assert descriptor.feature_mask_words == ((instruction.feature_bits,) if instruction.feature_bits else ())
        assert descriptor.immediates == ()
        assert tuple(operand.reg_alts[0].reg_class for operand in descriptor.operands) == ("spirv.id",) * (len(instruction.operand_types) + 1)


def test_generation_emits_complete_ordinary_vector_integer_matrix() -> None:
    assert len(ORDINARY_VECTOR_INTEGER_INSTRUCTIONS) == 309
    _assert_generated_ordinary_vector_instructions(ORDINARY_VECTOR_INTEGER_INSTRUCTIONS)


def test_generation_emits_complete_ordinary_vector_integer_conversions() -> None:
    assert len(ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS) == 54
    _assert_generated_ordinary_vector_instructions(ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS)


def test_generation_emits_complete_ordinary_vector_bit_layout_rows() -> None:
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS) == 102
    _assert_generated_ordinary_vector_instructions(ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS)


def test_generation_compacts_only_repeated_four_operand_types() -> None:
    packet_rows_by_key = {row.descriptor_key: row for row in _packet_rows()}

    for vector_type in ORDINARY_VECTOR_TYPES:
        construct = packet_rows_by_key[f"spirv.op_composite_construct.{vector_type.suffix}"]
        assert len(construct.operand_types) == vector_type.lane_count
        if vector_type.lane_count == 4:
            assert construct.encoded_operand_types() == construct.operand_types[:1]
        else:
            assert construct.encoded_operand_types() == construct.operand_types


def test_generation_emits_complete_address_conversion_rows() -> None:
    tables = generate_tables()

    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        scalar = scalar_pair.signed if scalar_pair.bit_width == 64 else scalar_pair.unsigned
        operation = "bitcast" if scalar_pair.bit_width == 64 else "uconvert"
        opcode = "LOOM_SPIRV_OP_BITCAST" if scalar_pair.bit_width == 64 else "LOOM_SPIRV_OP_U_CONVERT"
        expected_feature_mask = (scalar.feature_bits,) if scalar.feature_bits else ()
        for descriptor_key in (
            f"spirv.op_{operation}.{scalar.suffix}.offset64",
            f"spirv.op_{operation}.offset64.{scalar.suffix}",
        ):
            row = _generated_row(tables, descriptor_key)
            assert opcode in row
            assert "LOOM_SPIRV_PACKET_FORM_UNARY_TYPED" in row
            assert scalar.scalar_enum in row
            assert "LOOM_SPIRV_VALUE_CLASS_OFFSET64" in row
            assert _descriptor(descriptor_key).feature_mask_words == expected_feature_mask

    for query in BUILTIN_INDEX_QUERIES:
        for dimension in BUILTIN_DIMENSIONS:
            suffix = f"{query.descriptor_suffix}_{dimension.source_keyword}".upper()
            assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_BUILTIN_{suffix}" in tables
            assert query.builtin_enum in tables
            assert f".component_index = {dimension.component_index}" in tables

    assert "LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN" in tables
