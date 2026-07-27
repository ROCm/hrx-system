# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.spirv.spirv_packet_rows import _descriptor_ref_constant_name, generate_tables
from loom.target.arch.spirv.builtins import BUILTIN_DIMENSIONS, BUILTIN_INDEX_QUERIES
from loom.target.arch.spirv.cooperative_matrix import cooperative_matrix_descriptor_key
from loom.target.arch.spirv.scalar_memory import STORAGE_BUFFER_SCALARS


def _generated_row(tables: str, descriptor_key: str) -> str:
    marker = f"[{_descriptor_ref_constant_name(descriptor_key)}] ="
    start = tables.index(marker)
    end = tables.index("\n        },", start)
    return tables[start:end]


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
    assert "LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT" in tables


def test_generation_emits_index_to_offset_and_builtin_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_UCONVERT_I32_OFFSET64" in tables
    assert "LOOM_SPIRV_VALUE_CLASS_OFFSET64" in tables
    assert "LOOM_SPIRV_OP_U_CONVERT" in tables

    for query in BUILTIN_INDEX_QUERIES:
        for dimension in BUILTIN_DIMENSIONS:
            suffix = f"{query.descriptor_suffix}_{dimension.source_keyword}".upper()
            assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_BUILTIN_{suffix}" in tables
            assert query.builtin_enum in tables
            assert f".component_index = {dimension.component_index}" in tables

    assert "LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN" in tables
