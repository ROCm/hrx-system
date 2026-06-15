# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.spirv.spirv_cooperative_properties import (
    _descriptor_ref_constant_name,
    _mul_add_descriptor_key,
    generate_tables,
)
from loom.target.arch.spirv.cooperative_matrix import COOPERATIVE_MATRIX_CASES
from loom.target.arch.spirv.cooperative_vector import COOPERATIVE_VECTOR_CASES


def _generated_row(tables: str, property_name: str) -> str:
    marker = f'IREE_SVL("{property_name}")'
    start = tables.index(marker)
    end = tables.index("\n    },", start)
    return tables[start:end]


def test_generation_maps_matrix_properties_to_low_descriptors() -> None:
    tables = generate_tables()

    for case in COOPERATIVE_MATRIX_CASES:
        row = _generated_row(tables, case.property_name)
        assert f".lhs_type = {case.lhs_scalar.scalar_enum}" in row
        assert f".rhs_type = {case.rhs_scalar.scalar_enum}" in row
        assert f".accumulator_type = {case.accumulator_scalar.scalar_enum}" in row
        assert f".result_type = {case.result_scalar.scalar_enum}" in row
        assert f".operand_flags = {case.property_operand_flags}" in row
        assert _descriptor_ref_constant_name(_mul_add_descriptor_key(case)) in row

    assert "{.shape_key = UINT64_C(0x001000100010), .start = 0, .count = 2}" in tables
    assert "{.shape_key = UINT64_C(0x001000100020), .start = 2, .count = 2}" in tables


def test_generation_maps_vector_properties_to_component_rows() -> None:
    tables = generate_tables()

    for case in COOPERATIVE_VECTOR_CASES:
        row = _generated_row(tables, case.property_name)
        assert f".required_feature_bits = {case.feature_bits_c_expression}" in row
        assert f".m_size = {case.m_size}" in row
        assert f".k_size = {case.k_size}" in row
        assert f".input_type = {case.input_type}" in row
        assert f".input_interpretation = {case.input_interpretation}" in row
        assert f".matrix_interpretation = {case.matrix_interpretation}" in row
        assert f".bias_interpretation = {case.bias_interpretation}" in row
        assert f".result_type = {case.result_type}" in row
        assert f".matrix_layout_flags = {case.matrix_layout_flags}" in row
        assert f".storage_class_flags = {case.storage_class_flags}" in row
        assert f".flags = {case.flags}" in row

    assert "{.shape_key = UINT64_C(0x00100010), .start = 0, .count = 1}" in tables
    assert "{.shape_key = UINT64_C(0x00100020), .start = 1, .count = 1}" in tables
    assert "{.shape_key = UINT64_C(0x00200020), .start = 2, .count = 2}" in tables
