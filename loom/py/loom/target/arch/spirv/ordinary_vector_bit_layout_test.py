# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

import pytest

from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES,
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTION_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
    ORDINARY_VECTOR_BIT_LAYOUT_LANE_COUNTS,
    ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_TYPES,
    OrdinaryVectorBitLayoutCase,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_conversion import SCALAR_BITCAST_CONVERSIONS


def test_bit_layout_type_matrix_has_one_exact_representation_per_source() -> None:
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_TYPES) == 32
    assert {
        (value_type.element_type, value_type.lane_count)
        for value_type in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
    } == {
        (element_type, lane_count)
        for element_type in ORDINARY_VECTOR_BIT_LAYOUT_ELEMENT_TYPES
        for lane_count in ORDINARY_VECTOR_BIT_LAYOUT_LANE_COUNTS
    }

    for value_type in ORDINARY_VECTOR_BIT_LAYOUT_TYPES:
        assert value_type.component_type.kind in {
            OrdinaryVectorComponentKind.SIGNED_INTEGER,
            OrdinaryVectorComponentKind.FLOAT,
        }
        assert value_type.total_bit_width == (
            value_type.lane_count * value_type.component_type.bit_width
        )
        if value_type.lane_count == 1:
            assert value_type.value_type is value_type.component_type
            assert value_type.suffix == value_type.component_type.suffix
        else:
            assert isinstance(value_type.value_type, OrdinaryVectorType)
            assert value_type.value_type.lane_count == value_type.lane_count
            assert value_type.suffix.startswith(f"v{value_type.lane_count}")


def test_bit_layout_cases_are_the_exact_equal_total_bit_product() -> None:
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_CASES) == 140
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES) == 32
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES) == 108
    assert {
        (case.source, case.result) for case in ORDINARY_VECTOR_BIT_LAYOUT_CASES
    } == {
        (source, result)
        for source in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
        for result in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
        if source.total_bit_width == result.total_bit_width
    }
    assert Counter(
        case.source.total_bit_width for case in ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES
    ) == {
        16: 12,
        32: 30,
        48: 6,
        64: 42,
        96: 2,
        128: 12,
        192: 2,
        256: 2,
    }
    assert {case.source for case in ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES} == (
        set(ORDINARY_VECTOR_BIT_LAYOUT_TYPES)
    )
    assert all(
        case.source == case.result for case in ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES
    )


def test_bit_layout_reuses_every_existing_scalar_descriptor() -> None:
    assert len(SCALAR_BITCAST_CONVERSIONS) == 8
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES) == 8
    assert {case.key for case in ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES} == {
        row.key for row in SCALAR_BITCAST_CONVERSIONS
    }
    assert all(
        case.source.lane_count == 1 and case.result.lane_count == 1
        for case in ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES
    )
    scalar_rows_by_key = {row.key: row for row in SCALAR_BITCAST_CONVERSIONS}
    for case in ORDINARY_VECTOR_BIT_LAYOUT_REUSED_SCALAR_CASES:
        scalar_row = scalar_rows_by_key[case.key]
        assert scalar_row.display_mnemonic == (
            f"OpBitcast.{case.source.suffix}.{case.result.suffix}"
        )
        assert scalar_row.feature_bits == case.feature_bits


def test_bit_layout_instruction_rows_are_exact_and_unique() -> None:
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTION_CASES) == 100
    assert len(ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS) == 100

    keys = tuple(row.key for row in ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS)
    assert len(set(keys)) == len(keys)
    assert set(keys).isdisjoint(
        row.key
        for row in (
            *ORDINARY_VECTOR_INSTRUCTIONS,
            *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
            *ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
        )
    )
    assert set(keys).isdisjoint(row.key for row in SCALAR_BITCAST_CONVERSIONS)

    for case, instruction in zip(
        ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTION_CASES,
        ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
        strict=True,
    ):
        assert instruction == case.instruction
        assert instruction.key == case.key
        assert instruction.mnemonic == (
            f"OpBitcast.{case.source.suffix}.{case.result.suffix}"
        )
        assert instruction.opcode == "LOOM_SPIRV_OP_BITCAST"
        assert instruction.packet_form == "LOOM_SPIRV_PACKET_FORM_UNARY_TYPED"
        assert instruction.result_type == case.result.value_type
        assert instruction.operand_names == ("input",)
        assert instruction.operand_types == (case.source.value_type,)
        assert instruction.feature_bits == case.feature_bits


def test_identity_bit_layout_cases_have_no_instruction() -> None:
    for case in ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES:
        with pytest.raises(ValueError, match="alias their input"):
            _ = case.instruction


def test_bit_layout_case_rejects_unequal_total_widths() -> None:
    source = next(
        value_type
        for value_type in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
        if value_type.element_type == "i8" and value_type.lane_count == 1
    )
    result = next(
        value_type
        for value_type in ORDINARY_VECTOR_BIT_LAYOUT_TYPES
        if value_type.element_type == "i16" and value_type.lane_count == 1
    )
    with pytest.raises(ValueError, match="equal total bit widths"):
        OrdinaryVectorBitLayoutCase(source, result)
