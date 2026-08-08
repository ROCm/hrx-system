# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
    ORDINARY_VECTOR_INTEGER_CONVERSIONS,
    SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS,
    UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS,
)
from loom.target.arch.spirv.scalar_conversion import (
    SIGNED_INTEGER_WIDTH_CONVERSIONS,
    UNSIGNED_INTEGER_WIDTH_CONVERSIONS,
)


def test_integer_conversion_matrix_is_the_exact_scalar_lane_cross_product() -> None:
    assert len(SIGNED_INTEGER_WIDTH_CONVERSIONS) == 12
    assert len(UNSIGNED_INTEGER_WIDTH_CONVERSIONS) == 6
    assert len(SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS) == 36
    assert len(UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS) == 18
    assert len(ORDINARY_VECTOR_INTEGER_CONVERSIONS) == 54

    assert {
        (conversion.scalar_conversion, conversion.source_type.lane_count)
        for conversion in SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS
    } == {
        (scalar_conversion, lane_count)
        for scalar_conversion in SIGNED_INTEGER_WIDTH_CONVERSIONS
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    }
    assert {
        (conversion.scalar_conversion, conversion.source_type.lane_count)
        for conversion in UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS
    } == {
        (scalar_conversion, lane_count)
        for scalar_conversion in UNSIGNED_INTEGER_WIDTH_CONVERSIONS
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    }


def test_integer_conversion_rows_preserve_semantics_and_native_types() -> None:
    for conversion in SIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS:
        assert conversion.source_type.component_type.kind == (
            OrdinaryVectorComponentKind.SIGNED_INTEGER
        )
        assert conversion.result_type.component_type.kind == (
            OrdinaryVectorComponentKind.SIGNED_INTEGER
        )
        source_width = conversion.source_type.component_type.bit_width
        result_width = conversion.result_type.component_type.bit_width
        if conversion.scalar_conversion.source_op_key == "extsi":
            assert source_width < result_width
        else:
            assert conversion.scalar_conversion.source_op_key == "trunci"
            assert source_width > result_width

    for conversion in UNSIGNED_ORDINARY_VECTOR_INTEGER_CONVERSIONS:
        assert conversion.scalar_conversion.source_op_key == "extui"
        assert conversion.source_type.component_type.kind == (
            OrdinaryVectorComponentKind.UNSIGNED_INTEGER
        )
        assert conversion.result_type.component_type.kind == (
            OrdinaryVectorComponentKind.UNSIGNED_INTEGER
        )
        assert (
            conversion.source_type.component_type.bit_width
            < conversion.result_type.component_type.bit_width
        )

    for conversion in ORDINARY_VECTOR_INTEGER_CONVERSIONS:
        scalar_conversion = conversion.scalar_conversion
        assert conversion.source_type.lane_count == conversion.result_type.lane_count
        assert conversion.source_type.component_type.suffix == (
            scalar_conversion.source_type.suffix
        )
        assert conversion.result_type.component_type.suffix == (
            scalar_conversion.result_type.suffix
        )
        assert conversion.instruction.feature_bits == scalar_conversion.feature_bits


def test_integer_conversion_instructions_are_exact_and_unique() -> None:
    assert len(ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS) == 54
    instruction_keys = tuple(
        instruction.key
        for instruction in ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS
    )
    assert len(set(instruction_keys)) == len(instruction_keys)
    assert set(instruction_keys).isdisjoint(
        instruction.key
        for instruction in (
            *ORDINARY_VECTOR_INSTRUCTIONS,
            *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
        )
    )
    assert Counter(
        instruction.opcode
        for instruction in ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS
    ) == {
        "LOOM_SPIRV_OP_S_CONVERT": 36,
        "LOOM_SPIRV_OP_U_CONVERT": 18,
    }

    for conversion, instruction in zip(
        ORDINARY_VECTOR_INTEGER_CONVERSIONS,
        ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
        strict=True,
    ):
        assert instruction == conversion.instruction
        assert instruction.packet_form == "LOOM_SPIRV_PACKET_FORM_UNARY_TYPED"
        assert instruction.key == (
            f"spirv.op_{conversion.scalar_conversion.descriptor_suffix}."
            f"{conversion.source_type.suffix}.{conversion.result_type.suffix}"
        )
        assert instruction.mnemonic == (
            f"{conversion.scalar_conversion.mnemonic}."
            f"{conversion.source_type.suffix}.{conversion.result_type.suffix}"
        )
        assert instruction.opcode == conversion.scalar_conversion.opcode
        assert instruction.result_type == conversion.result_type
        assert instruction.operand_names == ("input",)
        assert instruction.operand_types == (conversion.source_type,)
