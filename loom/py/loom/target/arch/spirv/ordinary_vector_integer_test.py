# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_BOOLEAN_BINARY_INSTRUCTIONS,
    ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS,
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
    ORDINARY_VECTOR_INTEGER_TYPE_PAIRS,
    ORDINARY_VECTOR_INTEGER_VALUE_VIEW_INSTRUCTIONS,
    SIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS,
    SIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS,
    UNSIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS,
    UNSIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS,
)


def test_integer_component_pairs_preserve_signed_source_ownership() -> None:
    assert tuple(
        (
            pair.source_type,
            pair.signed.suffix,
            pair.unsigned.suffix,
            pair.signed.scalar_enum,
            pair.unsigned.scalar_enum,
            pair.feature_bits,
        )
        for pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
    ) == (
        (
            "i8",
            "i8",
            "u8",
            "LOOM_SPIRV_SCALAR_TYPE_S8",
            "LOOM_SPIRV_SCALAR_TYPE_U8",
            feature_bits_value(("int8",)),
        ),
        (
            "i16",
            "i16",
            "u16",
            "LOOM_SPIRV_SCALAR_TYPE_S16",
            "LOOM_SPIRV_SCALAR_TYPE_U16",
            feature_bits_value(("int16",)),
        ),
        (
            "i32",
            "i32",
            "u32",
            "LOOM_SPIRV_SCALAR_TYPE_S32",
            "LOOM_SPIRV_SCALAR_TYPE_U32",
            0,
        ),
        (
            "i64",
            "i64",
            "u64",
            "LOOM_SPIRV_SCALAR_TYPE_S64",
            "LOOM_SPIRV_SCALAR_TYPE_U64",
            feature_bits_value(("int64",)),
        ),
    )

    source_components = set(ORDINARY_VECTOR_COMPONENT_TYPES)
    for pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS:
        assert pair.signed.kind == OrdinaryVectorComponentKind.SIGNED_INTEGER
        assert pair.signed in source_components
        assert pair.unsigned.kind == OrdinaryVectorComponentKind.UNSIGNED_INTEGER
        assert pair.unsigned not in source_components
        assert pair.unsigned.source_types == ()
        assert pair.signed.bit_width == pair.unsigned.bit_width
        assert pair.signed.feature_bits == pair.unsigned.feature_bits


def test_native_integer_type_pairs_are_the_width_lane_cross_product() -> None:
    assert len(ORDINARY_VECTOR_INTEGER_TYPE_PAIRS) == (
        len(ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS)
        * len(NATIVE_ORDINARY_VECTOR_LANE_COUNTS)
    )
    assert {
        (pair.source_type, pair.lane_count)
        for pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS
    } == {
        (source_type, lane_count)
        for source_type in ("i8", "i16", "i32", "i64")
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
    }
    for pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS:
        assert pair.signed.lane_count == pair.lane_count
        assert pair.unsigned.lane_count == pair.lane_count
        assert pair.signed.suffix == (
            f"v{pair.lane_count}{pair.component_pair.signed.suffix}"
        )
        assert pair.unsigned.suffix == (
            f"v{pair.lane_count}{pair.component_pair.unsigned.suffix}"
        )


def test_native_integer_instruction_matrix_is_exact_and_unique() -> None:
    assert len(SIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS) == 132
    assert len(UNSIGNED_ORDINARY_VECTOR_INTEGER_BINARY_INSTRUCTIONS) == 24
    assert len(ORDINARY_VECTOR_INTEGER_VALUE_VIEW_INSTRUCTIONS) == 24
    assert len(SIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS) == 72
    assert len(UNSIGNED_ORDINARY_VECTOR_INTEGER_COMPARE_INSTRUCTIONS) == 48
    assert len(ORDINARY_VECTOR_BOOLEAN_BINARY_INSTRUCTIONS) == 9
    assert len(ORDINARY_VECTOR_INTEGER_INSTRUCTIONS) == 309

    instruction_keys = tuple(
        instruction.key for instruction in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS
    )
    assert len(set(instruction_keys)) == len(instruction_keys)
    assert set(instruction_keys).isdisjoint(
        instruction.key for instruction in ORDINARY_VECTOR_INSTRUCTIONS
    )

    packet_forms = Counter(
        instruction.packet_form for instruction in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS
    )
    assert packet_forms == {
        "LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE": 165,
        "LOOM_SPIRV_PACKET_FORM_UNARY_TYPED": 24,
        "LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE": 120,
    }


def test_each_native_integer_type_has_the_complete_primitive_vocabulary() -> None:
    instructions_by_suffix: dict[str, set[str]] = {}
    for instruction in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS:
        for operand_type in instruction.operand_types:
            if operand_type.component_type.kind in {
                OrdinaryVectorComponentKind.SIGNED_INTEGER,
                OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
            }:
                instructions_by_suffix.setdefault(operand_type.suffix, set()).add(
                    instruction.key
                )
                break

    signed_operations = {
        "iadd",
        "isub",
        "imul",
        "sdiv",
        "srem",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "shift_left_logical",
        "shift_right_arithmetic",
        "shift_right_logical",
        "i_equal",
        "i_not_equal",
        "s_less_than",
        "s_less_than_equal",
        "s_greater_than",
        "s_greater_than_equal",
    }
    unsigned_operations = {
        "udiv",
        "umod",
        "u_less_than",
        "u_less_than_equal",
        "u_greater_than",
        "u_greater_than_equal",
    }
    for type_pair in ORDINARY_VECTOR_INTEGER_TYPE_PAIRS:
        signed_suffix = type_pair.signed.suffix
        unsigned_suffix = type_pair.unsigned.suffix
        assert instructions_by_suffix[signed_suffix] == {
            *(
                f"spirv.op_{operation}.{signed_suffix}"
                for operation in signed_operations
            ),
            f"spirv.op_bitcast.{signed_suffix}.{unsigned_suffix}",
        }
        assert instructions_by_suffix[unsigned_suffix] == {
            *(
                f"spirv.op_{operation}.{unsigned_suffix}"
                for operation in unsigned_operations
            ),
            f"spirv.op_bitcast.{unsigned_suffix}.{signed_suffix}",
        }

    for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
        suffix = f"v{lane_count}bool"
        assert {
            instruction.key
            for instruction in ORDINARY_VECTOR_BOOLEAN_BINARY_INSTRUCTIONS
            if instruction.result_type.suffix == suffix
        } == {
            f"spirv.op_logical_and.{suffix}",
            f"spirv.op_logical_or.{suffix}",
            f"spirv.op_logical_not_equal.{suffix}",
        }


def test_integer_instruction_rows_keep_every_value_native() -> None:
    for instruction in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS:
        assert isinstance(instruction.result_type, OrdinaryVectorType)
        assert instruction.result_type.lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS
        for operand_type in instruction.operand_types:
            assert isinstance(operand_type, OrdinaryVectorType)
            assert operand_type.lane_count == instruction.result_type.lane_count

    view_directions = Counter(
        (
            instruction.operand_types[0].component_type.kind,
            instruction.result_type.component_type.kind,
        )
        for instruction in ORDINARY_VECTOR_INTEGER_VALUE_VIEW_INSTRUCTIONS
    )
    assert view_directions == {
        (
            OrdinaryVectorComponentKind.SIGNED_INTEGER,
            OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
        ): 12,
        (
            OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
            OrdinaryVectorComponentKind.SIGNED_INTEGER,
        ): 12,
    }


def test_integer_instruction_feature_bits_follow_the_component_width() -> None:
    feature_bits_by_source = {
        pair.source_type: pair.feature_bits
        for pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
    }
    for instruction in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS:
        numeric_types = tuple(
            value_type
            for value_type in (instruction.result_type, *instruction.operand_types)
            if value_type.component_type.kind
            in {
                OrdinaryVectorComponentKind.SIGNED_INTEGER,
                OrdinaryVectorComponentKind.UNSIGNED_INTEGER,
            }
        )
        if not numeric_types:
            assert instruction.feature_bits == 0
            continue
        source_type = next(
            component_pair.source_type
            for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
            if component_pair.signed == numeric_types[0].component_type
            or component_pair.unsigned == numeric_types[0].component_type
        )
        assert instruction.feature_bits == feature_bits_by_source[source_type]
