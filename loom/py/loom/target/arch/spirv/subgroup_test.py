# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.ordinary_vector import OrdinaryVectorType
from loom.target.arch.spirv.subgroup import (
    SPIRV_SUBGROUP_ARITHMETIC,
    SPIRV_SUBGROUP_BALLOT_BIT_EXTRACT_INSTRUCTION,
    SPIRV_SUBGROUP_BALLOT_INSTRUCTION,
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS,
    SPIRV_SUBGROUP_BROADCAST_FIRST_INSTRUCTIONS,
    SPIRV_SUBGROUP_SHUFFLES,
    SPIRV_SUBGROUP_U32_TYPE,
    SPIRV_SUBGROUP_U64_TYPE,
    SPIRV_SUBGROUP_V2U32_TYPE,
    SPIRV_SUBGROUP_V4U32_TYPE,
    SPIRV_SUBGROUP_VOTE_INSTRUCTIONS,
)


def test_ballot_uses_the_native_spirv_four_word_result() -> None:
    ballot = SPIRV_SUBGROUP_BALLOT_INSTRUCTION
    assert ballot.result_type == SPIRV_SUBGROUP_V4U32_TYPE
    assert ballot.result_type.suffix == "v4u32"
    assert ballot.operand_types[0].suffix == "bool"
    assert ballot.feature_bits == feature_bits_value(("group_non_uniform_ballot",))


def test_ballot_bit_extract_uses_the_native_mask_and_lane_types() -> None:
    extract = SPIRV_SUBGROUP_BALLOT_BIT_EXTRACT_INSTRUCTION
    assert extract.result_type.suffix == "bool"
    assert extract.operand_names == ("mask", "index")
    assert extract.operand_types == (
        SPIRV_SUBGROUP_V4U32_TYPE,
        SPIRV_SUBGROUP_U32_TYPE,
    )
    assert extract.packet_form == ("LOOM_SPIRV_PACKET_FORM_GROUP_NON_UNIFORM_BINARY")
    assert extract.feature_bits == feature_bits_value(("group_non_uniform_ballot",))


def test_ballot_packing_preserves_the_low_64_lane_bits() -> None:
    extract, construct, bitcast = SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS
    assert extract.result_type.suffix == "u32"
    assert extract.operand_types == (SPIRV_SUBGROUP_V4U32_TYPE,)
    assert extract.component_index_maximum == 3
    assert construct.result_type == SPIRV_SUBGROUP_V2U32_TYPE
    assert construct.operand_types == (
        SPIRV_SUBGROUP_V2U32_TYPE.component_type,
        SPIRV_SUBGROUP_V2U32_TYPE.component_type,
    )
    assert bitcast.operand_types == (SPIRV_SUBGROUP_V2U32_TYPE,)
    assert bitcast.result_type == SPIRV_SUBGROUP_U64_TYPE
    assert isinstance(bitcast.operand_types[0], OrdinaryVectorType)
    assert (
        bitcast.operand_types[0].lane_count
        * bitcast.operand_types[0].component_type.bit_width
        == bitcast.result_type.bit_width
        == 64
    )


def test_ballot_native_types_remain_source_ineligible() -> None:
    assert not SPIRV_SUBGROUP_V4U32_TYPE.component_type.source_types
    assert not SPIRV_SUBGROUP_V2U32_TYPE.component_type.source_types
    assert not SPIRV_SUBGROUP_U64_TYPE.source_types


def test_boolean_votes_have_native_capability_and_scalar_result() -> None:
    assert {row.opcode for row in SPIRV_SUBGROUP_VOTE_INSTRUCTIONS} == {
        "LOOM_SPIRV_OP_GROUP_NON_UNIFORM_ANY",
        "LOOM_SPIRV_OP_GROUP_NON_UNIFORM_ALL",
    }
    for row in SPIRV_SUBGROUP_VOTE_INSTRUCTIONS:
        assert row.result_type.suffix == "bool"
        assert row.operand_types == (row.result_type,)
        assert row.feature_bits == feature_bits_value(("group_non_uniform_vote",))


def test_shuffle_rows_cover_every_native_type_shape_and_mode() -> None:
    actual = set()
    for row in SPIRV_SUBGROUP_SHUFFLES:
        instruction = row.instruction
        value_type = instruction.result_type
        component = (
            value_type.component_type
            if isinstance(value_type, OrdinaryVectorType)
            else value_type
        )
        lanes = (
            value_type.lane_count if isinstance(value_type, OrdinaryVectorType) else 1
        )
        cell = (component.suffix, lanes, row.mode)
        assert cell not in actual, cell
        actual.add(cell)
        assert instruction.operand_names == ("value", "offset")
        assert instruction.operand_types == (value_type, SPIRV_SUBGROUP_U32_TYPE)
        assert instruction.packet_form == (
            "LOOM_SPIRV_PACKET_FORM_GROUP_NON_UNIFORM_BINARY"
        )
        capability = (
            "group_non_uniform_shuffle_relative"
            if row.mode in ("up", "down")
            else "group_non_uniform_shuffle"
        )
        assert instruction.feature_bits & feature_bits_value((capability,))

    assert actual == {
        (component, lanes, mode)
        for component in (
            "bool",
            "i8",
            "i16",
            "i32",
            "i64",
            "offset64",
            "f16",
            "f32",
            "f64",
        )
        for lanes in (1, 2, 3, 4)
        for mode in ("index", "xor", "up", "down")
    }


def test_shuffle_extended_payloads_require_device_admission() -> None:
    extended_types = feature_bits_value(("subgroup_extended_types",))
    for row in SPIRV_SUBGROUP_SHUFFLES:
        value_type = row.instruction.result_type
        component = (
            value_type.component_type
            if isinstance(value_type, OrdinaryVectorType)
            else value_type
        )
        assert bool(row.instruction.feature_bits & extended_types) == (
            component.suffix in ("i8", "i16", "i64", "offset64", "f16")
        )


def test_bfloat_payloads_never_claim_native_group_instructions() -> None:
    assert all("bf16" not in row.instruction.key for row in SPIRV_SUBGROUP_SHUFFLES)
    assert all(
        "bf16" not in row.key for row in SPIRV_SUBGROUP_BROADCAST_FIRST_INSTRUCTIONS
    )


def test_first_active_broadcast_covers_native_payloads_with_ballot_capability() -> None:
    indexed = {
        row.instruction.result_type: row.instruction
        for row in SPIRV_SUBGROUP_SHUFFLES
        if row.mode == "index"
    }
    assert len(SPIRV_SUBGROUP_BROADCAST_FIRST_INSTRUCTIONS) == len(indexed)
    for row in SPIRV_SUBGROUP_BROADCAST_FIRST_INSTRUCTIONS:
        assert row.opcode == "LOOM_SPIRV_OP_GROUP_NON_UNIFORM_BROADCAST_FIRST"
        assert row.packet_form == "LOOM_SPIRV_PACKET_FORM_GROUP_NON_UNIFORM_UNARY"
        assert row.operand_names == ("value",)
        assert row.operand_types == (row.result_type,)
        assert row.feature_bits == (
            feature_bits_value(("group_non_uniform_ballot",))
            | (
                indexed[row.result_type].feature_bits
                & ~feature_bits_value(("group_non_uniform_shuffle",))
            )
        )


def test_arithmetic_rows_cover_reduction_and_scan_group_operations() -> None:
    operations = {row.instruction.group_operation for row in SPIRV_SUBGROUP_ARITHMETIC}
    assert operations == {
        "LOOM_SPIRV_GROUP_OPERATION_REDUCE",
        "LOOM_SPIRV_GROUP_OPERATION_CLUSTERED_REDUCE",
        "LOOM_SPIRV_GROUP_OPERATION_INCLUSIVE_SCAN",
        "LOOM_SPIRV_GROUP_OPERATION_EXCLUSIVE_SCAN",
    }
    assert any(
        row.combining_kinds == ("addf",)
        and row.instruction.group_operation
        == "LOOM_SPIRV_GROUP_OPERATION_EXCLUSIVE_SCAN"
        for row in SPIRV_SUBGROUP_ARITHMETIC
    )
    for row in SPIRV_SUBGROUP_ARITHMETIC:
        instruction = row.instruction
        assert instruction.feature_bits & feature_bits_value(
            (
                "group_non_uniform_clustered"
                if instruction.clustered
                else "group_non_uniform_arithmetic",
            )
        )
