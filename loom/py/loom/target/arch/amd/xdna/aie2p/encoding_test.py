# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amd.xdna.aie.encoding import (
    BitMapping,
    BundleFieldEncoding,
    BundleFormatEncoding,
    EncodingTable,
    InstructionInstance,
    decode_instruction_fields,
    encode_instruction,
    encode_witness,
    gather_bits,
    instruction_fixed_pattern_overlaps,
    scatter_bits,
    validate_encoding_table,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import (
    CORE_ENCODING_TABLE,
    CORE_ENCODING_WITNESSES,
    SLOT_BIT_COUNTS,
)


def test_core_table_is_structurally_complete() -> None:
    validate_encoding_table(CORE_ENCODING_TABLE, SLOT_BIT_COUNTS)

    assert len(CORE_ENCODING_TABLE.instructions) == 880
    assert len(CORE_ENCODING_TABLE.bundle_formats) == 77
    assert (
        len(
            {
                tuple(sorted(field.slot for field in bundle_format.fields))
                for bundle_format in CORE_ENCODING_TABLE.bundle_formats
            }
        )
        == 77
    )
    assert (
        len(
            {
                field.name
                for instruction in CORE_ENCODING_TABLE.instructions
                for field in instruction.fields
            }
        )
        == 28
    )
    assert (
        sum(
            len(field.mappings)
            for instruction in CORE_ENCODING_TABLE.instructions
            for field in instruction.fields
        )
        + sum(
            len(field.mappings)
            for bundle_format in CORE_ENCODING_TABLE.bundle_formats
            for field in bundle_format.fields
        )
        == 14_856
    )

    overlaps = instruction_fixed_pattern_overlaps(CORE_ENCODING_TABLE)
    assert len(overlaps) == 97


def test_all_instruction_fields_round_trip() -> None:
    for instruction in CORE_ENCODING_TABLE.instructions:
        for use_maximum_values in (False, True):
            field_values = tuple(
                (
                    field.name,
                    field.value_mask if use_maximum_values else 0,
                )
                for field in instruction.fields
            )
            _, encoded_value = encode_instruction(
                CORE_ENCODING_TABLE,
                InstructionInstance(instruction.name, field_values),
            )
            assert decode_instruction_fields(instruction, encoded_value) == dict(
                field_values
            )


def test_all_bundle_slot_mappings_round_trip() -> None:
    for bundle_format in CORE_ENCODING_TABLE.bundle_formats:
        for use_maximum_values in (False, True):
            encoded_value = bundle_format.fixed_value
            expected_values = {}
            for field in bundle_format.fields:
                field_value = field.value_mask if use_maximum_values else 0
                expected_values[field.slot] = field_value
                encoded_value = scatter_bits(
                    encoded_value,
                    field.mappings,
                    field_value,
                )
            assert (
                encoded_value & bundle_format.fixed_mask
            ) == bundle_format.fixed_value
            assert {
                field.slot: gather_bits(encoded_value, field.mappings)
                for field in bundle_format.fields
            } == expected_values


def test_bundle_slot_mappings_must_be_invertible() -> None:
    table = EncodingTable(
        instructions=(),
        bundle_formats=(
            BundleFormatEncoding(
                name="repeated_slot_bit",
                bit_count=8,
                fixed_mask=0,
                fixed_value=0,
                fields=(
                    BundleFieldEncoding(
                        "SLOT",
                        (
                            BitMapping(0, 0),
                            BitMapping(1, 0),
                            *(
                                BitMapping(target_bit, target_bit - 1)
                                for target_bit in range(2, 8)
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="repeated value bits"):
        validate_encoding_table(table, {"SLOT": 7})


def test_bundle_prefixes_must_be_unambiguous() -> None:
    table = EncodingTable(
        instructions=(),
        bundle_formats=(
            BundleFormatEncoding(
                name="short",
                bit_count=8,
                fixed_mask=0x0F,
                fixed_value=0x0A,
                fields=(
                    BundleFieldEncoding(
                        "SHORT_SLOT",
                        tuple(BitMapping(bit + 4, bit) for bit in range(4)),
                    ),
                ),
            ),
            BundleFormatEncoding(
                name="long",
                bit_count=16,
                fixed_mask=0xF00F,
                fixed_value=0xA00A,
                fields=(
                    BundleFieldEncoding(
                        "LONG_SLOT",
                        tuple(BitMapping(bit + 4, bit) for bit in range(8)),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="bundle prefix decode is ambiguous"):
        validate_encoding_table(table, {"SHORT_SLOT": 4, "LONG_SLOT": 8})


def test_core_table_reproduces_retained_vector_leaves() -> None:
    for witness in CORE_ENCODING_WITNESSES:
        assert encode_witness(CORE_ENCODING_TABLE, witness) == witness.expected_bytes
