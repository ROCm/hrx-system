# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor invariants consumed by the native Wasm immediate readers."""

from loom.target.arch.wasm.descriptors import (
    WASM_CORE_SIMD128_DESCRIPTOR_SET,
    WASM_INTEGER_ARITHMETIC_INSTRUCTIONS,
)
from loom.target.low_descriptors import ImmediateFlag, ImmediateKind


def test_single_immediates_are_numeric_with_only_a_zero_offset_default():
    for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors:
        if descriptor.key in ("wasm.v128.const", "wasm.i8x16.shuffle"):
            continue
        assert len(descriptor.immediates) <= 1
        for immediate in descriptor.immediates:
            if descriptor.key in ("wasm.br", "wasm.br_if.i32"):
                # Native control flow uses structural Low operations; symbolic
                # branch packets are rejected before immediate emission.
                assert immediate.kind is ImmediateKind.ORDINAL
                assert immediate.flags == (ImmediateFlag.SYMBOLIC,)
                continue
            assert immediate.kind in (ImmediateKind.SIGNED, ImmediateKind.UNSIGNED)
            if ImmediateFlag.DEFAULT_VALUE in immediate.flags:
                assert immediate.field_name == "offset"
                assert immediate.kind is ImmediateKind.UNSIGNED
                assert immediate.bit_width == 32
                assert immediate.unsigned_max == (1 << 32) - 1
                assert immediate.default_value == 0
            else:
                assert not immediate.flags


def test_vector_constant_words_are_required_full_width_bits():
    descriptor = next(
        descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
        if descriptor.key == "wasm.v128.const"
    )
    assert sorted(value.field_name for value in descriptor.immediates) == [
        "hi64",
        "lo64",
    ]
    assert [value.field_name for value in descriptor.asm_forms[0].immediates] == [
        "lo64",
        "hi64",
    ]
    for immediate in descriptor.immediates:
        assert immediate.kind is ImmediateKind.UNSIGNED
        assert immediate.bit_width == 64
        assert immediate.unsigned_max == (1 << 64) - 1
        assert not immediate.flags


def test_shuffle_has_sixteen_required_lanes_from_both_inputs():
    descriptor = next(
        descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
        if descriptor.key == "wasm.i8x16.shuffle"
    )
    wire_names = [f"lane{lane}" for lane in range(16)]
    assert sorted(value.field_name for value in descriptor.immediates) == sorted(
        wire_names
    )
    assert [
        value.field_name for value in descriptor.asm_forms[0].immediates
    ] == wire_names
    for immediate in descriptor.immediates:
        assert immediate.kind is ImmediateKind.UNSIGNED
        assert immediate.unsigned_max == 31
        assert not immediate.flags


def test_narrow_lanes_use_unsigned_extract_encodings_and_exact_lane_domains():
    descriptors = {
        descriptor.key: descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
    }
    for shape, lanes, splat_opcode, extract_opcode, replace_opcode in (
        ("i8x16", 16, 0xFD0F, 0xFD16, 0xFD17),
        ("i16x8", 8, 0xFD10, 0xFD19, 0xFD1A),
    ):
        assert descriptors[f"wasm.{shape}.splat"].encoding_id == splat_opcode
        for operation, opcode in (
            ("extract_lane_u", extract_opcode),
            ("replace_lane", replace_opcode),
        ):
            descriptor = descriptors[f"wasm.{shape}.{operation}"]
            assert descriptor.encoding_id == opcode
            (immediate,) = descriptor.immediates
            assert immediate.field_name == "lane"
            assert immediate.kind is ImmediateKind.UNSIGNED
            assert immediate.unsigned_max == lanes - 1
            assert not immediate.flags


def test_simd_compare_descriptors_cover_the_native_instruction_matrix():
    descriptors = {
        descriptor.key: descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
    }
    expected = {
        **{
            f"wasm.{shape}.{operation}": (0xFD << 8) | (base + index)
            for shape, base in (("i8x16", 0x23), ("i16x8", 0x2D), ("i32x4", 0x37))
            for index, operation in enumerate(
                (
                    "eq",
                    "ne",
                    "lt_s",
                    "lt_u",
                    "gt_s",
                    "gt_u",
                    "le_s",
                    "le_u",
                    "ge_s",
                    "ge_u",
                )
            )
        },
        **{
            f"wasm.i64x2.{operation}": (0xFD << 8) | opcode
            for operation, opcode in (
                ("eq", 0xD6),
                ("ne", 0xD7),
                ("lt_s", 0xD8),
                ("gt_s", 0xD9),
                ("le_s", 0xDA),
                ("ge_s", 0xDB),
            )
        },
        **{
            f"wasm.{shape}.{operation}": (0xFD << 8) | (base + index)
            for shape, base in (("f32x4", 0x41), ("f64x2", 0x47))
            for index, operation in enumerate(("eq", "ne", "lt", "gt", "le", "ge"))
        },
    }
    actual = {
        key: descriptor.encoding_id
        for key, descriptor in descriptors.items()
        if key.startswith(
            (
                "wasm.i8x16.",
                "wasm.i16x8.",
                "wasm.i32x4.",
                "wasm.i64x2.",
                "wasm.f32x4.",
                "wasm.f64x2.",
            )
        )
        and descriptor.semantic_tag.startswith("vector.cmp.")
    }
    assert actual == expected
    for key in expected:
        descriptor = descriptors[key]
        assert not descriptor.immediates
        assert len(descriptor.operands) == 3


def test_v128_not_uses_the_simd128_logical_opcode():
    descriptor = next(
        descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
        if descriptor.key == "wasm.v128.not"
    )
    assert descriptor.encoding_id == 0xFD4D
    assert [operand.field_name for operand in descriptor.operands] == ["dst", "input"]


def test_integer_arithmetic_descriptors_match_the_simd128_instruction_matrix():
    expected = {
        ("i8x16", "abs"): (0x60, 1),
        ("i8x16", "neg"): (0x61, 1),
        ("i8x16", "add"): (0x6E, 2),
        ("i8x16", "sub"): (0x71, 2),
        ("i8x16", "min_s"): (0x76, 2),
        ("i8x16", "min_u"): (0x77, 2),
        ("i8x16", "max_s"): (0x78, 2),
        ("i8x16", "max_u"): (0x79, 2),
        ("i16x8", "abs"): (0x80, 1),
        ("i16x8", "neg"): (0x81, 1),
        ("i16x8", "add"): (0x8E, 2),
        ("i16x8", "sub"): (0x91, 2),
        ("i16x8", "mul"): (0x95, 2),
        ("i16x8", "min_s"): (0x96, 2),
        ("i16x8", "min_u"): (0x97, 2),
        ("i16x8", "max_s"): (0x98, 2),
        ("i16x8", "max_u"): (0x99, 2),
        ("i32x4", "abs"): (0xA0, 1),
        ("i32x4", "neg"): (0xA1, 1),
        ("i32x4", "add"): (0xAE, 2),
        ("i32x4", "sub"): (0xB1, 2),
        ("i32x4", "mul"): (0xB5, 2),
        ("i32x4", "min_s"): (0xB6, 2),
        ("i32x4", "min_u"): (0xB7, 2),
        ("i32x4", "max_s"): (0xB8, 2),
        ("i32x4", "max_u"): (0xB9, 2),
        ("i64x2", "abs"): (0xC0, 1),
        ("i64x2", "neg"): (0xC1, 1),
        ("i64x2", "add"): (0xCE, 2),
        ("i64x2", "sub"): (0xD1, 2),
        ("i64x2", "mul"): (0xD5, 2),
    }
    actual = {
        (instruction.shape, instruction.operation): (
            instruction.subopcode,
            instruction.arity,
        )
        for instruction in WASM_INTEGER_ARITHMETIC_INSTRUCTIONS
    }
    assert actual == expected

    descriptors = {
        descriptor.key: descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
    }
    for (shape, operation), (subopcode, arity) in expected.items():
        descriptor = descriptors[f"wasm.{shape}.{operation}"]
        assert descriptor.encoding_id == 0xFD00 | subopcode
        assert not descriptor.immediates
        assert len(descriptor.operands) == arity + 1


def test_integer_arithmetic_recipe_descriptors_match_simd128_encodings():
    descriptors = {
        descriptor.key: descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
    }
    for key, opcode in (
        ("wasm.i16x8.extmul_low_i8x16_u", 0xFD9E),
        ("wasm.i16x8.extmul_high_i8x16_u", 0xFD9F),
        ("wasm.i64x2.lt_s", 0xFDD8),
    ):
        descriptor = descriptors[key]
        assert descriptor.encoding_id == opcode
        assert not descriptor.immediates
        assert [operand.field_name for operand in descriptor.operands] == [
            "dst",
            "lhs",
            "rhs",
        ]


def test_simd_shifts_have_one_i32_count_for_every_integer_lane_width():
    descriptors = {
        descriptor.key: descriptor
        for descriptor in WASM_CORE_SIMD128_DESCRIPTOR_SET.descriptors
    }
    for shape, encodings in (
        ("i8x16", (0xFD6B, 0xFD6C, 0xFD6D)),
        ("i16x8", (0xFD8B, 0xFD8C, 0xFD8D)),
        ("i32x4", (0xFDAB, 0xFDAC, 0xFDAD)),
        ("i64x2", (0xFDCB, 0xFDCC, 0xFDCD)),
    ):
        for operation, encoding in zip(
            ("shl", "shr_s", "shr_u"), encodings, strict=True
        ):
            descriptor = descriptors[f"wasm.{shape}.{operation}"]
            assert descriptor.encoding_id == encoding
            assert not descriptor.immediates
            assert [operand.field_name for operand in descriptor.operands] == [
                "dst",
                "value",
                "count",
            ]
            assert [
                alternative.reg_class
                for operand in descriptor.operands
                for alternative in operand.reg_alts
            ] == ["wasm.v128", "wasm.v128", "wasm.i32"]
