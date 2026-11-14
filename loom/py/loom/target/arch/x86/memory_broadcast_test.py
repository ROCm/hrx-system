# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VL
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingBehavior,
    VectorRegisterSelector,
    validate_vector_encoding_recipe,
    vector_encoding_recipe,
)
from loom.target.low_descriptors import (
    DescriptorFlag,
    EffectKind,
    OperandAddressMapKind,
)


def test_memory_broadcasts_cover_every_storage_width_and_native_carrier() -> None:
    for descriptor_set, profile, widths, evex in (
        (X86_AVX2_DESCRIPTOR_SET, "avx2", (128, 256), False),
        (X86_AVX512_CORE_DESCRIPTOR_SET, "avx512", (128, 256, 512), True),
    ):
        descriptors = {row.key: row for row in descriptor_set.descriptors}
        expected_keys = set()
        for bits, mnemonic, opcode, recipe in (
            (8, "vpbroadcastb", 0x78, 0xC140),
            (16, "vpbroadcastw", 0x79, 0xC1D0),
            (32, "vpbroadcastd", 0x58, 0xC1E0),
            (64, "vpbroadcastq", 0x59, 0xC1F0),
        ):
            for width in widths:
                suffix = {128: "xmm", 256: "ymm", 512: "zmm"}[width]
                for indexed in (False, True):
                    address = ".indexed" if indexed else ""
                    key = f"x86.{profile}.{mnemonic}.load{address}.{suffix}"
                    expected_keys.add(key)
                    row = descriptors[key]
                    assert row.encoding_format_id == recipe
                    assert row.encoding_id == (
                        opcode
                        | 0x600
                        | (0x2000 if evex else 0)
                        | (0x1000 if evex and bits == 64 else 0)
                        | ({128: 0, 256: 1, 512: 2}[width] << 14)
                    )
                    assert tuple(operand.field_name for operand in row.operands) == (
                        ("dst", "base", "index") if indexed else ("dst", "base")
                    )
                    result = row.operands[0]
                    assert tuple(alt.reg_class for alt in result.reg_alts) == (
                        f"x86.{suffix}",
                    )
                    assert result.addressable_unit_count == (0 if evex else 16)
                    assert result.address_map_kind == (
                        OperandAddressMapKind.DIRECT
                        if evex
                        else OperandAddressMapKind.LOW_SUBSET
                    )
                    assert row.feature_mask_words == (
                        (FEATURE_AVX512_VL,) if evex and width < 512 else ()
                    )
                    assert len(row.effects) == 1
                    assert row.effects[0].kind is EffectKind.READ
                    assert row.effects[0].width_bits == bits
                    assert DescriptorFlag.SIDE_EFFECTING in row.flags
        actual_keys = {
            key
            for key in descriptors
            if key.startswith(f"x86.{profile}.vpbroadcast") and ".load" in key
        }
        assert actual_keys == expected_keys


def test_scalar_memory_tuple_recipes_preserve_unused_vvvv() -> None:
    selector = VectorRegisterSelector
    for bits, expected in ((8, 0xC140), (16, 0xC1D0), (32, 0xC1E0), (64, 0xC1F0)):
        recipe = vector_encoding_recipe(
            selector.RESULT,
            selector.NONE,
            selector.INPUT_0,
            VectorEncodingBehavior.LOAD,
            scalar_memory_bit_width=bits,
        )
        assert recipe == expected
        validate_vector_encoding_recipe(recipe)


def test_scalar_memory_tuple_rejects_conflicting_native_facts() -> None:
    selector = VectorRegisterSelector
    for keywords in (
        {"scalar_memory_bit_width": 128},
        {"scalar_memory_bit_width": 32, "full_vector_tuple": True},
        {"scalar_memory_bit_width": 32, "broadcast32": True},
    ):
        try:
            vector_encoding_recipe(
                selector.RESULT,
                selector.NONE,
                selector.INPUT_0,
                VectorEncodingBehavior.LOAD,
                **keywords,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting memory tuple was accepted")
    try:
        vector_encoding_recipe(
            selector.RESULT,
            selector.INPUT_1,
            selector.INPUT_0,
            VectorEncodingBehavior.LOAD,
            scalar_memory_bit_width=32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("scalar tuple discarded a real vvvv operand")
