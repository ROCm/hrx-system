# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.descriptors import (
    X86_AVX2_FEATURES_DESCRIPTOR_SET,
    X86_AVX512_FEATURES_DESCRIPTOR_SET,
    X86_AVX_NE_CONVERT_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX_NE_CONVERT
from loom.target.low_descriptors import OperandAddressMapKind


def test_avx_ne_convert_rows_cover_the_complete_public_family() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in X86_AVX_NE_CONVERT_DESCRIPTOR_SET.descriptors
    }
    expected_memory = {
        "vbcstnebf162ps": (0x0AB1, 16),
        "vbcstnesh2ps": (0x06B1, 16),
        "vcvtneebf162ps": (0x0AB0, None),
        "vcvtneeph2ps": (0x06B0, None),
        "vcvtneobf162ps": (0x0EB0, None),
        "vcvtneoph2ps": (0x02B0, None),
    }
    expected_keys = set()
    for mnemonic, (xmm_encoding_id, fixed_memory_width) in expected_memory.items():
        for register_suffix, result_width, vector_length_bit in (
            ("xmm", 128, 0),
            ("ymm", 256, 0x4000),
        ):
            for operation, encoding_format_id in (
                ("load", 0xC1C0),
                ("load.indexed", 0xC1C0),
            ):
                key = f"x86.avx_ne_convert.{mnemonic}.{operation}.{register_suffix}"
                expected_keys.add(key)
                descriptor = descriptors[key]
                assert descriptor.encoding_format_id == encoding_format_id
                assert descriptor.encoding_id == xmm_encoding_id | vector_length_bit
                assert descriptor.feature_mask_words == (FEATURE_AVX_NE_CONVERT,)
                assert descriptor.effects[0].width_bits == (
                    fixed_memory_width or result_width
                )

    expected_narrowing = {
        "x86.avx_ne_convert.vcvtneps2bf16.xmm.xmm": 0x0A72,
        "x86.avx_ne_convert.vcvtneps2bf16.xmm.ymm": 0x4A72,
    }
    expected_keys.update(expected_narrowing)
    for key, encoding_id in expected_narrowing.items():
        descriptor = descriptors[key]
        assert descriptor.encoding_format_id == 0x8140
        assert descriptor.encoding_id == encoding_id
        assert descriptor.feature_mask_words == (FEATURE_AVX_NE_CONVERT,)

    assert set(descriptors) == expected_keys


def test_avx_ne_convert_rows_stay_in_low_simd_registers() -> None:
    for descriptor in X86_AVX_NE_CONVERT_DESCRIPTOR_SET.descriptors:
        simd_operands = tuple(
            operand
            for operand in descriptor.operands
            if any(
                alternative.reg_class in ("x86.xmm", "x86.ymm")
                for alternative in operand.reg_alts
            )
        )
        assert simd_operands
        for operand in simd_operands:
            assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
            assert operand.addressable_unit_count == 16


def test_avx_ne_convert_rows_share_both_dynamic_feature_sets() -> None:
    expected_keys = {
        descriptor.key for descriptor in X86_AVX_NE_CONVERT_DESCRIPTOR_SET.descriptors
    }
    for descriptor_set in (
        X86_AVX2_FEATURES_DESCRIPTOR_SET,
        X86_AVX512_FEATURES_DESCRIPTOR_SET,
    ):
        actual_keys = {descriptor.key for descriptor in descriptor_set.descriptors}
        assert expected_keys <= actual_keys
