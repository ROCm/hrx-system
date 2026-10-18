# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.target.arch.x86.contracts.half_conversion import half_conversion_rules
from loom.target.arch.x86.descriptors import (
    X86_AVX2_FEATURES_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VL, FEATURE_F16C
from loom.target.contracts import GuardKind, Scalar, Vector, descriptor_by_key
from loom.target.low_descriptors import OperandAddressMapKind


@pytest.mark.parametrize("evex", [False, True])
def test_half_conversion_covers_scalar_and_each_native_packet(evex: bool) -> None:
    descriptor_set = (
        X86_AVX512_CORE_DESCRIPTOR_SET if evex else X86_AVX2_FEATURES_DESCRIPTOR_SET
    )
    rules = half_conversion_rules(
        lambda key: descriptor_by_key(descriptor_set, key), evex=evex
    )
    cells = set()
    for rule in rules:
        types = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        cells.add((types["input"], types["result"]))
        for emit in rule.emit:
            if emit.descriptor.mnemonic == "vcvtps2ph":
                assert emit.immediates == {"rounding": 0}
    lanes = (2, 4, 8, 16) if evex else (2, 4, 8)
    double_lanes = (2, 4, 8) if evex else (2, 4)
    assert cells == {
        (Scalar("f16"), Scalar("f32")),
        (Scalar("f32"), Scalar("f16")),
        *((Vector("f16", lanes=n), Vector("f32", lanes=n)) for n in lanes),
        *((Vector("f32", lanes=n), Vector("f16", lanes=n)) for n in lanes),
        *((Vector("f16", lanes=n), Vector("f64", lanes=n)) for n in double_lanes),
    }


@pytest.mark.parametrize("evex", [False, True])
def test_legacy_half_encodings_and_feature_requirements(evex: bool) -> None:
    descriptor_set = (
        X86_AVX512_CORE_DESCRIPTOR_SET if evex else X86_AVX2_FEATURES_DESCRIPTOR_SET
    )
    feature = "avx512" if evex else "f16c"
    suffixes = (("xmm", "xmm", 0), ("ymm", "xmm", 0x4000))
    if evex:
        suffixes += (("zmm", "ymm", 0x8000),)
    for wide, half, width_bits in suffixes:
        for narrow in (False, True):
            mnemonic = "vcvtps2ph" if narrow else "vcvtph2ps"
            suffix = f"{half}.{wide}" if narrow else f"{wide}.{half}"
            descriptor = descriptor_by_key(
                descriptor_set, f"x86.{feature}.{mnemonic}.{suffix}"
            )
            assert descriptor.encoding_format_id == (0x9041 if narrow else 0x8140)
            assert descriptor.encoding_id == (
                (0x071D if narrow else 0x0613) | width_bits | (0x2000 if evex else 0)
            )
            assert descriptor.feature_mask_words == (
                (FEATURE_AVX512_VL,)
                if evex and wide != "zmm"
                else (() if evex else (FEATURE_F16C,))
            )
            for operand in descriptor.operands:
                if evex:
                    assert (
                        operand.address_map_kind is not OperandAddressMapKind.LOW_SUBSET
                    )
                else:
                    assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
                    assert operand.addressable_unit_count == 16
