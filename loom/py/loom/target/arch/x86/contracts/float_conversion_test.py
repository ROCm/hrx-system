# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.target.arch.x86.contracts.float_conversion import float_conversion_rules
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import GuardKind, Scalar, Vector, descriptor_by_key
from loom.target.low_descriptors import OperandAddressMapKind


@pytest.mark.parametrize("evex", [False, True])
def test_conversion_covers_each_supported_source_and_result_carrier(
    evex: bool,
) -> None:
    descriptor_set = X86_AVX512_CORE_DESCRIPTOR_SET if evex else X86_AVX2_DESCRIPTOR_SET
    rules = float_conversion_rules(
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
        assert rule.priority == int(evex)
    double_lanes = (2, 4, 8) if evex else (2, 4)
    assert cells == {
        (Scalar("f32"), Scalar("f64")),
        (Scalar("f64"), Scalar("f32")),
        *((Vector("f32", lanes=n), Vector("f64", lanes=n)) for n in double_lanes),
        *((Vector("f64", lanes=n), Vector("f32", lanes=n)) for n in double_lanes),
        *((Vector("bf16", lanes=n), Vector("f64", lanes=n)) for n in double_lanes),
        *(
            (Vector("bf16", lanes=n), Vector("f32", lanes=n))
            for n in ((16,) if evex else (2, 4, 8))
        ),
    }


def test_conversion_native_encodings_and_register_domains() -> None:
    expected = {
        "x86.avx2.vcvtss2sd.xmm": (0x8210, 0x095A),
        "x86.avx2.vcvtps2pd.xmm.xmm": (0x8140, 0x015A),
        "x86.avx2.vcvtps2pd.ymm.xmm": (0x8140, 0x415A),
        "x86.avx2.vcvtsd2ss.xmm": (0x8210, 0x0D5A),
        "x86.avx2.vcvtpd2ps.xmm.xmm": (0x8140, 0x055A),
        "x86.avx2.vcvtpd2ps.xmm.ymm": (0x8140, 0x455A),
        "x86.avx512.vcvtss2sd.xmm": (0x8210, 0x295A),
        "x86.avx512.vcvtps2pd.xmm.xmm": (0x8140, 0x215A),
        "x86.avx512.vcvtps2pd.ymm.xmm": (0x8140, 0x615A),
        "x86.avx512.vcvtps2pd.zmm.ymm": (0x8140, 0xA15A),
        "x86.avx512.vcvtsd2ss.xmm": (0x8210, 0x3D5A),
        "x86.avx512.vcvtpd2ps.xmm.xmm": (0x8140, 0x355A),
        "x86.avx512.vcvtpd2ps.xmm.ymm": (0x8140, 0x755A),
        "x86.avx512.vcvtpd2ps.ymm.zmm": (0x8140, 0xB55A),
    }
    for key, (recipe, encoding) in expected.items():
        descriptor = descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)
        assert (descriptor.encoding_format_id, descriptor.encoding_id) == (
            recipe,
            encoding,
        )
        for operand in descriptor.operands:
            if key.startswith("x86.avx2."):
                assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
                assert operand.addressable_unit_count == 16
            else:
                assert operand.address_map_kind is not OperandAddressMapKind.LOW_SUBSET
