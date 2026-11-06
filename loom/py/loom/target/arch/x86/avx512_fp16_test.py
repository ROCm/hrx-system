# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.descriptors import X86_AVX512_FP16_DESCRIPTOR_SET
from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX512_FP16,
    FEATURE_AVX512_VL,
)
from loom.target.arch.x86.vector_families import (
    AVX512_FP16_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES,
)


def test_avx512_fp16_rows_cover_the_native_scalar_and_vector_family() -> None:
    expected: dict[str, tuple[int, int]] = {}

    def add(key: str, encoding_id: int, feature_bits: int) -> None:
        expected[f"x86.avx512_fp16.{key}"] = (encoding_id, feature_bits)

    packed_opcodes = {
        "vaddph": 0x58,
        "vsubph": 0x5C,
        "vmulph": 0x59,
        "vdivph": 0x5E,
        "vminph": 0x5D,
        "vmaxph": 0x5F,
    }
    scalar_opcodes = {
        "vaddsh": 0x58,
        "vsubsh": 0x5C,
        "vmulsh": 0x59,
        "vdivsh": 0x5E,
        "vminsh": 0x5D,
        "vmaxsh": 0x5F,
    }
    for mnemonic, opcode in packed_opcodes.items():
        for suffix, length_bits in (("xmm", 0), ("ymm", 0x4000), ("zmm", 0x8000)):
            feature_bits = FEATURE_AVX512_FP16
            if suffix != "zmm":
                feature_bits |= FEATURE_AVX512_VL
            add(f"{mnemonic}.{suffix}", opcode | length_bits, feature_bits)
    for mnemonic, opcode in scalar_opcodes.items():
        add(f"{mnemonic}.xmm", opcode | 0x0800, FEATURE_AVX512_FP16)

    for mnemonic, opcode in (("vsqrtph", 0x51), ("vrsqrtph", 0x244E)):
        for suffix, length_bits in (("xmm", 0), ("ymm", 0x4000), ("zmm", 0x8000)):
            feature_bits = FEATURE_AVX512_FP16
            if suffix != "zmm":
                feature_bits |= FEATURE_AVX512_VL
            add(f"{mnemonic}.{suffix}", opcode | length_bits, feature_bits)
    add("vsqrtsh.xmm", 0x0851, FEATURE_AVX512_FP16)
    add("vrsqrtsh.xmm", 0x244F, FEATURE_AVX512_FP16)
    for suffix, length_bits in (("xmm", 0), ("ymm", 0x4000), ("zmm", 0x8000)):
        feature_bits = FEATURE_AVX512_FP16
        if suffix != "zmm":
            feature_bits |= FEATURE_AVX512_VL
        add(f"vrndscaleph.{suffix}", 0x2308 | length_bits, feature_bits)
    add("vrndscalesh.xmm", 0x230A, FEATURE_AVX512_FP16)

    for suffix, length_bits in (("xmm", 0), ("ymm", 0x4000), ("zmm", 0x8000)):
        feature_bits = FEATURE_AVX512_FP16
        if suffix != "zmm":
            feature_bits |= FEATURE_AVX512_VL
        add(f"vfmadd231ph.{suffix}", 0x24B8 | length_bits, feature_bits)
        add(f"vcmpph.{suffix}", 0x23C2 | length_bits, feature_bits)
    add("vfmadd231sh.xmm", 0x24B9, FEATURE_AVX512_FP16)
    add("vcmpsh.xmm", 0x2BC2, FEATURE_AVX512_FP16)

    add("vcvtph2psx.xmm.xmm", 0x2413, FEATURE_AVX512_FP16 | FEATURE_AVX512_VL)
    add("vcvtph2psx.ymm.xmm", 0x6413, FEATURE_AVX512_FP16 | FEATURE_AVX512_VL)
    add("vcvtph2psx.zmm.ymm", 0xA413, FEATURE_AVX512_FP16)
    add("vcvtps2phx.xmm.xmm", 0x041D, FEATURE_AVX512_FP16 | FEATURE_AVX512_VL)
    add("vcvtps2phx.xmm.ymm", 0x441D, FEATURE_AVX512_FP16 | FEATURE_AVX512_VL)
    add("vcvtps2phx.ymm.zmm", 0x841D, FEATURE_AVX512_FP16)
    add("vcvtsh2ss.xmm", 0x2013, FEATURE_AVX512_FP16)
    add("vcvtss2sh.xmm", 0x001D, FEATURE_AVX512_FP16)
    add("vpbroadcastw.zmm.xmm", 0xA679, FEATURE_AVX512_FP16)

    descriptors = {
        descriptor.key: descriptor
        for descriptor in X86_AVX512_FP16_DESCRIPTOR_SET.descriptors
    }
    assert tuple(
        reg_class.name for reg_class in X86_AVX512_FP16_DESCRIPTOR_SET.reg_classes
    ) == ("x86.xmm", "x86.ymm", "x86.zmm", "x86.k")
    assert set(descriptors) == set(expected)
    actual = {
        key: (descriptor.encoding_id, descriptor.feature_mask_words[0])
        for key, descriptor in descriptors.items()
    }
    assert actual == expected, (
        f"unexpected={sorted(actual.items() - expected.items())}, "
        f"missing={sorted(expected.items() - actual.items())}"
    )
    assert all(descriptor.encoding_format_id for descriptor in descriptors.values())


def test_avx512_fp16_arithmetic_families_share_source_operations() -> None:
    expected_operations = ("addf", "subf", "mulf", "divf")
    assert (
        tuple(family.source_operation for family in AVX512_FP16_FLOAT_BINARY_FAMILIES)
        == expected_operations
    )
    assert (
        tuple(
            family.source_operation
            for family in AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES
        )
        == expected_operations
    )
