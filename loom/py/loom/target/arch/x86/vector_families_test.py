# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT8_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_BINARY_FAMILIES,
    AVX2_FLOAT_COMPARE_MNEMONICS,
    AVX2_FLOAT_FMA_MNEMONICS,
    AVX2_INTEGER_BINARY_FAMILIES,
    AVX2_INTEGER_COMPARE_MNEMONICS,
    AVX2_VECTOR_BIT_WIDTHS,
    AVX512_BITWISE_FAMILIES,
    AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
    AVX512_FLOAT_BINARY_FAMILIES,
    AVX512_FLOAT_COMPARE_MNEMONICS,
    AVX512_FLOAT_FMA_MNEMONICS,
    AVX512_INTEGER_BINARY_FAMILIES,
    AVX512_INTEGER_COMPARE_MNEMONICS,
    AVX512_INTEGER_REDUCTION_FAMILIES,
    AVX512_SELECT_MNEMONICS,
    AVX512_VECTOR_BIT_WIDTHS,
    AVX512VL_INTEGER_BINARY_FAMILIES,
    AVX512VL_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    INTEGER_ELEMENTS,
)


def test_avx2_direct_integer_matrix_matches_isa_families() -> None:
    cells = {
        (family.source_operation, family.element.name)
        for family in AVX2_INTEGER_BINARY_FAMILIES
    }
    assert cells == {
        *(
            (operation, element)
            for operation in ("addi", "subi")
            for element in ("i8", "i16", "i32", "i64")
        ),
        *(
            (operation, element)
            for operation in ("minsi", "maxsi", "minui", "maxui")
            for element in ("i8", "i16", "i32")
        ),
        ("muli", "i16"),
        ("muli", "i32"),
        ("shli", "i32"),
        ("shli", "i64"),
        ("shrsi", "i32"),
        ("shrui", "i32"),
        ("shrui", "i64"),
    }


def test_avx2_float_and_compare_matrices_cover_every_native_element() -> None:
    assert {
        (family.source_operation, family.element.name)
        for family in AVX2_FLOAT_BINARY_FAMILIES
    } == {
        (operation, element)
        for operation in ("addf", "subf", "mulf", "divf")
        for element in ("f32", "f64")
    }
    assert set(AVX2_INTEGER_COMPARE_MNEMONICS) == {
        element.name for element in INTEGER_ELEMENTS
    }
    assert set(AVX2_FLOAT_COMPARE_MNEMONICS) == {
        element.name for element in FLOAT_ELEMENTS
    }
    assert set(AVX2_FLOAT_FMA_MNEMONICS) == {element.name for element in FLOAT_ELEMENTS}


def test_avx2_family_rows_materialize_both_register_widths() -> None:
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    suffixes = {128: "xmm", 256: "ymm"}
    family_mnemonics = {
        family.mnemonic
        for family in (*AVX2_INTEGER_BINARY_FAMILIES, *AVX2_FLOAT_BINARY_FAMILIES)
    }
    family_mnemonics.update(
        mnemonic
        for pair in AVX2_INTEGER_COMPARE_MNEMONICS.values()
        for mnemonic in pair
    )
    family_mnemonics.update(AVX2_FLOAT_COMPARE_MNEMONICS.values())
    family_mnemonics.update(AVX2_FLOAT_FMA_MNEMONICS.values())
    assert {
        f"x86.avx2.{mnemonic}.{suffixes[vector_bit_width]}"
        for mnemonic in family_mnemonics
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    } <= descriptor_keys


def test_avx_vnni_int8_rows_have_complete_direct_encodings() -> None:
    descriptors = {
        descriptor.key: (descriptor.encoding_format_id, descriptor.encoding_id)
        for descriptor in X86_AVX_VNNI_INT8_DESCRIPTOR_SET.descriptors
    }
    expected_encoding_ids = {
        "x86.avx_vnni_int8.vpdpbssd.xmm": 0x0E50,
        "x86.avx_vnni_int8.vpdpbssd.ymm": 0x4E50,
        "x86.avx_vnni_int8.vpdpbssds.xmm": 0x0E51,
        "x86.avx_vnni_int8.vpdpbssds.ymm": 0x4E51,
        "x86.avx_vnni_int8.vpdpbsud.xmm": 0x0A50,
        "x86.avx_vnni_int8.vpdpbsud.ymm": 0x4A50,
        "x86.avx_vnni_int8.vpdpbsuds.xmm": 0x0A51,
        "x86.avx_vnni_int8.vpdpbsuds.ymm": 0x4A51,
        "x86.avx_vnni_int8.vpdpbuud.xmm": 0x0250,
        "x86.avx_vnni_int8.vpdpbuud.ymm": 0x4250,
        "x86.avx_vnni_int8.vpdpbuuds.xmm": 0x0251,
        "x86.avx_vnni_int8.vpdpbuuds.ymm": 0x4251,
    }
    assert descriptors == {
        key: (0x8320, encoding_id) for key, encoding_id in expected_encoding_ids.items()
    }


def test_avx512_direct_integer_matrix_matches_core_isa_families() -> None:
    assert {
        (family.source_operation, family.element.name)
        for family in AVX512_INTEGER_BINARY_FAMILIES
    } == {
        *(
            (operation, element)
            for operation in ("addi", "subi")
            for element in ("i8", "i16", "i32", "i64")
        ),
        *(
            (operation, element)
            for operation in ("minsi", "maxsi", "minui", "maxui")
            for element in ("i8", "i16", "i32", "i64")
        ),
        *(("muli", element) for element in ("i16", "i32", "i64")),
        *(
            (operation, element)
            for operation in ("shli", "shrsi", "shrui")
            for element in ("i16", "i32", "i64")
        ),
    }


def test_avx512_integer_reduction_matrix_matches_core_isa_families() -> None:
    assert {
        (family.source_operation, family.element.name)
        for family in AVX512_INTEGER_REDUCTION_FAMILIES
    } == {
        *(("addi", element) for element in ("i8", "i16", "i32", "i64")),
        *(("muli", element) for element in ("i16", "i32", "i64")),
        *(
            (operation, element)
            for operation in ("minsi", "maxsi", "minui", "maxui")
            for element in ("i8", "i16", "i32", "i64")
        ),
        *(
            (operation, element)
            for operation in ("andi", "ori", "xori")
            for element in ("i8", "i16", "i32", "i64")
        ),
    }


def test_avx512_family_rows_materialize_complete_zmm_arithmetic() -> None:
    assert AVX512_VECTOR_BIT_WIDTHS == (512,)
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    family_mnemonics = {
        family.mnemonic
        for family in (
            *AVX512_INTEGER_BINARY_FAMILIES,
            *AVX512_FLOAT_BINARY_FAMILIES,
        )
    }
    family_mnemonics.update(mnemonic for _, mnemonic, _ in AVX512_BITWISE_FAMILIES)
    family_mnemonics.update(AVX512_FLOAT_FMA_MNEMONICS.values())
    assert {f"x86.avx512.{mnemonic}.zmm" for mnemonic in family_mnemonics} <= (
        descriptor_keys
    )


def test_avx512vl_rows_materialize_only_non_avx2_integer_cells() -> None:
    assert AVX512VL_VECTOR_BIT_WIDTHS == AVX2_VECTOR_BIT_WIDTHS
    assert {
        (family.source_operation, family.element.name)
        for family in AVX512VL_INTEGER_BINARY_FAMILIES
    } == {
        ("muli", "i64"),
        ("minsi", "i64"),
        ("maxsi", "i64"),
        ("minui", "i64"),
        ("maxui", "i64"),
        ("shli", "i16"),
        ("shrsi", "i16"),
        ("shrui", "i16"),
        ("shrsi", "i64"),
    }

    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    register_suffixes = {128: "xmm", 256: "ymm"}
    assert {
        f"x86.avx512.{family.mnemonic}.{register_suffixes[vector_bit_width]}"
        for family in AVX512VL_INTEGER_BINARY_FAMILIES
        for vector_bit_width in AVX512VL_VECTOR_BIT_WIDTHS
    } <= descriptor_keys


def test_avx512_direct_broadcasts_cover_every_width_and_payload_size() -> None:
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    expected_keys = {
        f"x86.avx512.{mnemonic}.{register_suffixes[vector_bit_width]}"
        for mnemonic in ("vpbroadcastb", "vpbroadcastw", "vpbroadcastd", "vpbroadcastq")
        for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
    }
    assert expected_keys <= descriptor_keys


def test_avx512_predicate_descriptors_cover_every_core_width_and_element() -> None:
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    assert {
        f"x86.avx512.{mnemonic}.{register_suffixes[vector_bit_width]}"
        for vector_bit_width in (128, 256, 512)
        for mnemonics in AVX512_INTEGER_COMPARE_MNEMONICS.values()
        for mnemonic in mnemonics
    } <= descriptor_keys
    assert {
        f"x86.avx512.{mnemonic}.{register_suffixes[vector_bit_width]}"
        for vector_bit_width in (128, 256, 512)
        for mnemonic in AVX512_FLOAT_COMPARE_MNEMONICS.values()
    } <= descriptor_keys
    assert {
        f"x86.avx512.{mnemonic}.{register_suffixes[vector_bit_width]}"
        for vector_bit_width in (128, 256, 512)
        for mnemonic in AVX512_SELECT_MNEMONICS.values()
    } <= descriptor_keys

    carrier_rows = {
        2: ("q", "xmm"),
        4: ("d", "xmm"),
        8: ("w", "xmm"),
        16: ("b", "xmm"),
        32: ("b", "ymm"),
        64: ("b", "zmm"),
    }
    assert {
        key
        for element_suffix, register_suffix in carrier_rows.values()
        for key in (
            f"x86.avx512.vpmovm2{element_suffix}.{register_suffix}.k",
            f"x86.avx512.vpmov{element_suffix}2m.k.{register_suffix}",
        )
    } <= descriptor_keys
    assert {
        "x86.avx512.kmovq.k.gpr64",
        "x86.avx512.kmovq.gpr64.k",
    } <= descriptor_keys
