# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
    X86_AVX512_FP16_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT8_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_encoding import (
    VectorEncoding,
    VectorEncodingPrefix,
    VectorOpcodeMap,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_BINARY_FAMILIES,
    AVX2_FLOAT_COMPARE_MNEMONICS,
    AVX2_FLOAT_FMA_MNEMONICS,
    AVX2_INTEGER_BINARY_FAMILIES,
    AVX2_INTEGER_COMPARE_MNEMONICS,
    AVX2_UNIFORM_SHIFT_FAMILIES,
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
    AVX512_UNIFORM_SHIFT_FAMILIES,
    AVX512_VECTOR_BIT_WIDTHS,
    AVX512VL_INTEGER_BINARY_FAMILIES,
    AVX512VL_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    INTEGER_ELEMENTS,
    X86_VECTOR_INTEGER_NARROWING_FAMILIES,
    X86_VECTOR_INTEGER_WIDENING_FAMILIES,
)
from loom.target.low_descriptors import Constraint, ConstraintKind


def test_extended_evex_opcode_maps_preserve_legacy_encoding_ids() -> None:
    vex = VectorEncoding(
        VectorEncodingPrefix.VEX,
        VectorOpcodeMap.MAP_0F,
        0,
        0,
        0x58,
        (128,),
    )
    evex = VectorEncoding(
        VectorEncodingPrefix.EVEX,
        VectorOpcodeMap.MAP_0F,
        0,
        0,
        0x58,
        (128,),
    )
    map_5 = VectorEncoding(
        VectorEncodingPrefix.EVEX,
        VectorOpcodeMap.MAP_5,
        0,
        0,
        0x58,
        (128, 256, 512),
    )
    map_6 = VectorEncoding(
        VectorEncodingPrefix.EVEX,
        VectorOpcodeMap.MAP_6,
        1,
        0,
        0xB8,
        (128, 256, 512),
    )

    assert vex.encoding_id(128) == 0x0158
    assert evex.encoding_id(128) == 0x2158
    assert map_5.encoding_id(512) == 0x8058
    assert map_6.encoding_id(512) == 0xA4B8


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


def test_uniform_shift_matrix_and_native_encodings() -> None:
    expected = {
        (operation, element)
        for operation in ("shli", "shrsi", "shrui")
        for element in ("i16", "i32", "i64")
    }
    assert {
        (row.source_operation, row.element.name) for row in AVX2_UNIFORM_SHIFT_FAMILIES
    } == expected - {("shrsi", "i64")}
    assert {
        (row.source_operation, row.element.name)
        for row in AVX512_UNIFORM_SHIFT_FAMILIES
    } == expected

    # ISA opcode/group facts are independent of the family generator. Qword
    # arithmetic immediate shifts share opcode 72 with dwords and set EVEX.W.
    encodings = (
        ("vpsllw", 0x71, 6, 0xF1),
        ("vpslld", 0x72, 6, 0xF2),
        ("vpsllq", 0x73, 6, 0xF3),
        ("vpsraw", 0x71, 4, 0xE1),
        ("vpsrad", 0x72, 4, 0xE2),
        ("vpsrlw", 0x71, 2, 0xD1),
        ("vpsrld", 0x72, 2, 0xD2),
        ("vpsrlq", 0x73, 2, 0xD3),
        ("vpsraq", 0x72, 4, None),
    )
    for profile, descriptor_set in (
        ("avx2", X86_AVX2_DESCRIPTOR_SET),
        ("avx512", X86_AVX512_CORE_DESCRIPTOR_SET),
    ):
        descriptors = {
            descriptor.key: descriptor for descriptor in descriptor_set.descriptors
        }
        for mnemonic, opcode, group, count_opcode in encodings:
            if profile == "avx2" and count_opcode is None:
                continue
            widths = (
                (128, 256)
                if profile == "avx2"
                else ((128, 256, 512) if count_opcode is None else (512,))
            )
            for width in widths:
                suffix = {128: "xmm", 256: "ymm", 512: "zmm"}[width]
                prefix = f"x86.{profile}.{mnemonic}"
                descriptor = descriptors[f"{prefix}.{suffix}"]
                vector_length = {128: 0, 256: 1, 512: 2}[width]
                flags = 0x500 | (vector_length << 14)
                if profile == "avx512":
                    flags |= 0x2000 | (0x1000 if mnemonic.endswith("q") else 0)
                assert descriptor.encoding_id == flags | opcode
                assert (
                    descriptor.encoding_format_id
                    == {2: 0x9105, 4: 0x9107, 6: 0x9108}[group]
                )
                if profile == "avx2":
                    count_descriptor = descriptors[f"{prefix}.count.{suffix}"]
                    assert count_descriptor.encoding_id == flags | count_opcode
                    assert count_descriptor.encoding_format_id == 0x8210
                    count_operand = next(
                        operand
                        for operand in count_descriptor.operands
                        if operand.field_name == "count"
                    )
                    assert {
                        alternative.reg_class for alternative in count_operand.reg_alts
                    } == {"x86.xmm"}


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


def test_integer_widening_rows_cover_every_pair_signedness_and_register_width() -> None:
    assert {
        (family.source.name, family.result.name)
        for family in X86_VECTOR_INTEGER_WIDENING_FAMILIES
    } == {
        (source.name, result.name)
        for source_index, source in enumerate(INTEGER_ELEMENTS[:-1])
        for result in INTEGER_ELEMENTS[source_index + 1 :]
    }

    avx2_descriptors = {
        descriptor.key: descriptor for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    avx512_descriptors = {
        descriptor.key: descriptor
        for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    for family_index, family in enumerate(X86_VECTOR_INTEGER_WIDENING_FAMILIES):
        for mnemonic, opcode in (
            (family.signed_mnemonic, 0x20 + family_index),
            (family.unsigned_mnemonic, 0x30 + family_index),
        ):
            for result_vector_bit_width, result_suffix, encoding_prefix in (
                (128, "xmm", 0x0600),
                (256, "ymm", 0x4600),
            ):
                key = f"x86.avx2.{mnemonic}.{result_suffix}.xmm"
                descriptor = avx2_descriptors[key]
                assert descriptor.encoding_id == encoding_prefix | opcode
                lane_count = result_vector_bit_width // family.result.bit_width
                signedness = (
                    "signed" if mnemonic == family.signed_mnemonic else "unsigned"
                )
                assert descriptor.semantic_tag == (
                    f"integer.extend.{signedness}."
                    f"{family.source.name}x{lane_count}."
                    f"{family.result.name}x{lane_count}"
                )

            source_vector_bit_width = (
                512 * family.source.bit_width // family.result.bit_width
            )
            source_suffix = "ymm" if source_vector_bit_width == 256 else "xmm"
            descriptor = avx512_descriptors[
                f"x86.avx512.{mnemonic}.zmm.{source_suffix}"
            ]
            assert descriptor.encoding_id == 0xA600 | opcode


def test_integer_narrowing_rows_cover_every_pair_and_register_width() -> None:
    assert {
        (family.source.name, family.result.name)
        for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES
    } == {
        (source.name, result.name)
        for source_index, source in enumerate(INTEGER_ELEMENTS[1:], start=1)
        for result in INTEGER_ELEMENTS[:source_index]
    }

    descriptors = {
        descriptor.key: descriptor
        for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    vector_length_bits = {128: 0x0000, 256: 0x4000, 512: 0x8000}
    for family_index, family in enumerate(X86_VECTOR_INTEGER_NARROWING_FAMILIES):
        for source_vector_bit_width in (128, 256, 512):
            result_vector_bit_width = max(
                128,
                source_vector_bit_width
                * family.result.bit_width
                // family.source.bit_width,
            )
            descriptor = descriptors[
                f"x86.avx512.{family.mnemonic}."
                f"{suffixes[result_vector_bit_width]}."
                f"{suffixes[source_vector_bit_width]}"
            ]
            assert descriptor.encoding_id == (
                0x2A30 | family_index | vector_length_bits[source_vector_bit_width]
            )
            lane_count = source_vector_bit_width // family.source.bit_width
            assert descriptor.semantic_tag == (
                f"integer.truncate.{family.source.name}x{lane_count}."
                f"{family.result.name}x{lane_count}"
            )

    avx2_descriptors = {
        descriptor.key: descriptor for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    for mnemonic, element_bit_width, opcode in (
        ("vpunpcklwd", 16, 0x60),
        ("vpunpckldq", 32, 0x62),
        ("vpunpcklqdq", 64, 0x6C),
    ):
        descriptor = avx2_descriptors[f"x86.avx2.{mnemonic}.xmm"]
        assert descriptor.encoding_id == 0x0500 | opcode
        assert descriptor.semantic_tag == (
            f"bits.interleave.low.i{element_bit_width}x{128 // element_bit_width}"
        )


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


def test_vector_splat_materializations_replay_their_source_chain() -> None:
    descriptors = (
        *X86_AVX2_DESCRIPTOR_SET.descriptors,
        *X86_AVX512_CORE_DESCRIPTOR_SET.descriptors,
        *X86_AVX512_FP16_DESCRIPTOR_SET.descriptors,
    )
    rematerializable = Constraint(ConstraintKind.REMATERIALIZABLE, 0, 1)
    splat_descriptors = tuple(
        descriptor for descriptor in descriptors if ".splat." in descriptor.semantic_tag
    )
    assert splat_descriptors
    assert all(
        rematerializable in descriptor.constraints for descriptor in splat_descriptors
    )

    descriptors_by_key = {descriptor.key: descriptor for descriptor in descriptors}
    for key in (
        "x86.avx2.vmovd.xmm.gpr32",
        "x86.avx2.vmovq.xmm.gpr64",
    ):
        assert rematerializable in descriptors_by_key[key].constraints


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

    predicate_conversion_keys = {
        key
        for key in descriptor_keys
        if ".vpmovm2" in key or key.endswith(("2m.k.xmm", "2m.k.ymm", "2m.k.zmm"))
    }
    assert predicate_conversion_keys == {
        key
        for element_suffix in ("b", "w", "d", "q")
        for register_suffix in ("xmm", "ymm", "zmm")
        for key in (
            f"x86.avx512.vpmovm2{element_suffix}.{register_suffix}.k",
            f"x86.avx512.vpmov{element_suffix}2m.k.{register_suffix}",
        )
    }
    assert {
        "x86.avx512.kmovq.k.gpr64",
        "x86.avx512.kmovq.gpr64.k",
    } <= descriptor_keys
