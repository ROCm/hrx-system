# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared source data for x86 packed-dot contracts and low descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX10_2,
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
    FEATURE_AVX512_VNNI,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
    FEATURE_AVX_VNNI_INT16,
)
from loom.target.native_contraction_layout import (
    ROLE_ACCUMULATOR,
    ROLE_LHS,
    ROLE_RESULT,
    ROLE_RHS,
    ExactContractionLayout,
    ExactContractionRoleLayout,
    contiguous_element_layout,
    grouped_dot_contraction_layout,
)
from loom.target.native_layout_facts import (
    NativeContractionFacts,
    NativeContractionRoleFacts,
    exact_native_contraction_role_facts,
)

CONTRACT_FLAG_SATURATING = 1 << 0

FAMILY_AVX512_VNNI = "LOOM_X86_PACKED_DOT_FAMILY_AVX512_VNNI"
FAMILY_AVX_VNNI = "LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI"
FAMILY_AVX_VNNI_INT8 = "LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT8"
FAMILY_AVX_VNNI_INT16 = "LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT16"
FAMILY_AVX10_2 = "LOOM_X86_PACKED_DOT_FAMILY_AVX10_2"
FAMILY_AVX512_BF16 = "LOOM_X86_PACKED_DOT_FAMILY_AVX512_BF16"

NUMERIC_I8 = "LOOM_X86_PACKED_DOT_NUMERIC_I8"
NUMERIC_U8 = "LOOM_X86_PACKED_DOT_NUMERIC_U8"
NUMERIC_I16 = "LOOM_X86_PACKED_DOT_NUMERIC_I16"
NUMERIC_U16 = "LOOM_X86_PACKED_DOT_NUMERIC_U16"
NUMERIC_F16 = "LOOM_X86_PACKED_DOT_NUMERIC_F16"
NUMERIC_BF16 = "LOOM_X86_PACKED_DOT_NUMERIC_BF16"
NUMERIC_I32 = "LOOM_X86_PACKED_DOT_NUMERIC_I32"
NUMERIC_F32 = "LOOM_X86_PACKED_DOT_NUMERIC_F32"

_NUMERIC_TYPE_BIT_COUNTS = {
    NUMERIC_I8: 8,
    NUMERIC_U8: 8,
    NUMERIC_I16: 16,
    NUMERIC_U16: 16,
    NUMERIC_F16: 16,
    NUMERIC_BF16: 16,
    NUMERIC_I32: 32,
    NUMERIC_F32: 32,
}


@dataclass(frozen=True, slots=True)
class PackedDotDescriptor:
    key: str
    mnemonic: str
    family: str
    required_feature_bits: int
    flags: int
    vector_bit_width: int
    lhs_numeric_type: str
    rhs_numeric_type: str
    accumulator_numeric_type: str
    result_numeric_type: str


@cache
def packed_dot_native_layout(
    descriptor: PackedDotDescriptor,
) -> ExactContractionLayout:
    """Returns the exact native contraction layout of an x86 packed dot."""

    lhs_bit_count = _NUMERIC_TYPE_BIT_COUNTS[descriptor.lhs_numeric_type]
    rhs_bit_count = _NUMERIC_TYPE_BIT_COUNTS[descriptor.rhs_numeric_type]
    accumulator_bit_count = _NUMERIC_TYPE_BIT_COUNTS[
        descriptor.accumulator_numeric_type
    ]
    result_bit_count = _NUMERIC_TYPE_BIT_COUNTS[descriptor.result_numeric_type]
    if lhs_bit_count != rhs_bit_count:
        raise ValueError(
            f"x86 packed-dot descriptor '{descriptor.key}' has differently "
            "sized source elements"
        )
    if accumulator_bit_count != result_bit_count:
        raise ValueError(
            f"x86 packed-dot descriptor '{descriptor.key}' has differently "
            "sized accumulator and result elements"
        )
    if (
        descriptor.vector_bit_width % lhs_bit_count != 0
        or descriptor.vector_bit_width % result_bit_count != 0
        or result_bit_count % lhs_bit_count != 0
    ):
        raise ValueError(
            f"x86 packed-dot descriptor '{descriptor.key}' cannot factor its "
            "vector and numeric widths into a contraction"
        )

    element_layouts = {
        role: contiguous_element_layout(
            key=f"{descriptor.key}.{role}",
            element_count=descriptor.vector_bit_width // atom_bit_width,
            atom_bit_width=atom_bit_width,
            physical_dimension_name="value",
        )
        for role, atom_bit_width in (
            (ROLE_LHS, lhs_bit_count),
            (ROLE_RHS, rhs_bit_count),
            (ROLE_ACCUMULATOR, accumulator_bit_count),
            (ROLE_RESULT, result_bit_count),
        )
    }
    return grouped_dot_contraction_layout(
        group_size=result_bit_count // lhs_bit_count,
        lhs=element_layouts[ROLE_LHS],
        rhs=element_layouts[ROLE_RHS],
        accumulator=element_layouts[ROLE_ACCUMULATOR],
        result=element_layouts[ROLE_RESULT],
    )


@cache
def packed_dot_native_contraction_facts(
    descriptor: PackedDotDescriptor,
) -> NativeContractionFacts:
    """Summarizes one x86 packed dot for shipping compiler consumers."""

    layout = packed_dot_native_layout(descriptor)
    element_bit_counts = {
        ROLE_LHS: _NUMERIC_TYPE_BIT_COUNTS[descriptor.lhs_numeric_type],
        ROLE_RHS: _NUMERIC_TYPE_BIT_COUNTS[descriptor.rhs_numeric_type],
        ROLE_ACCUMULATOR: _NUMERIC_TYPE_BIT_COUNTS[descriptor.accumulator_numeric_type],
        ROLE_RESULT: _NUMERIC_TYPE_BIT_COUNTS[descriptor.result_numeric_type],
    }

    def role_facts(
        role_layout: ExactContractionRoleLayout,
    ) -> NativeContractionRoleFacts:
        element_bit_count = element_bit_counts[role_layout.role]
        return exact_native_contraction_role_facts(
            role_layout.role,
            role_layout.coordinate_map,
            element_bit_count=element_bit_count,
            register_count=1,
            payload_element_count=descriptor.vector_bit_width // element_bit_count,
        )

    return NativeContractionFacts(
        shape=layout.shape,
        participant_count=1,
        lhs=role_facts(layout.lhs),
        rhs=role_facts(layout.rhs),
        accumulator=role_facts(layout.accumulator),
        result=role_facts(layout.result),
    )


# Integer variants share source signedness and the accumulator bit pattern.
# Saturation follows the ISA: unsigned/unsigned dots saturate unsigned; every
# other integer dot saturates signed. Numeric I32 describes the 32-bit carrier.
_VNNI_DOTS = (
    ("vpdpbusd", NUMERIC_U8, NUMERIC_I8),
    ("vpdpwssd", NUMERIC_I16, NUMERIC_I16),
)
_INT8_DOTS = (
    ("vpdpbssd", NUMERIC_I8, NUMERIC_I8),
    ("vpdpbsud", NUMERIC_I8, NUMERIC_U8),
    ("vpdpbuud", NUMERIC_U8, NUMERIC_U8),
)
_INT16_DOTS = (
    ("vpdpwsud", NUMERIC_I16, NUMERIC_U16),
    ("vpdpwusd", NUMERIC_U16, NUMERIC_I16),
    ("vpdpwuud", NUMERIC_U16, NUMERIC_U16),
)


def _integer_variants(
    dots: tuple[tuple[str, str, str], ...],
) -> tuple[tuple[str, str, str, str, int], ...]:
    return tuple(
        (mnemonic + suffix, lhs, rhs, NUMERIC_I32, flags)
        for mnemonic, lhs, rhs in dots
        for suffix, flags in (("", 0), ("s", CONTRACT_FLAG_SATURATING))
    )


# A family declares feature eligibility and native widths once. AVX512 narrow
# forms require VL, whereas AVX10.2 supplies its own width qualification.
_PACKED_DOT_FAMILIES = (
    (
        "avx512_vnni",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI,
        (128, 256, 512),
        FEATURE_AVX512_VL,
        _integer_variants(_VNNI_DOTS),
    ),
    (
        "avx_vnni",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        (128, 256),
        0,
        _integer_variants(_VNNI_DOTS),
    ),
    (
        "avx_vnni_int8",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        (128, 256),
        0,
        _integer_variants(_INT8_DOTS),
    ),
    (
        "avx_vnni_int16",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        (128, 256),
        0,
        _integer_variants(_INT16_DOTS),
    ),
    (
        "avx10_2",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        (128, 256, 512),
        0,
        (
            *_integer_variants((*_INT8_DOTS, *_INT16_DOTS)),
            ("vdpphps", NUMERIC_F16, NUMERIC_F16, NUMERIC_F32, 0),
        ),
    ),
    (
        "avx512_bf16",
        FAMILY_AVX512_BF16,
        FEATURE_AVX512_BF16,
        (128, 256, 512),
        FEATURE_AVX512_VL,
        (("vdpbf16ps", NUMERIC_BF16, NUMERIC_BF16, NUMERIC_F32, 0),),
    ),
)

_VECTOR_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}

X86_PACKED_DOT_DESCRIPTORS = tuple(
    PackedDotDescriptor(
        key=f"x86.{stem}.{mnemonic}.{_VECTOR_REGISTER_SUFFIXES[width]}",
        mnemonic=mnemonic,
        family=family,
        required_feature_bits=features | (narrow_features if width < 512 else 0),
        flags=flags,
        vector_bit_width=width,
        lhs_numeric_type=lhs,
        rhs_numeric_type=rhs,
        accumulator_numeric_type=accumulator,
        result_numeric_type=accumulator,
    )
    for (
        stem,
        family,
        features,
        widths,
        narrow_features,
        variants,
    ) in _PACKED_DOT_FAMILIES
    for mnemonic, lhs, rhs, accumulator, flags in variants
    for width in widths
)
