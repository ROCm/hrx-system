# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared source data for x86 packed-dot contracts and low descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

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

FEATURE_AVX512_VNNI = 1 << 0
FEATURE_AVX512_VL = 1 << 1
FEATURE_AVX_VNNI = 1 << 2
FEATURE_AVX_VNNI_INT8 = 1 << 3
FEATURE_AVX_VNNI_INT16 = 1 << 4
FEATURE_AVX10_2 = 1 << 5
FEATURE_AVX512_BF16 = 1 << 6

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

LLVM_SOURCE_ABI_PAYLOAD = "LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD"
LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR = (
    "LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR"
)


@dataclass(frozen=True, slots=True)
class PackedDotDescriptor:
    key: str
    llvm_intrinsic_name: str
    llvm_source_abi: str
    mnemonic: str
    family: str
    required_feature_bits: int
    flags: int
    vector_bit_width: int
    lhs_numeric_type: str
    rhs_numeric_type: str
    accumulator_numeric_type: str
    result_numeric_type: str


def pd(
    key: str,
    llvm_intrinsic_name: str,
    llvm_source_abi: str,
    mnemonic: str,
    family: str,
    required_feature_bits: int,
    flags: int,
    vector_bit_width: int,
    lhs_numeric_type: str,
    rhs_numeric_type: str,
    accumulator_numeric_type: str,
    result_numeric_type: str,
) -> PackedDotDescriptor:
    return PackedDotDescriptor(
        key=key,
        llvm_intrinsic_name=llvm_intrinsic_name,
        llvm_source_abi=llvm_source_abi,
        mnemonic=mnemonic,
        family=family,
        required_feature_bits=required_feature_bits,
        flags=flags,
        vector_bit_width=vector_bit_width,
        lhs_numeric_type=lhs_numeric_type,
        rhs_numeric_type=rhs_numeric_type,
        accumulator_numeric_type=accumulator_numeric_type,
        result_numeric_type=result_numeric_type,
    )


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


X86_PACKED_DOT_DESCRIPTORS = (
    pd(
        "x86.avx512_vnni.vpdpbusd.xmm",
        "llvm.x86.avx512.vpdpbusd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        0,
        128,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpbusd.ymm",
        "llvm.x86.avx512.vpdpbusd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        0,
        256,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpbusd.zmm",
        "llvm.x86.avx512.vpdpbusd.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI,
        0,
        512,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpbusds.xmm",
        "llvm.x86.avx512.vpdpbusds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpbusds.ymm",
        "llvm.x86.avx512.vpdpbusds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpbusds.zmm",
        "llvm.x86.avx512.vpdpbusds.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI,
        CONTRACT_FLAG_SATURATING,
        512,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssd.xmm",
        "llvm.x86.avx512.vpdpwssd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        0,
        128,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssd.ymm",
        "llvm.x86.avx512.vpdpwssd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        0,
        256,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssd.zmm",
        "llvm.x86.avx512.vpdpwssd.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssd",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI,
        0,
        512,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssds.xmm",
        "llvm.x86.avx512.vpdpwssds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssds.ymm",
        "llvm.x86.avx512.vpdpwssds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx512_vnni.vpdpwssds.zmm",
        "llvm.x86.avx512.vpdpwssds.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssds",
        FAMILY_AVX512_VNNI,
        FEATURE_AVX512_VNNI,
        CONTRACT_FLAG_SATURATING,
        512,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpbusd.xmm",
        "llvm.x86.avx512.vpdpbusd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusd",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        0,
        128,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpbusd.ymm",
        "llvm.x86.avx512.vpdpbusd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusd",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        0,
        256,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpbusds.xmm",
        "llvm.x86.avx512.vpdpbusds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusds",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpbusds.ymm",
        "llvm.x86.avx512.vpdpbusds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbusds",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_U8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpwssd.xmm",
        "llvm.x86.avx512.vpdpwssd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssd",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        0,
        128,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpwssd.ymm",
        "llvm.x86.avx512.vpdpwssd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssd",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        0,
        256,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpwssds.xmm",
        "llvm.x86.avx512.vpdpwssds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssds",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni.vpdpwssds.ymm",
        "llvm.x86.avx512.vpdpwssds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwssds",
        FAMILY_AVX_VNNI,
        FEATURE_AVX_VNNI,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_I16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbssd.xmm",
        "llvm.x86.avx2.vpdpbssd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssd",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        128,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbssd.ymm",
        "llvm.x86.avx2.vpdpbssd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssd",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        256,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbssds.xmm",
        "llvm.x86.avx2.vpdpbssds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbssds.ymm",
        "llvm.x86.avx2.vpdpbssds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbsud.xmm",
        "llvm.x86.avx2.vpdpbsud.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsud",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        128,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbsud.ymm",
        "llvm.x86.avx2.vpdpbsud.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsud",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        256,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbsuds.xmm",
        "llvm.x86.avx2.vpdpbsuds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsuds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbsuds.ymm",
        "llvm.x86.avx2.vpdpbsuds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsuds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbuud.xmm",
        "llvm.x86.avx2.vpdpbuud.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuud",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        128,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbuud.ymm",
        "llvm.x86.avx2.vpdpbuud.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuud",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        0,
        256,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbuuds.xmm",
        "llvm.x86.avx2.vpdpbuuds.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuuds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        128,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int8.vpdpbuuds.ymm",
        "llvm.x86.avx2.vpdpbuuds.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuuds",
        FAMILY_AVX_VNNI_INT8,
        FEATURE_AVX_VNNI_INT8,
        CONTRACT_FLAG_SATURATING,
        256,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwsud.xmm",
        "llvm.x86.avx2.vpdpwsud.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwsud",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        128,
        NUMERIC_I16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwsud.ymm",
        "llvm.x86.avx2.vpdpwsud.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwsud",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        256,
        NUMERIC_I16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwusd.xmm",
        "llvm.x86.avx2.vpdpwusd.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwusd",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        128,
        NUMERIC_U16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwusd.ymm",
        "llvm.x86.avx2.vpdpwusd.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwusd",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        256,
        NUMERIC_U16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwuud.xmm",
        "llvm.x86.avx2.vpdpwuud.128",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwuud",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        128,
        NUMERIC_U16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx_vnni_int16.vpdpwuud.ymm",
        "llvm.x86.avx2.vpdpwuud.256",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwuud",
        FAMILY_AVX_VNNI_INT16,
        FEATURE_AVX_VNNI_INT16,
        0,
        256,
        NUMERIC_U16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbssd.zmm",
        "llvm.x86.avx10.vpdpbssd.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssd",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbsud.zmm",
        "llvm.x86.avx10.vpdpbsud.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsud",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbuud.zmm",
        "llvm.x86.avx10.vpdpbuud.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuud",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbssds.zmm",
        "llvm.x86.avx10.vpdpbssds.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbssds",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        CONTRACT_FLAG_SATURATING,
        512,
        NUMERIC_I8,
        NUMERIC_I8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbsuds.zmm",
        "llvm.x86.avx10.vpdpbsuds.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbsuds",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        CONTRACT_FLAG_SATURATING,
        512,
        NUMERIC_I8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpbuuds.zmm",
        "llvm.x86.avx10.vpdpbuuds.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpbuuds",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        CONTRACT_FLAG_SATURATING,
        512,
        NUMERIC_U8,
        NUMERIC_U8,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpwsud.zmm",
        "llvm.x86.avx10.vpdpwsud.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwsud",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_I16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpwusd.zmm",
        "llvm.x86.avx10.vpdpwusd.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwusd",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_U16,
        NUMERIC_I16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vpdpwuud.zmm",
        "llvm.x86.avx10.vpdpwuud.512",
        LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
        "vpdpwuud",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_U16,
        NUMERIC_U16,
        NUMERIC_I32,
        NUMERIC_I32,
    ),
    pd(
        "x86.avx10_2.vdpphps.xmm",
        "llvm.x86.avx10.vdpphps.128",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpphps",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        128,
        NUMERIC_F16,
        NUMERIC_F16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
    pd(
        "x86.avx10_2.vdpphps.ymm",
        "llvm.x86.avx10.vdpphps.256",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpphps",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        256,
        NUMERIC_F16,
        NUMERIC_F16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
    pd(
        "x86.avx10_2.vdpphps.zmm",
        "llvm.x86.avx10.vdpphps.512",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpphps",
        FAMILY_AVX10_2,
        FEATURE_AVX10_2,
        0,
        512,
        NUMERIC_F16,
        NUMERIC_F16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
    pd(
        "x86.avx512_bf16.vdpbf16ps.xmm",
        "llvm.x86.avx512bf16.dpbf16ps.128",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpbf16ps",
        FAMILY_AVX512_BF16,
        FEATURE_AVX512_BF16 | FEATURE_AVX512_VL,
        0,
        128,
        NUMERIC_BF16,
        NUMERIC_BF16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
    pd(
        "x86.avx512_bf16.vdpbf16ps.ymm",
        "llvm.x86.avx512bf16.dpbf16ps.256",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpbf16ps",
        FAMILY_AVX512_BF16,
        FEATURE_AVX512_BF16 | FEATURE_AVX512_VL,
        0,
        256,
        NUMERIC_BF16,
        NUMERIC_BF16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
    pd(
        "x86.avx512_bf16.vdpbf16ps.zmm",
        "llvm.x86.avx512bf16.dpbf16ps.512",
        LLVM_SOURCE_ABI_PAYLOAD,
        "vdpbf16ps",
        FAMILY_AVX512_BF16,
        FEATURE_AVX512_BF16,
        0,
        512,
        NUMERIC_BF16,
        NUMERIC_BF16,
        NUMERIC_F32,
        NUMERIC_F32,
    ),
)
