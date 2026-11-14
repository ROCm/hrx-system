# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-half extraction, insertion, and concatenation of SIMD registers."""

from dataclasses import replace

from loom.target.arch.x86 import native_vector as native
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_I32_YMM,
    _SCHEDULE_VECTOR_I32_ZMM,
    _TWO_LANE_IMMEDIATE,
    _asm,
    _avx512_descriptor,
    _vector_operand,
    _vector_result,
    _vex_descriptor,
)


def vector_half_descriptors(vector_bit_width: int) -> tuple[Descriptor, ...]:
    half_bit_width = vector_bit_width // 2
    if vector_bit_width == 256:
        prefix, wide, narrow = "avx2", "ymm", "xmm"
        extract, insert = native.VEXTRACTF128, native.VINSERTF128
        extract_semantic = "float.extract.f32x8.half"
        insert_semantic = "bits.insert.v128.v256"
        extract_schedule = _SCHEDULE_VECTOR_F32_XMM
        insert_schedule = _SCHEDULE_VECTOR_I32_YMM
    else:
        prefix, wide, narrow = "avx512", "zmm", "ymm"
        extract, insert = native.VEXTRACTI64X4, native.VINSERTI64X4
        extract_semantic = "bits.extract.256.zmm"
        insert_semantic = "bits.insert.256.zmm"
        extract_schedule = insert_schedule = _SCHEDULE_VECTOR_I32_ZMM

    extract_descriptor = Descriptor(
        key=f"x86.{prefix}.{extract.descriptor_mnemonic}.{narrow}.{wide}",
        mnemonic=extract.descriptor_mnemonic,
        semantic_tag=extract_semantic,
        operands=(
            _vector_result(half_bit_width),
            _vector_operand(vector_bit_width, "source"),
        ),
        immediates=(_TWO_LANE_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=f"{extract.descriptor_mnemonic}.{narrow}",
            results=("dst",),
            operands=("source",),
            immediates=("lane",),
        ),
        schedule_class=extract_schedule,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    insert_descriptor = Descriptor(
        key=f"x86.{prefix}.{insert.descriptor_mnemonic}.{wide}.{narrow}",
        mnemonic=insert.descriptor_mnemonic,
        semantic_tag=insert_semantic,
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "dest"),
            _vector_operand(half_bit_width, "value"),
        ),
        immediates=(_TWO_LANE_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=f"{insert.descriptor_mnemonic}.{wide}.{narrow}",
            results=("dst",),
            operands=("dest", "value"),
            immediates=("lane",),
        ),
        schedule_class=insert_schedule,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    # Fixed upper-half insertion reads only the low half of its first source.
    # Describing those exact inputs avoids creating an artificial wide carrier.
    concat_descriptor = Descriptor(
        key=f"x86.{prefix}.{insert.descriptor_mnemonic}.concat.{wide}",
        mnemonic=insert.descriptor_mnemonic,
        semantic_tag=f"bits.concat.{half_bit_width}.{vector_bit_width}",
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(half_bit_width, "low"),
            _vector_operand(half_bit_width, "high"),
        ),
        asm_forms=_asm(
            mnemonic=f"{insert.descriptor_mnemonic}.concat.{wide}",
            results=("dst",),
            operands=("low", "high"),
        ),
        schedule_class=insert_schedule,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return tuple(
        _vex_descriptor(descriptor, instruction)
        if vector_bit_width == 256
        else _avx512_descriptor(descriptor, instruction, scalar=False)
        for descriptor, instruction in (
            (extract_descriptor, extract),
            (insert_descriptor, insert),
            (concat_descriptor, replace(insert, immediates=(), fixed_immediate=1)),
        )
    )
