# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar and packed integer-to-F32/F64 conversion descriptors."""

from collections.abc import Sequence

from loom.target.arch.x86.native_conversion import (
    integer_float_instruction,
    vector_integer_float_instruction,
    vector_integer_float_mnemonic,
)
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.arch.x86.vector_families import (
    FLOAT_ELEMENTS,
    IntegerFloatVectorForm,
)
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_F32_ZMM,
    _asm,
    _avx512_descriptor,
    _gpr32_operand,
    _gpr64_operand,
    _vector_operand,
    _vector_result,
    _xmm_operand,
    _xmm_result,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_SCHEDULES = {
    128: _SCHEDULE_VECTOR_F32_XMM,
    256: _SCHEDULE_VECTOR_F32_YMM,
    512: _SCHEDULE_VECTOR_F32_ZMM,
}


def integer_float_descriptors(*, unsigned: bool) -> tuple[Descriptor, ...]:
    descriptors = []
    feature = "avx512" if unsigned else "avx2"
    for element in FLOAT_ELEMENTS:
        for width, operand in ((32, _gpr32_operand), (64, _gpr64_operand)):
            instruction = integer_float_instruction(
                width, element.bit_width, unsigned=unsigned
            )
            mnemonic = instruction.descriptor_mnemonic
            descriptor = Descriptor(
                key=f"x86.{feature}.{mnemonic}.xmm.gpr{width}",
                mnemonic=mnemonic,
                semantic_tag=(
                    f"integer.{'u' if unsigned else 's'}itofp.i{width}.{element.name}"
                ),
                operands=(
                    _xmm_result(),
                    _xmm_operand("passthrough"),
                    operand("input"),
                ),
                asm_forms=_asm(
                    mnemonic=f"{mnemonic}.xmm.gpr{width}",
                    results=("dst",),
                    operands=("passthrough", "input"),
                ),
                schedule_class=_SCHEDULE_VECTOR_F32_XMM,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            descriptors.append(
                _avx512_descriptor(descriptor, instruction, scalar=True)
                if unsigned
                else instruction.bind(descriptor, VectorEncodingPrefix.VEX)
            )
    return tuple(descriptors)


def vector_integer_float_descriptors(
    forms: Sequence[IntegerFloatVectorForm],
    *,
    unsigned: bool,
    evex: bool,
) -> tuple[Descriptor, ...]:
    feature = "avx512" if evex else "avx2"
    descriptors = []
    seen_keys: set[str] = set()
    for form in forms:
        source_bit_width = max(128, form.source_vector_bit_width)
        result_bit_width = max(128, form.result_vector_bit_width)
        source_suffix = _REGISTER_SUFFIXES[source_bit_width]
        result_suffix = _REGISTER_SUFFIXES[result_bit_width]
        instruction = vector_integer_float_instruction(
            form.source.bit_width,
            form.result.bit_width,
            unsigned=unsigned,
        )
        mnemonic = vector_integer_float_mnemonic(
            form.source.bit_width,
            form.result.bit_width,
            unsigned=unsigned,
        )
        key = f"x86.{feature}.{mnemonic}.{result_suffix}.{source_suffix}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        physical_lane_count = form.lane_count
        if (
            form.source.bit_width == form.result.bit_width
            and form.source_vector_bit_width < 128
        ):
            physical_lane_count = 128 // form.source.bit_width
        descriptor = Descriptor(
            key=key,
            mnemonic=mnemonic,
            semantic_tag=(
                f"integer.{'u' if unsigned else 's'}itofp."
                f"{form.source.name}x{physical_lane_count}."
                f"{form.result.name}x{physical_lane_count}"
            ),
            operands=(
                _vector_result(result_bit_width),
                _vector_operand(source_bit_width, "input"),
            ),
            asm_forms=_asm(
                mnemonic=(
                    f"{'avx512.' if evex else ''}{mnemonic}."
                    f"{result_suffix}.{source_suffix}"
                ),
                results=("dst",),
                operands=("input",),
            ),
            schedule_class=_SCHEDULES[max(source_bit_width, result_bit_width)],
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        descriptors.append(
            _avx512_descriptor(descriptor, instruction, scalar=False)
            if evex
            else instruction.bind(descriptor, VectorEncodingPrefix.VEX)
        )
    return tuple(descriptors)
