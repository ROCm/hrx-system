# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar and packed F32/F64-to-integer conversion descriptors."""

from collections.abc import Sequence

from loom.target.arch.x86.native_conversion import (
    scalar_float_integer_instruction,
    scalar_float_integer_mnemonic,
    vector_float_integer_instruction,
    vector_float_integer_mnemonic,
)
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.arch.x86.vector_families import (
    FLOAT_ELEMENTS,
    FloatIntegerVectorForm,
)
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_F32_ZMM,
    _asm,
    _avx512_descriptor,
    _gpr32_result,
    _gpr64_result,
    _vector_operand,
    _vector_result,
    _xmm_operand,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_SCHEDULES = {
    128: _SCHEDULE_VECTOR_F32_XMM,
    256: _SCHEDULE_VECTOR_F32_YMM,
    512: _SCHEDULE_VECTOR_F32_ZMM,
}


def scalar_float_integer_descriptors(*, unsigned: bool) -> tuple[Descriptor, ...]:
    feature = "avx512" if unsigned else "avx2"
    descriptors = []
    for source in FLOAT_ELEMENTS:
        for result_bit_width, result in (
            (32, _gpr32_result),
            (64, _gpr64_result),
        ):
            instruction = scalar_float_integer_instruction(
                source.bit_width, result_bit_width, unsigned=unsigned
            )
            mnemonic = scalar_float_integer_mnemonic(
                source.bit_width, unsigned=unsigned
            )
            descriptor = Descriptor(
                key=f"x86.{feature}.{mnemonic}.gpr{result_bit_width}.xmm",
                mnemonic=mnemonic,
                semantic_tag=(
                    f"float.fpto{'u' if unsigned else 's'}i."
                    f"{source.name}.i{result_bit_width}"
                ),
                operands=(result(), _xmm_operand("input")),
                asm_forms=_asm(
                    mnemonic=(
                        f"{'avx512.' if unsigned else ''}{mnemonic}."
                        f"gpr{result_bit_width}.xmm"
                    ),
                    results=("dst",),
                    operands=("input",),
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


def vector_float_integer_descriptors(
    forms: Sequence[FloatIntegerVectorForm],
    *,
    unsigned: bool,
    evex: bool,
) -> tuple[Descriptor, ...]:
    feature = "avx512" if evex else "avx2"
    descriptors = []
    for form in forms:
        source_bit_width = max(128, form.source_vector_bit_width)
        result_bit_width = max(128, form.result_vector_bit_width)
        source_suffix = _REGISTER_SUFFIXES[source_bit_width]
        result_suffix = _REGISTER_SUFFIXES[result_bit_width]
        instruction = vector_float_integer_instruction(
            form.source.bit_width,
            form.result_bit_width,
            unsigned=unsigned,
        )
        mnemonic = vector_float_integer_mnemonic(
            form.source.bit_width,
            form.result_bit_width,
            unsigned=unsigned,
        )
        descriptor = Descriptor(
            key=f"x86.{feature}.{mnemonic}.{result_suffix}.{source_suffix}",
            mnemonic=mnemonic,
            semantic_tag=(
                f"float.fpto{'u' if unsigned else 's'}i."
                f"{form.source.name}x{form.lane_count}."
                f"i{form.result_bit_width}x{form.lane_count}"
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
            _avx512_descriptor(
                descriptor,
                instruction,
                scalar=False,
            )
            if evex
            else instruction.bind(descriptor, VectorEncodingPrefix.VEX)
        )
    return tuple(descriptors)
