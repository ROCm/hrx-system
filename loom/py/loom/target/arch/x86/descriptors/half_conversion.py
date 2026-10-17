# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Legacy F16C and AVX512F/VL packed half conversion descriptors."""

from loom.target.arch.x86.feature_bits import FEATURE_F16C
from loom.target.arch.x86.native_conversion import half_conversion_instruction
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    Immediate,
    ImmediateKind,
)

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_F32_ZMM,
    _asm,
    _avx512_descriptor,
    _vector_operand,
    _vector_result,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_SCHEDULES = {
    128: _SCHEDULE_VECTOR_F32_XMM,
    256: _SCHEDULE_VECTOR_F32_YMM,
    512: _SCHEDULE_VECTOR_F32_ZMM,
}
_ROUNDING = Immediate("rounding", ImmediateKind.UNSIGNED, bit_width=8, unsigned_max=7)


def half_conversion_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    feature = "avx512" if evex else "f16c"
    assembly_prefix = "avx512." if evex else ""
    descriptors = []
    for narrow in (False, True):
        instruction = half_conversion_instruction(narrow=narrow)
        mnemonic = instruction.descriptor_mnemonic
        for wide_width in (128, 256, 512) if evex else (128, 256):
            half_width = max(128, wide_width // 2)
            input_width, result_width = (
                (wide_width, half_width) if narrow else (half_width, wide_width)
            )
            suffix = (
                f"{_REGISTER_SUFFIXES[result_width]}.{_REGISTER_SUFFIXES[input_width]}"
            )
            descriptor = Descriptor(
                key=f"x86.{feature}.{mnemonic}.{suffix}",
                mnemonic=mnemonic,
                semantic_tag=(
                    f"float.truncate.f32.f16x{wide_width // 32}"
                    if narrow
                    else f"float.extend.f16.f32x{wide_width // 32}"
                ),
                operands=(
                    _vector_result(result_width),
                    _vector_operand(input_width, "input"),
                ),
                immediates=(_ROUNDING,) if narrow else (),
                asm_forms=_asm(
                    mnemonic=f"{assembly_prefix}{mnemonic}.{suffix}",
                    results=("dst",),
                    operands=("input",),
                    immediates=("rounding",) if narrow else (),
                ),
                schedule_class=_SCHEDULES[wide_width],
                feature_mask_words=() if evex else (FEATURE_F16C,),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            descriptors.append(
                _avx512_descriptor(descriptor, instruction, scalar=False)
                if evex
                else instruction.bind(descriptor, VectorEncodingPrefix.VEX)
            )
    return tuple(descriptors)
