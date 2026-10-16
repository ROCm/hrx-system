# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""F32/F64 scalar and packed conversion descriptors."""

from loom.target.arch.x86.native_conversion import float_conversion_instruction
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _SCHEDULE_VECTOR_F32_YMM,
    _SCHEDULE_VECTOR_F32_ZMM,
    _asm,
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


def float_conversion_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    prefix = VectorEncodingPrefix.EVEX if evex else VectorEncodingPrefix.VEX
    feature = "avx512" if evex else "avx2"
    assembly_prefix = "avx512." if evex else ""
    descriptors = []
    double_widths = (128, 256, 512) if evex else (128, 256)
    for narrow in (False, True):
        scalar_mnemonic = "vcvtsd2ss" if narrow else "vcvtss2sd"
        scalar_semantic = "float.truncate.f64.f32" if narrow else "float.extend.f32.f64"
        scalar = Descriptor(
            key=f"x86.{feature}.{scalar_mnemonic}.xmm",
            mnemonic=scalar_mnemonic,
            semantic_tag=scalar_semantic,
            operands=(
                _xmm_result(),
                _xmm_operand("passthrough"),
                _xmm_operand("input"),
            ),
            asm_forms=_asm(
                mnemonic=f"{assembly_prefix}{scalar_mnemonic}.xmm",
                results=("dst",),
                operands=("passthrough", "input"),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        descriptors.append(
            float_conversion_instruction(narrow=narrow, scalar=True).bind(
                scalar, prefix
            )
        )

        mnemonic = "vcvtpd2ps" if narrow else "vcvtps2pd"
        instruction = float_conversion_instruction(narrow=narrow, scalar=False)
        for double_width in double_widths:
            float_width = max(128, double_width // 2)
            input_width, result_width = (
                (double_width, float_width) if narrow else (float_width, double_width)
            )
            suffix = (
                f"{_REGISTER_SUFFIXES[result_width]}.{_REGISTER_SUFFIXES[input_width]}"
            )
            lane_count = double_width // 64
            descriptor = Descriptor(
                key=f"x86.{feature}.{mnemonic}.{suffix}",
                mnemonic=mnemonic,
                semantic_tag=(
                    f"float.truncate.f64x{lane_count}.f32x{lane_count}"
                    if narrow
                    else f"float.extend.f32x{lane_count}.f64x{lane_count}"
                ),
                operands=(
                    _vector_result(result_width),
                    _vector_operand(input_width, "input"),
                ),
                asm_forms=_asm(
                    mnemonic=f"{assembly_prefix}{mnemonic}.{suffix}",
                    results=("dst",),
                    operands=("input",),
                ),
                schedule_class=_SCHEDULES[double_width],
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            descriptors.append(instruction.bind(descriptor, prefix))
    return tuple(descriptors)
