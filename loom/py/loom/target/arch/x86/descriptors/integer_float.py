# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar integer-to-F32/F64 conversion descriptors."""

from loom.target.arch.x86.native_conversion import integer_float_instruction
from loom.target.arch.x86.vector_encoding import VectorEncodingPrefix
from loom.target.arch.x86.vector_families import FLOAT_ELEMENTS
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_VECTOR_F32_XMM,
    _asm,
    _gpr32_operand,
    _gpr64_operand,
    _xmm_operand,
    _xmm_result,
)


def integer_float_descriptors(*, unsigned: bool) -> tuple[Descriptor, ...]:
    descriptors = []
    prefix = VectorEncodingPrefix.EVEX if unsigned else VectorEncodingPrefix.VEX
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
            descriptors.append(instruction.bind(descriptor, prefix))
    return tuple(descriptors)
