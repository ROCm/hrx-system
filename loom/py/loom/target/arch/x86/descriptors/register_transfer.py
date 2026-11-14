# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bit-preserving transfers between general-purpose and SIMD registers."""

from loom.target.arch.x86 import native_vector as native
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    InstructionClass,
)

from .common import (
    _SCHEDULE_VECTOR_I32_XMM,
    _asm,
    _avx512_descriptor,
    _gpr32_operand,
    _gpr32_result,
    _gpr64_operand,
    _gpr64_result,
    _vex_descriptor,
    _xmm_operand,
    _xmm_result,
)


def register_transfer_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    """Declares both directions of the 32- and 64-bit scalar bit transfers."""
    descriptors = []
    for mnemonic, suffix, result, operand, instructions in (
        (
            "vmovd",
            "gpr32",
            _gpr32_result,
            _gpr32_operand,
            (native.VMOVD_TO_GPR32, native.VMOVD_FROM_GPR32),
        ),
        (
            "vmovq",
            "gpr64",
            _gpr64_result,
            _gpr64_operand,
            (native.VMOVQ_TO_GPR64, native.VMOVQ_FROM_GPR64),
        ),
    ):
        for to_simd, instruction in zip((False, True), instructions, strict=True):
            destination, source = ("xmm", suffix) if to_simd else (suffix, "xmm")
            form = f"{mnemonic}.{destination}.{source}"
            descriptor = Descriptor(
                key=f"x86.{'avx512' if evex else 'avx2'}.{form}",
                mnemonic=mnemonic,
                semantic_tag=f"bits.move.{source}.{destination}",
                operands=(
                    (_xmm_result(), operand("input"))
                    if to_simd
                    else (result(), _xmm_operand("input"))
                ),
                constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0, 1),)
                if to_simd
                else (),
                asm_forms=_asm(
                    mnemonic=f"avx512.{form}" if evex else form,
                    results=("dst",),
                    operands=("input",),
                ),
                schedule_class=_SCHEDULE_VECTOR_I32_XMM,
                instruction_classes=(InstructionClass.REGISTER_MOVE,),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            descriptors.append(
                _avx512_descriptor(descriptor, instruction, scalar=True)
                if evex
                else _vex_descriptor(descriptor, instruction)
            )
    return tuple(descriptors)
