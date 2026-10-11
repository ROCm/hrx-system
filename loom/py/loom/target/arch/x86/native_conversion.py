# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native GPR-to-scalar floating-point instruction family."""

from loom.target.arch.x86.vector_encoding import (
    VectorEncoding,
    VectorEncodingPrefix,
    VectorMachineInstruction,
    VectorOpcodeMap,
    VectorOperandShape,
    VectorRegisterSelector,
    vector_encoding_recipe,
)
from loom.target.low_descriptors import OperandRole


def integer_float_instruction(
    integer_bit_width: int, float_bit_width: int, *, unsigned: bool
) -> VectorMachineInstruction:
    if integer_bit_width not in (32, 64) or float_bit_width not in (32, 64):
        raise ValueError("scalar integer conversion requires 32/64-bit operands")
    mnemonic = ("vcvtusi2" if unsigned else "vcvtsi2") + (
        "ss" if float_bit_width == 32 else "sd"
    )
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.INPUT_0,
            VectorRegisterSelector.INPUT_1,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, (f"x86.gpr{integer_bit_width}",)),
        ),
        immediates=(),
        encodings=(
            VectorEncoding(
                prefix=(
                    VectorEncodingPrefix.EVEX if unsigned else VectorEncodingPrefix.VEX
                ),
                opcode_map=VectorOpcodeMap.MAP_0F,
                mandatory_prefix=2 if float_bit_width == 32 else 3,
                w=int(integer_bit_width == 64),
                opcode=0x7B if unsigned else 0x2A,
                vector_bit_widths=(128,),
            ),
        ),
    )
