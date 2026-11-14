# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native encodings of destructive packed integer and floating-point dots."""

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

_VEX = VectorEncodingPrefix.VEX
_EVEX = VectorEncodingPrefix.EVEX


def _instruction(
    mnemonic: str,
    mandatory_prefix: int,
    opcode: int,
    prefixes: tuple[VectorEncodingPrefix, ...] = (_VEX, _EVEX),
) -> VectorMachineInstruction:
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.INPUT_1,
            VectorRegisterSelector.INPUT_2,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=tuple(
            VectorEncoding(
                prefix=prefix,
                opcode_map=VectorOpcodeMap.MAP_0F38,
                mandatory_prefix=mandatory_prefix,
                w=0,
                opcode=opcode,
                vector_bit_widths=(128, 256) if prefix is _VEX else (128, 256, 512),
            )
            for prefix in prefixes
        ),
    )


PACKED_DOT_INSTRUCTIONS = {
    instruction.descriptor_mnemonic: instruction
    for instruction in (
        _instruction("vpdpbusd", 1, 0x50),
        _instruction("vpdpbusds", 1, 0x51),
        _instruction("vpdpwssd", 1, 0x52),
        _instruction("vpdpwssds", 1, 0x53),
        _instruction("vpdpbssd", 3, 0x50),
        _instruction("vpdpbssds", 3, 0x51),
        _instruction("vpdpbsud", 2, 0x50),
        _instruction("vpdpbsuds", 2, 0x51),
        _instruction("vpdpbuud", 0, 0x50),
        _instruction("vpdpbuuds", 0, 0x51),
        _instruction("vpdpwsud", 2, 0xD2),
        _instruction("vpdpwsuds", 2, 0xD3),
        _instruction("vpdpwusd", 1, 0xD2),
        _instruction("vpdpwusds", 1, 0xD3),
        _instruction("vpdpwuud", 0, 0xD2),
        _instruction("vpdpwuuds", 0, 0xD3),
        _instruction("vdpbf16ps", 2, 0x52, (_EVEX,)),
        _instruction("vdpphps", 0, 0x52, (_EVEX,)),
    )
}
