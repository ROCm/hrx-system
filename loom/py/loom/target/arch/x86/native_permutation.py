# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native bit-preserving unpack and two-source permutation register forms."""

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
_R = VectorRegisterSelector


def _instruction(
    mnemonic: str,
    opcode: int,
    element_bit_width: int,
    *,
    destructive: bool,
) -> VectorMachineInstruction:
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            _R.RESULT,
            _R.INPUT_1 if destructive else _R.INPUT_0,
            _R.INPUT_2 if destructive else _R.INPUT_1,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            *(
                VectorOperandShape(OperandRole.OPERAND, ("x86.simd",))
                for _ in range(3 if destructive else 2)
            ),
        ),
        immediates=(),
        encodings=tuple(
            VectorEncoding(
                prefix=prefix,
                opcode_map=(
                    VectorOpcodeMap.MAP_0F38 if destructive else VectorOpcodeMap.MAP_0F
                ),
                mandatory_prefix=1,
                w=int(
                    prefix is _EVEX
                    and (
                        element_bit_width in (16, 64)
                        if destructive
                        else element_bit_width == 64
                    )
                ),
                opcode=opcode,
                vector_bit_widths=(128, 256) if prefix is _VEX else (128, 256, 512),
            )
            for prefix in ((_EVEX,) if destructive else (_VEX, _EVEX))
        ),
    )


UNPACK_INSTRUCTIONS = {
    (element_bit_width, high): _instruction(
        f"vpunpck{'h' if high else 'l'}{suffix}",
        opcode + (8 if high and element_bit_width != 64 else int(high)),
        element_bit_width,
        destructive=False,
    )
    for element_bit_width, suffix, opcode in (
        (8, "bw", 0x60),
        (16, "wd", 0x61),
        (32, "dq", 0x62),
        (64, "qdq", 0x6C),
    )
    for high in (False, True)
}

TWO_SOURCE_PERMUTE_INSTRUCTIONS = {
    (element_bit_width, overwrite_index): _instruction(
        f"vperm{'i' if overwrite_index else 't'}2{suffix}",
        opcode + (0 if overwrite_index else 8),
        element_bit_width,
        destructive=True,
    )
    for element_bit_width, suffix, opcode in (
        (8, "b", 0x75),
        (16, "w", 0x75),
        (32, "d", 0x76),
        (64, "q", 0x76),
    )
    for overwrite_index in (False, True)
}
