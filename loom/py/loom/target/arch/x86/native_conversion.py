# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native integer and floating-point conversion instruction families."""

from loom.target.arch.x86.vector_encoding import (
    VectorEncoding,
    VectorEncodingBehavior,
    VectorEncodingPrefix,
    VectorImmediateShape,
    VectorMachineInstruction,
    VectorOpcodeMap,
    VectorOperandShape,
    VectorRegisterSelector,
    vector_encoding_recipe,
)
from loom.target.low_descriptors import ImmediateKind, OperandRole


def half_conversion_instruction(*, narrow: bool) -> VectorMachineInstruction:
    mnemonic = "vcvtps2ph" if narrow else "vcvtph2ps"
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.INPUT_0 if narrow else VectorRegisterSelector.RESULT,
            VectorRegisterSelector.NONE,
            VectorRegisterSelector.RESULT if narrow else VectorRegisterSelector.INPUT_0,
            VectorEncodingBehavior.IMMEDIATE
            if narrow
            else VectorEncodingBehavior.REGISTERS,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(VectorImmediateShape(ImmediateKind.UNSIGNED, 8),) if narrow else (),
        encodings=tuple(
            VectorEncoding(
                prefix=prefix,
                opcode_map=VectorOpcodeMap.MAP_0F3A
                if narrow
                else VectorOpcodeMap.MAP_0F38,
                mandatory_prefix=1,
                w=0,
                opcode=0x1D if narrow else 0x13,
                vector_bit_widths=widths,
            )
            for prefix, widths in (
                (VectorEncodingPrefix.VEX, (128, 256)),
                (VectorEncodingPrefix.EVEX, (128, 256, 512)),
            )
        ),
    )


def float_conversion_instruction(
    *, narrow: bool, scalar: bool
) -> VectorMachineInstruction:
    if scalar:
        mnemonic = "vcvtsd2ss" if narrow else "vcvtss2sd"
    else:
        mnemonic = "vcvtpd2ps" if narrow else "vcvtps2pd"
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.INPUT_0 if scalar else VectorRegisterSelector.NONE,
            VectorRegisterSelector.INPUT_1
            if scalar
            else VectorRegisterSelector.INPUT_0,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            *(
                (VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),)
                if scalar
                else ()
            ),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=tuple(
            VectorEncoding(
                prefix=prefix,
                opcode_map=VectorOpcodeMap.MAP_0F,
                mandatory_prefix=(3 if scalar else 1)
                if narrow
                else (2 if scalar else 0),
                w=int(narrow and prefix is VectorEncodingPrefix.EVEX),
                opcode=0x5A,
                vector_bit_widths=(128,) if scalar else widths,
            )
            for prefix, widths in (
                (VectorEncodingPrefix.VEX, (128, 256)),
                (VectorEncodingPrefix.EVEX, (128, 256, 512)),
            )
        ),
    )


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


def vector_integer_float_mnemonic(
    source_bit_width: int, result_bit_width: int, *, unsigned: bool
) -> str:
    if source_bit_width not in (32, 64) or result_bit_width not in (32, 64):
        raise ValueError("packed integer conversion requires 32/64-bit elements")
    source_suffix = "dq" if source_bit_width == 32 else "qq"
    result_suffix = "ps" if result_bit_width == 32 else "pd"
    return f"vcvt{'u' if unsigned else ''}{source_suffix}2{result_suffix}"


def vector_integer_float_instruction(
    source_bit_width: int, result_bit_width: int, *, unsigned: bool
) -> VectorMachineInstruction:
    mnemonic = vector_integer_float_mnemonic(
        source_bit_width, result_bit_width, unsigned=unsigned
    )
    if result_bit_width == 32:
        opcode = 0x7A if unsigned else 0x5B
        mandatory_prefix = 3 if unsigned else 0
    else:
        opcode = 0x7A if unsigned else 0xE6
        mandatory_prefix = 2
    encodings = []
    if source_bit_width == 32 and not unsigned:
        encodings.append(
            VectorEncoding(
                prefix=VectorEncodingPrefix.VEX,
                opcode_map=VectorOpcodeMap.MAP_0F,
                mandatory_prefix=mandatory_prefix,
                w=0,
                opcode=opcode,
                vector_bit_widths=(128, 256),
            )
        )
    encodings.append(
        VectorEncoding(
            prefix=VectorEncodingPrefix.EVEX,
            opcode_map=VectorOpcodeMap.MAP_0F,
            mandatory_prefix=mandatory_prefix,
            w=int(source_bit_width == 64),
            opcode=opcode,
            vector_bit_widths=(128, 256, 512),
        )
    )
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.NONE,
            VectorRegisterSelector.INPUT_0,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=tuple(encodings),
    )


def scalar_float_integer_mnemonic(source_bit_width: int, *, unsigned: bool) -> str:
    if source_bit_width not in (32, 64):
        raise ValueError("scalar float conversion requires an F32/F64 source")
    return (
        f"vcvtt{'ss' if source_bit_width == 32 else 'sd'}2{'u' if unsigned else ''}si"
    )


def scalar_float_integer_instruction(
    source_bit_width: int, result_bit_width: int, *, unsigned: bool
) -> VectorMachineInstruction:
    if result_bit_width not in (32, 64):
        raise ValueError("scalar float conversion requires a 32/64-bit result")
    mnemonic = scalar_float_integer_mnemonic(source_bit_width, unsigned=unsigned)
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.NONE,
            VectorRegisterSelector.INPUT_0,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, (f"x86.gpr{result_bit_width}",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=(
            VectorEncoding(
                prefix=(
                    VectorEncodingPrefix.EVEX if unsigned else VectorEncodingPrefix.VEX
                ),
                opcode_map=VectorOpcodeMap.MAP_0F,
                mandatory_prefix=2 if source_bit_width == 32 else 3,
                w=int(result_bit_width == 64),
                opcode=0x78 if unsigned else 0x2C,
                vector_bit_widths=(128,),
            ),
        ),
    )


def vector_float_integer_mnemonic(
    source_bit_width: int, result_bit_width: int, *, unsigned: bool
) -> str:
    if source_bit_width not in (32, 64) or result_bit_width not in (32, 64):
        raise ValueError("packed float conversion requires 32/64-bit elements")
    source_suffix = "ps" if source_bit_width == 32 else "pd"
    result_suffix = "dq" if result_bit_width == 32 else "qq"
    return f"vcvtt{source_suffix}2{'u' if unsigned else ''}{result_suffix}"


def vector_float_integer_instruction(
    source_bit_width: int, result_bit_width: int, *, unsigned: bool
) -> VectorMachineInstruction:
    mnemonic = vector_float_integer_mnemonic(
        source_bit_width, result_bit_width, unsigned=unsigned
    )
    if result_bit_width == 32 and not unsigned:
        opcode = 0x5B if source_bit_width == 32 else 0xE6
    elif result_bit_width == 64 and not unsigned:
        opcode = 0x7A
    else:
        opcode = 0x78
    mandatory_prefix = (
        2
        if result_bit_width == 32 and not unsigned and source_bit_width == 32
        else 1
        if result_bit_width == 64 or (result_bit_width == 32 and not unsigned)
        else 0
    )
    encodings = []
    if result_bit_width == 32 and not unsigned:
        encodings.append(
            VectorEncoding(
                prefix=VectorEncodingPrefix.VEX,
                opcode_map=VectorOpcodeMap.MAP_0F,
                mandatory_prefix=mandatory_prefix,
                w=0,
                opcode=opcode,
                vector_bit_widths=(128, 256),
            )
        )
    encodings.append(
        VectorEncoding(
            prefix=VectorEncodingPrefix.EVEX,
            opcode_map=VectorOpcodeMap.MAP_0F,
            mandatory_prefix=mandatory_prefix,
            w=int(source_bit_width == 64),
            opcode=opcode,
            vector_bit_widths=(128, 256, 512),
        )
    )
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.NONE,
            VectorRegisterSelector.INPUT_0,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=tuple(encodings),
    )
