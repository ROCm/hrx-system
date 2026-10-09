# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed VEX/EVEX facts referenced directly by x86 descriptor families."""

from __future__ import annotations

from loom.target.low_descriptors import ImmediateKind, OperandRole

from .vector_encoding import (
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

_R = VectorRegisterSelector
_B = VectorEncodingBehavior
_VEX = VectorEncodingPrefix.VEX
_EVEX = VectorEncodingPrefix.EVEX
_MAP_1 = VectorOpcodeMap.MAP_0F
_MAP_2 = VectorOpcodeMap.MAP_0F38
_MAP_3 = VectorOpcodeMap.MAP_0F3A
_MAP_5 = VectorOpcodeMap.MAP_5
_MAP_6 = VectorOpcodeMap.MAP_6

_ZERO = vector_encoding_recipe(_R.RESULT, _R.RESULT, _R.RESULT)
_REVERSE_UNARY = vector_encoding_recipe(_R.INPUT_0, _R.NONE, _R.RESULT)
_NDS_SWAPPED_INPUTS = vector_encoding_recipe(_R.RESULT, _R.INPUT_1, _R.INPUT_0)
_UNARY = vector_encoding_recipe(_R.RESULT, _R.NONE, _R.INPUT_0)
_NDS = vector_encoding_recipe(_R.RESULT, _R.INPUT_0, _R.INPUT_1)
_DESTRUCTIVE_NDS = vector_encoding_recipe(_R.RESULT, _R.INPUT_1, _R.INPUT_2)
_REVERSE_UNARY_IMMEDIATE = vector_encoding_recipe(
    _R.INPUT_0, _R.NONE, _R.RESULT, _B.IMMEDIATE
)
_SHIFT_IMMEDIATE_2 = vector_encoding_recipe(
    _R.FIXED_2, _R.RESULT, _R.INPUT_0, _B.IMMEDIATE
)
_SHIFT_IMMEDIATE_3 = vector_encoding_recipe(
    _R.FIXED_3, _R.RESULT, _R.INPUT_0, _B.IMMEDIATE
)
_SHIFT_IMMEDIATE_4 = vector_encoding_recipe(
    _R.FIXED_4, _R.RESULT, _R.INPUT_0, _B.IMMEDIATE
)
_SHIFT_IMMEDIATE_6 = vector_encoding_recipe(
    _R.FIXED_6, _R.RESULT, _R.INPUT_0, _B.IMMEDIATE
)
_UNARY_IMMEDIATE = vector_encoding_recipe(_R.RESULT, _R.NONE, _R.INPUT_0, _B.IMMEDIATE)
_NDS_IMMEDIATE = vector_encoding_recipe(_R.RESULT, _R.INPUT_0, _R.INPUT_1, _B.IMMEDIATE)
_VEX_BLENDV = vector_encoding_recipe(_R.RESULT, _R.INPUT_0, _R.INPUT_1, _B.BLEND_MASK)
_EVEX_MASK_SELECT = vector_encoding_recipe(
    _R.RESULT, _R.INPUT_2, _R.INPUT_1, _B.EVEX_MASK
)
_INDEXED_LOAD = vector_encoding_recipe(
    _R.RESULT, _R.NONE, _R.INPUT_0, _B.LOAD, full_vector_tuple=True
)
_LOAD = vector_encoding_recipe(
    _R.RESULT, _R.NONE, _R.INPUT_0, _B.LOAD, full_vector_tuple=True
)
_INDEXED_STORE = vector_encoding_recipe(
    _R.INPUT_0, _R.NONE, _R.INPUT_1, _B.STORE, full_vector_tuple=True
)
_STORE = vector_encoding_recipe(
    _R.INPUT_0, _R.NONE, _R.INPUT_1, _B.STORE, full_vector_tuple=True
)
_RIP_RELATIVE_LOAD = vector_encoding_recipe(_R.RESULT, _R.NONE, _R.NONE, _B.RIP_LOAD)

_SIMD = ("x86.simd",)
_GPR32 = ("x86.gpr32",)
_GPR64 = ("x86.gpr64",)
_K = ("x86.k",)


def _operand(role: OperandRole, classes: tuple[str, ...]) -> VectorOperandShape:
    return VectorOperandShape(role, classes)


_OPERANDS_0 = (
    _operand(OperandRole.OPERAND, _SIMD),
    _operand(OperandRole.RESOURCE, _GPR64),
)
_OPERANDS_1 = (*_OPERANDS_0, _operand(OperandRole.RESOURCE, _GPR64))
_OPERANDS_2 = (
    _operand(OperandRole.RESULT, _GPR32),
    _operand(OperandRole.OPERAND, _SIMD),
)
_OPERANDS_3 = (_operand(OperandRole.RESULT, _GPR64), _operand(OperandRole.OPERAND, _K))
_OPERANDS_4 = (
    _operand(OperandRole.RESULT, _GPR64),
    _operand(OperandRole.OPERAND, _SIMD),
)
_OPERANDS_5 = (_operand(OperandRole.RESULT, _K), _operand(OperandRole.OPERAND, _GPR64))
_OPERANDS_6 = (
    _operand(OperandRole.RESULT, _K),
    _operand(OperandRole.OPERAND, _K),
    _operand(OperandRole.OPERAND, _K),
)
_OPERANDS_7 = (_operand(OperandRole.RESULT, _K), _operand(OperandRole.OPERAND, _SIMD))
_OPERANDS_8 = (
    _operand(OperandRole.RESULT, _K),
    _operand(OperandRole.OPERAND, _SIMD),
    _operand(OperandRole.OPERAND, _SIMD),
)
_OPERANDS_9 = (
    _operand(OperandRole.RESULT, _SIMD),
    _operand(OperandRole.OPERAND, _GPR32),
)
_OPERANDS_10 = (
    _operand(OperandRole.RESULT, _SIMD),
    _operand(OperandRole.OPERAND, _GPR64),
)
_OPERANDS_11 = (_operand(OperandRole.RESULT, _SIMD), _operand(OperandRole.OPERAND, _K))
_OPERANDS_12 = (
    _operand(OperandRole.RESULT, _SIMD),
    _operand(OperandRole.OPERAND, _K),
    _operand(OperandRole.OPERAND, _SIMD),
    _operand(OperandRole.OPERAND, _SIMD),
)
_OPERANDS_13 = (
    _operand(OperandRole.RESULT, _SIMD),
    _operand(OperandRole.OPERAND, _SIMD),
)
_OPERANDS_14 = (*_OPERANDS_13, _operand(OperandRole.OPERAND, _GPR32))
_OPERANDS_15 = (*_OPERANDS_13, _operand(OperandRole.OPERAND, _GPR64))
_OPERANDS_16 = (*_OPERANDS_13, _operand(OperandRole.OPERAND, _SIMD))
_OPERANDS_17 = (*_OPERANDS_16, _operand(OperandRole.OPERAND, _SIMD))
_OPERANDS_18 = (
    _operand(OperandRole.RESULT, _SIMD),
    _operand(OperandRole.RESOURCE, _GPR64),
)
_OPERANDS_19 = (*_OPERANDS_18, _operand(OperandRole.RESOURCE, _GPR64))
_OPERANDS_20 = (_operand(OperandRole.RESULT, _SIMD),)

_IMMEDIATES_0: tuple[VectorImmediateShape, ...] = ()
_IMMEDIATES_1 = (VectorImmediateShape(ImmediateKind.ORDINAL, 32),)
_IMMEDIATES_2 = (
    VectorImmediateShape(ImmediateKind.SIGNED, 32),
    VectorImmediateShape(ImmediateKind.ENUM, 8),
)
_IMMEDIATES_3 = (VectorImmediateShape(ImmediateKind.SIGNED, 32),)
_IMMEDIATES_4 = (VectorImmediateShape(ImmediateKind.UNSIGNED, 5),)
_IMMEDIATES_5 = (VectorImmediateShape(ImmediateKind.UNSIGNED, 8),)


def _encoding(
    prefix: VectorEncodingPrefix,
    opcode_map: VectorOpcodeMap,
    mandatory_prefix: int,
    w: int,
    opcode: int,
    vector_bit_widths: tuple[int, ...] = (),
    *,
    fixed: int | None = None,
) -> VectorEncoding:
    return VectorEncoding(
        prefix,
        opcode_map,
        mandatory_prefix,
        w,
        opcode,
        vector_bit_widths,
        fixed,
    )


def _instruction(
    mnemonic: str,
    recipe: int,
    operands: tuple[VectorOperandShape, ...],
    immediates: tuple[VectorImmediateShape, ...],
    *encodings: VectorEncoding,
) -> VectorMachineInstruction:
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=recipe,
        operands=operands,
        immediates=immediates,
        encodings=encodings,
    )


def _vex_memory_load_instructions(
    mnemonic: str, mandatory_prefix: int, opcode: int
) -> tuple[VectorMachineInstruction, VectorMachineInstruction]:
    encoding = _encoding(_VEX, _MAP_2, mandatory_prefix, 0, opcode, (128, 256))
    return (
        _instruction(mnemonic, _LOAD, _OPERANDS_18, _IMMEDIATES_3, encoding),
        _instruction(mnemonic, _INDEXED_LOAD, _OPERANDS_19, _IMMEDIATES_2, encoding),
    )


KANDQ = _instruction(
    "kandq",
    _NDS,
    _OPERANDS_6,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 0, 1, 0x41, fixed=1),
)
KMOVQ_TO_GPR64 = _instruction(
    "kmovq",
    _UNARY,
    _OPERANDS_3,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 1, 0x93, fixed=0),
)
KMOVQ_FROM_GPR64 = _instruction(
    "kmovq",
    _UNARY,
    _OPERANDS_5,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 1, 0x92, fixed=0),
)
KORQ = _instruction(
    "korq",
    _NDS,
    _OPERANDS_6,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 0, 1, 0x45, fixed=1),
)
KXORQ = _instruction(
    "kxorq",
    _NDS,
    _OPERANDS_6,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 0, 1, 0x47, fixed=1),
)
VADDPH = _instruction(
    "vaddph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x58, (128, 256, 512)),
)
VADDSH = _instruction(
    "vaddsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x58, (128,)),
)
VADDPD = _instruction(
    "vaddpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x58, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x58, (128, 256)),
)
VADDPS = _instruction(
    "vaddps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x58, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x58, (128, 256)),
)
VADDSD = _instruction(
    "vaddsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x58, (128,)),
)
VADDSS = _instruction(
    "vaddss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x58, (128,)),
)
VBCSTNEBF162PS_MEMORY = _vex_memory_load_instructions("vbcstnebf162ps", 2, 0xB1)
VBCSTNESH2PS_MEMORY = _vex_memory_load_instructions("vbcstnesh2ps", 1, 0xB1)
VBLENDMPD = _instruction(
    "vblendmpd",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x65, (128, 256, 512)),
)
VBLENDMPS = _instruction(
    "vblendmps",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x65, (128, 256, 512)),
)
VBLENDVPD = _instruction(
    "vblendvpd",
    _VEX_BLENDV,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_3, 1, 0, 0x4B, (128,)),
)
VBLENDVPS = _instruction(
    "vblendvps",
    _VEX_BLENDV,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_3, 1, 0, 0x4A, (256,)),
)
VBROADCASTSD = _instruction(
    "vbroadcastsd",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x19, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x19, (256,)),
    _encoding(_VEX, _MAP_1, 3, 0, 0x12, (128,)),
)
VBROADCASTSS = _instruction(
    "vbroadcastss",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x18, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x18, (128, 256)),
)
VCMPPD_MASK_COMPARE = _instruction(
    "vcmppd",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_1, 1, 1, 0xC2, (128, 256, 512)),
)
VCMPPD_VECTOR_BINARY = _instruction(
    "vcmppd",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_4,
    _encoding(_VEX, _MAP_1, 1, 0, 0xC2, (128, 256)),
)
VCMPPS_MASK_COMPARE = _instruction(
    "vcmpps",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_1, 0, 0, 0xC2, (128, 256, 512)),
)
VCMPPS_VECTOR_BINARY = _instruction(
    "vcmpps",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_4,
    _encoding(_VEX, _MAP_1, 0, 0, 0xC2, (128, 256)),
)
VCMPPH = _instruction(
    "vcmpph",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 0, 0, 0xC2, (128, 256, 512)),
)
VCMPSH = _instruction(
    "vcmpsh",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 2, 0, 0xC2, (128,)),
)
VCVTNEEBF162PS_MEMORY = _vex_memory_load_instructions("vcvtneebf162ps", 2, 0xB0)
VCVTNEEPH2PS_MEMORY = _vex_memory_load_instructions("vcvtneeph2ps", 1, 0xB0)
VCVTNEOBF162PS_MEMORY = _vex_memory_load_instructions("vcvtneobf162ps", 3, 0xB0)
VCVTNEOPH2PS_MEMORY = _vex_memory_load_instructions("vcvtneoph2ps", 0, 0xB0)
VCVTNEPS2BF16 = _instruction(
    "vcvtneps2bf16",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 2, 0, 0x72, (128, 256)),
)
VCVTPH2PSX = _instruction(
    "vcvtph2psx",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_6, 1, 0, 0x13, (128, 256, 512)),
)
VCVTPS2PHX = _instruction(
    "vcvtps2phx",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 1, 0, 0x1D, (128, 256, 512)),
)
VCVTSH2SS = _instruction(
    "vcvtsh2ss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_6, 0, 0, 0x13, (128,)),
)
VCVTSS2SH = _instruction(
    "vcvtss2sh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x1D, (128,)),
)
VDIVPD = _instruction(
    "vdivpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x5E, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x5E, (128, 256)),
)
VDIVPS = _instruction(
    "vdivps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x5E, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x5E, (128, 256)),
)
VDIVSD = _instruction(
    "vdivsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x5E, (128,)),
)
VDIVSS = _instruction(
    "vdivss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x5E, (128,)),
)
VDIVPH = _instruction(
    "vdivph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x5E, (128, 256, 512)),
)
VDIVSH = _instruction(
    "vdivsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x5E, (128,)),
)
VEXTRACTF128 = _instruction(
    "vextractf128",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x19, (256,)),
)
VEXTRACTF32X4 = _instruction(
    "vextractf32x4",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x19, (512,)),
)
VEXTRACTI64X4 = _instruction(
    "vextracti64x4",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x3B, (512,)),
)
VFMADD231PD = _instruction(
    "vfmadd231pd",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0xB8, (512,)),
    _encoding(_VEX, _MAP_2, 1, 1, 0xB8, (128, 256)),
)
VFMADD231PS = _instruction(
    "vfmadd231ps",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0xB8, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0xB8, (128, 256)),
)
VFMADD231SD = _instruction(
    "vfmadd231sd",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 1, 0xB9, (128,)),
)
VFMADD231SS = _instruction(
    "vfmadd231ss",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0xB9, (128,)),
)
VFMADD231PH = _instruction(
    "vfmadd231ph",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_6, 1, 0, 0xB8, (128, 256, 512)),
)
VFMADD231SH = _instruction(
    "vfmadd231sh",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_6, 1, 0, 0xB9, (128,)),
)
VINSERTF128 = _instruction(
    "vinsertf128",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x18, (256,)),
)
VINSERTF32X4 = _instruction(
    "vinsertf32x4",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x18, (512,)),
)
VINSERTI64X4 = _instruction(
    "vinserti64x4",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x3A, (512,)),
)
VINSERTPS = _instruction(
    "vinsertps",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x21, (128,)),
)
VMAXPD = _instruction(
    "vmaxpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x5F, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x5F, (128, 256)),
)
VMAXPS = _instruction(
    "vmaxps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x5F, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x5F, (128, 256)),
)
VMAXSD = _instruction(
    "vmaxsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x5F, (128,)),
)
VMAXSS = _instruction(
    "vmaxss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x5F, (128,)),
)
VMAXPH = _instruction(
    "vmaxph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x5F, (128, 256, 512)),
)
VMAXSH = _instruction(
    "vmaxsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x5F, (128,)),
)
VMINPD = _instruction(
    "vminpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x5D, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x5D, (128, 256)),
)
VMINPS = _instruction(
    "vminps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x5D, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x5D, (128, 256)),
)
VMINSD = _instruction(
    "vminsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x5D, (128,)),
)
VMINSS = _instruction(
    "vminss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x5D, (128,)),
)
VMINPH = _instruction(
    "vminph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x5D, (128, 256, 512)),
)
VMINSH = _instruction(
    "vminsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x5D, (128,)),
)
VMOVD_TO_GPR32 = _instruction(
    "vmovd",
    _REVERSE_UNARY,
    _OPERANDS_2,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x7E, (128,)),
)
VMOVD_FROM_GPR32 = _instruction(
    "vmovd",
    _UNARY,
    _OPERANDS_9,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x6E, (128,)),
)
VMOVDQU32_INDEXED_LOAD = _instruction(
    "vmovdqu32",
    _INDEXED_LOAD,
    _OPERANDS_19,
    _IMMEDIATES_2,
    _encoding(_VEX, _MAP_1, 2, 0, 0x6F, (128, 256)),
    _encoding(_EVEX, _MAP_1, 2, 0, 0x6F, (512,)),
)
VMOVDQU32_LOAD = _instruction(
    "vmovdqu32",
    _LOAD,
    _OPERANDS_18,
    _IMMEDIATES_3,
    _encoding(_VEX, _MAP_1, 2, 0, 0x6F, (128, 256)),
    _encoding(_EVEX, _MAP_1, 2, 0, 0x6F, (512,)),
)
VMOVDQU32_INDEXED_STORE_SOURCE = _instruction(
    "vmovdqu32",
    _INDEXED_STORE,
    _OPERANDS_1,
    _IMMEDIATES_2,
    _encoding(_VEX, _MAP_1, 2, 0, 0x7F, (128, 256)),
    _encoding(_EVEX, _MAP_1, 2, 0, 0x7F, (512,)),
)
VMOVDQU32_STORE_SOURCE = _instruction(
    "vmovdqu32",
    _STORE,
    _OPERANDS_0,
    _IMMEDIATES_3,
    _encoding(_VEX, _MAP_1, 2, 0, 0x7F, (128, 256)),
    _encoding(_EVEX, _MAP_1, 2, 0, 0x7F, (512,)),
)
VMOVDQU = _instruction(
    "vmovdqu",
    _RIP_RELATIVE_LOAD,
    _OPERANDS_20,
    _IMMEDIATES_1,
    _encoding(_VEX, _MAP_1, 2, 0, 0x6F, (128, 256)),
)
VMOVDQU64 = _instruction(
    "vmovdqu64",
    _RIP_RELATIVE_LOAD,
    _OPERANDS_20,
    _IMMEDIATES_1,
    _encoding(_EVEX, _MAP_1, 2, 1, 0x6F, (512,)),
)
VMOVSD_INDEXED_LOAD = _instruction(
    "vmovsd",
    _INDEXED_LOAD,
    _OPERANDS_19,
    _IMMEDIATES_2,
    _encoding(_VEX, _MAP_1, 3, 0, 0x10, (128,)),
)
VMOVSD_LOAD = _instruction(
    "vmovsd",
    _LOAD,
    _OPERANDS_18,
    _IMMEDIATES_3,
    _encoding(_VEX, _MAP_1, 3, 0, 0x10, (128,)),
)
VMOVSD_INDEXED_STORE_SOURCE = _instruction(
    "vmovsd",
    _INDEXED_STORE,
    _OPERANDS_1,
    _IMMEDIATES_2,
    _encoding(_VEX, _MAP_1, 3, 0, 0x11, (128,)),
)
VMOVSD_STORE_SOURCE = _instruction(
    "vmovsd",
    _STORE,
    _OPERANDS_0,
    _IMMEDIATES_3,
    _encoding(_VEX, _MAP_1, 3, 0, 0x11, (128,)),
)
VMOVQ_TO_GPR64 = _instruction(
    "vmovq",
    _REVERSE_UNARY,
    _OPERANDS_4,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 1, 0x7E, (128,)),
)
VMOVQ_FROM_GPR64 = _instruction(
    "vmovq",
    _UNARY,
    _OPERANDS_10,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 1, 0x6E, (128,)),
)
VMULPD = _instruction(
    "vmulpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x59, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x59, (128, 256)),
)
VMULPS = _instruction(
    "vmulps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x59, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x59, (128, 256)),
)
VMULSD = _instruction(
    "vmulsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x59, (128,)),
)
VMULSS = _instruction(
    "vmulss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x59, (128,)),
)
VMULPH = _instruction(
    "vmulph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x59, (128, 256, 512)),
)
VMULSH = _instruction(
    "vmulsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x59, (128,)),
)
VPACKSSDW = _instruction(
    "vpackssdw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x6B, (128,)),
)
VPACKSSWB = _instruction(
    "vpacksswb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x63, (128,)),
)
VPACKUSWB = _instruction(
    "vpackuswb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x67, (128,)),
)
VPADDB = _instruction(
    "vpaddb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xFC, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xFC, (128, 256)),
)
VPADDD = _instruction(
    "vpaddd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xFE, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xFE, (128, 256)),
)
VPADDQ = _instruction(
    "vpaddq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0xD4, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xD4, (128, 256)),
)
VPADDW = _instruction(
    "vpaddw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xFD, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xFD, (128, 256)),
)
VPAND = _instruction(
    "vpand",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xDB, (128, 256)),
)
VPANDD = _instruction(
    "vpandd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xDB, (512,)),
)
VPBLENDMB = _instruction(
    "vpblendmb",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x66, (128, 256, 512)),
)
VPBLENDMD = _instruction(
    "vpblendmd",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x64, (128, 256, 512)),
)
VPBLENDMQ = _instruction(
    "vpblendmq",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x64, (128, 256, 512)),
)
VPBLENDMW = _instruction(
    "vpblendmw",
    _EVEX_MASK_SELECT,
    _OPERANDS_12,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x66, (128, 256, 512)),
)
VPBLENDVB = _instruction(
    "vpblendvb",
    _VEX_BLENDV,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_3, 1, 0, 0x4C, (128, 256)),
)
VPBROADCASTB_FROM_GPR32 = _instruction(
    "vpbroadcastb",
    _UNARY,
    _OPERANDS_9,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x7A, (128, 256, 512)),
)
VPBROADCASTB_VECTOR_UNARY = _instruction(
    "vpbroadcastb",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x78, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x78, (128, 256)),
)
VPBROADCASTD_FROM_GPR32 = _instruction(
    "vpbroadcastd",
    _UNARY,
    _OPERANDS_9,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x7C, (128, 256, 512)),
)
VPBROADCASTD_VECTOR_UNARY = _instruction(
    "vpbroadcastd",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x58, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x58, (128, 256)),
)
VPBROADCASTQ_FROM_GPR64 = _instruction(
    "vpbroadcastq",
    _UNARY,
    _OPERANDS_10,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x7C, (128, 256, 512)),
)
VPBROADCASTQ_VECTOR_UNARY = _instruction(
    "vpbroadcastq",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x59, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x59, (128, 256)),
)
VPBROADCASTW_FROM_GPR32 = _instruction(
    "vpbroadcastw",
    _UNARY,
    _OPERANDS_9,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x7B, (128, 256, 512)),
)
VPBROADCASTW_VECTOR_UNARY = _instruction(
    "vpbroadcastw",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x79, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x79, (128, 256)),
)
VPCMPB = _instruction(
    "vpcmpb",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x3F, (128, 256, 512)),
)
VPCMPD = _instruction(
    "vpcmpd",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x1F, (128, 256, 512)),
)
VPCMPEQB = _instruction(
    "vpcmpeqb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x74, (128, 256)),
)
VPCMPEQD = _instruction(
    "vpcmpeqd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x76, (128, 256)),
)
VPCMPEQQ = _instruction(
    "vpcmpeqq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x29, (128, 256)),
)
VPCMPEQW = _instruction(
    "vpcmpeqw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x75, (128, 256)),
)
VPCMPGTB = _instruction(
    "vpcmpgtb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x64, (128, 256)),
)
VPCMPGTD = _instruction(
    "vpcmpgtd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x66, (128, 256)),
)
VPCMPGTQ = _instruction(
    "vpcmpgtq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x37, (128, 256)),
)
VPCMPGTW = _instruction(
    "vpcmpgtw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x65, (128, 256)),
)
VPCMPQ = _instruction(
    "vpcmpq",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x1F, (128, 256, 512)),
)
VPCMPUB = _instruction(
    "vpcmpub",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x3E, (128, 256, 512)),
)
VPCMPUD = _instruction(
    "vpcmpud",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 0, 0x1E, (128, 256, 512)),
)
VPCMPUQ = _instruction(
    "vpcmpuq",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x1E, (128, 256, 512)),
)
VPCMPUW = _instruction(
    "vpcmpuw",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x3E, (128, 256, 512)),
)
VPCMPW = _instruction(
    "vpcmpw",
    _NDS_IMMEDIATE,
    _OPERANDS_8,
    _IMMEDIATES_4,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x3F, (128, 256, 512)),
)
VPDPBSSD = _instruction(
    "vpdpbssd",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 3, 0, 0x50, (128, 256)),
)
VPDPBSSDS = _instruction(
    "vpdpbssds",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 3, 0, 0x51, (128, 256)),
)
VPDPBSUD = _instruction(
    "vpdpbsud",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 2, 0, 0x50, (128, 256)),
)
VPDPBSUDS = _instruction(
    "vpdpbsuds",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 2, 0, 0x51, (128, 256)),
)
VPDPBUUD = _instruction(
    "vpdpbuud",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 0, 0, 0x50, (128, 256)),
)
VPDPBUUDS = _instruction(
    "vpdpbuuds",
    _DESTRUCTIVE_NDS,
    _OPERANDS_17,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 0, 0, 0x51, (128, 256)),
)
VPERMD = _instruction(
    "vpermd",
    _NDS_SWAPPED_INPUTS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x36, (512,)),
)
VPERMILPD = _instruction(
    "vpermilpd",
    _UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x05, (128,)),
)
VPERMILPS = _instruction(
    "vpermilps",
    _UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x04, (128,)),
)
VPERMPS = _instruction(
    "vpermps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x16, (256,)),
)
VPERMQ_VECTOR_BINARY = _instruction(
    "vpermq",
    _NDS_SWAPPED_INPUTS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x36, (512,)),
)
VPERMQ_VECTOR_UNARY = _instruction(
    "vpermq",
    _UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 1, 0x00, (256,)),
)
VPERMW = _instruction(
    "vpermw",
    _NDS_SWAPPED_INPUTS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x8D, (512,)),
)
VPEXTRB = _instruction(
    "vpextrb",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_2,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x14, (128,)),
)
VPEXTRD = _instruction(
    "vpextrd",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_2,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x16, (128,)),
)
VPEXTRQ = _instruction(
    "vpextrq",
    _REVERSE_UNARY_IMMEDIATE,
    _OPERANDS_4,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 1, 0x16, (128,)),
)
VPEXTRW = _instruction(
    "vpextrw",
    _UNARY_IMMEDIATE,
    _OPERANDS_2,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0xC5, (128,)),
)
VPINSRB = _instruction(
    "vpinsrb",
    _NDS_IMMEDIATE,
    _OPERANDS_14,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x20, (128,)),
)
VPINSRD = _instruction(
    "vpinsrd",
    _NDS_IMMEDIATE,
    _OPERANDS_14,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 0, 0x22, (128,)),
)
VPINSRQ = _instruction(
    "vpinsrq",
    _NDS_IMMEDIATE,
    _OPERANDS_15,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_3, 1, 1, 0x22, (128,)),
)
VPINSRW = _instruction(
    "vpinsrw",
    _NDS_IMMEDIATE,
    _OPERANDS_14,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0xC4, (128,)),
)
VPMAXSB = _instruction(
    "vpmaxsb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3C, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3C, (128, 256)),
)
VPMAXSD = _instruction(
    "vpmaxsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3D, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3D, (128, 256)),
)
VPMAXSQ = _instruction(
    "vpmaxsq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x3D, (128, 256, 512)),
)
VPMAXSW = _instruction(
    "vpmaxsw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xEE, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xEE, (128, 256)),
)
VPMAXUB = _instruction(
    "vpmaxub",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xDE, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xDE, (128, 256)),
)
VPMAXUD = _instruction(
    "vpmaxud",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3F, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3F, (128, 256)),
)
VPMAXUQ = _instruction(
    "vpmaxuq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x3F, (128, 256, 512)),
)
VPMAXUW = _instruction(
    "vpmaxuw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3E, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3E, (128, 256)),
)
VPMINSB = _instruction(
    "vpminsb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x38, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x38, (128, 256)),
)
VPMINSD = _instruction(
    "vpminsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x39, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x39, (128, 256)),
)
VPMINSQ = _instruction(
    "vpminsq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x39, (128, 256, 512)),
)
VPMINSW = _instruction(
    "vpminsw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xEA, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xEA, (128, 256)),
)
VPMINUB = _instruction(
    "vpminub",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xDA, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xDA, (128, 256)),
)
VPMINUD = _instruction(
    "vpminud",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3B, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3B, (128, 256)),
)
VPMINUQ = _instruction(
    "vpminuq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x3B, (128, 256, 512)),
)
VPMINUW = _instruction(
    "vpminuw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x3A, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x3A, (128, 256)),
)
VPMOVB2M = _instruction(
    "vpmovb2m",
    _UNARY,
    _OPERANDS_7,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 0, 0x29, (128, 256, 512)),
)
VPMOVD2M = _instruction(
    "vpmovd2m",
    _UNARY,
    _OPERANDS_7,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 0, 0x39, (128,)),
)
VPMOVM2B = _instruction(
    "vpmovm2b",
    _UNARY,
    _OPERANDS_11,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 0, 0x28, (128, 256, 512)),
)
VPMOVM2D = _instruction(
    "vpmovm2d",
    _UNARY,
    _OPERANDS_11,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 0, 0x38, (128,)),
)
VPMOVM2Q = _instruction(
    "vpmovm2q",
    _UNARY,
    _OPERANDS_11,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 1, 0x38, (128,)),
)
VPMOVM2W = _instruction(
    "vpmovm2w",
    _UNARY,
    _OPERANDS_11,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 1, 0x28, (128,)),
)
VPMOVQ2M = _instruction(
    "vpmovq2m",
    _UNARY,
    _OPERANDS_7,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 1, 0x39, (128,)),
)
VPMOVSXBW = _instruction(
    "vpmovsxbw",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x20, (256,)),
)
VPMOVSXDQ = _instruction(
    "vpmovsxdq",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x25, (256,)),
)
VPMOVSXWD = _instruction(
    "vpmovsxwd",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x23, (256,)),
)
VPMOVW2M = _instruction(
    "vpmovw2m",
    _UNARY,
    _OPERANDS_7,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 1, 0x29, (128,)),
)
VPMOVWB = _instruction(
    "vpmovwb",
    _REVERSE_UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 2, 0, 0x30, (512,)),
)
VPMOVZXBD = _instruction(
    "vpmovzxbd",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x31, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x31, (128, 256)),
)
VPMOVZXBQ = _instruction(
    "vpmovzxbq",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x32, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x32, (128, 256)),
)
VPMOVZXBW = _instruction(
    "vpmovzxbw",
    _UNARY,
    _OPERANDS_13,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x30, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x30, (128, 256)),
)
VPMULLD = _instruction(
    "vpmulld",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x40, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x40, (128, 256)),
)
VPMULLQ = _instruction(
    "vpmullq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x40, (128, 256, 512)),
)
VPMULLW = _instruction(
    "vpmullw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xD5, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xD5, (128, 256)),
)
VPOR = _instruction(
    "vpor",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xEB, (128, 256)),
)
VPORD = _instruction(
    "vpord",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xEB, (512,)),
)
VPSHUFB = _instruction(
    "vpshufb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_2, 1, 0, 0x00, (128, 256)),
)
VPSHUFD = _instruction(
    "vpshufd",
    _UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x70, (128,)),
)
VPSHUFLW = _instruction(
    "vpshuflw",
    _UNARY_IMMEDIATE,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 3, 0, 0x70, (128,)),
)
VPSLLD = _instruction(
    "vpslld",
    _SHIFT_IMMEDIATE_6,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x72, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 0, 0x72, (512,)),
)
VPSLLD_COUNT = _instruction(
    "vpslld",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xF2, (128, 256)),
)
VPSLLQ = _instruction(
    "vpsllq",
    _SHIFT_IMMEDIATE_6,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x73, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 1, 0x73, (512,)),
)
VPSLLQ_COUNT = _instruction(
    "vpsllq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xF3, (128, 256)),
)
VPSLLVD = _instruction(
    "vpsllvd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x47, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x47, (128, 256)),
)
VPSLLVQ = _instruction(
    "vpsllvq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x47, (512,)),
    _encoding(_VEX, _MAP_2, 1, 1, 0x47, (128, 256)),
)
VPSLLVW = _instruction(
    "vpsllvw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x12, (128, 256, 512)),
)
VPSLLW = _instruction(
    "vpsllw",
    _SHIFT_IMMEDIATE_6,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_1, 1, 0, 0x71, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x71, (128, 256)),
)
VPSLLW_COUNT = _instruction(
    "vpsllw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xF1, (128, 256)),
)
VPSRAD = _instruction(
    "vpsrad",
    _SHIFT_IMMEDIATE_4,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x72, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 0, 0x72, (512,)),
)
VPSRAD_COUNT = _instruction(
    "vpsrad",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xE2, (128, 256)),
)
VPSRAQ = _instruction(
    "vpsraq",
    _SHIFT_IMMEDIATE_4,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x72, (128, 256, 512)),
)
VPSRAW = _instruction(
    "vpsraw",
    _SHIFT_IMMEDIATE_4,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x71, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 0, 0x71, (512,)),
)
VPSRAW_COUNT = _instruction(
    "vpsraw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xE1, (128, 256)),
)
VPSRAVD = _instruction(
    "vpsravd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x46, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x46, (128, 256)),
)
VPSRAVQ = _instruction(
    "vpsravq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x46, (128, 256, 512)),
)
VPSRAVW = _instruction(
    "vpsravw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x11, (128, 256, 512)),
)
VPSRLD = _instruction(
    "vpsrld",
    _SHIFT_IMMEDIATE_2,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x72, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 0, 0x72, (512,)),
)
VPSRLD_COUNT = _instruction(
    "vpsrld",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xD2, (128, 256)),
)
VPSRLDQ = _instruction(
    "vpsrldq",
    _SHIFT_IMMEDIATE_3,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_1, 1, 0, 0x73, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x73, (128,)),
)
VPSRLQ = _instruction(
    "vpsrlq",
    _SHIFT_IMMEDIATE_2,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0x73, (128, 256)),
    _encoding(_EVEX, _MAP_1, 1, 1, 0x73, (512,)),
)
VPSRLQ_COUNT = _instruction(
    "vpsrlq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xD3, (128, 256)),
)
VPSRLVD = _instruction(
    "vpsrlvd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 0, 0x45, (512,)),
    _encoding(_VEX, _MAP_2, 1, 0, 0x45, (128, 256)),
)
VPSRLVQ = _instruction(
    "vpsrlvq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x45, (512,)),
    _encoding(_VEX, _MAP_2, 1, 1, 0x45, (128, 256)),
)
VPSRLVW = _instruction(
    "vpsrlvw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_2, 1, 1, 0x10, (128, 256, 512)),
)
VPSRLW = _instruction(
    "vpsrlw",
    _SHIFT_IMMEDIATE_2,
    _OPERANDS_13,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_1, 1, 0, 0x71, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x71, (128, 256)),
)
VPSRLW_COUNT = _instruction(
    "vpsrlw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xD1, (128, 256)),
)
VPSUBB = _instruction(
    "vpsubb",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xF8, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xF8, (128, 256)),
)
VPSUBD = _instruction(
    "vpsubd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xFA, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xFA, (128, 256)),
)
VPSUBQ = _instruction(
    "vpsubq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0xFB, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xFB, (128, 256)),
)
VPSUBW = _instruction(
    "vpsubw",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xF9, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0xF9, (128, 256)),
)
VPUNPCKLQDQ = _instruction(
    "vpunpcklqdq",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0x6C, (128,)),
)
VPXOR = _instruction(
    "vpxor",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 1, 0, 0xEF, (128, 256)),
)
VPXORD = _instruction(
    "vpxord",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 0, 0xEF, (512,)),
)
VSHUFI64X2 = _instruction(
    "vshufi64x2",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_EVEX, _MAP_3, 1, 1, 0x43, (512,)),
)
VSHUFPD = _instruction(
    "vshufpd",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 1, 0, 0xC6, (128,)),
)
VSHUFPS = _instruction(
    "vshufps",
    _NDS_IMMEDIATE,
    _OPERANDS_16,
    _IMMEDIATES_5,
    _encoding(_VEX, _MAP_1, 0, 0, 0xC6, (128,)),
)
VSUBPD = _instruction(
    "vsubpd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 1, 1, 0x5C, (512,)),
    _encoding(_VEX, _MAP_1, 1, 0, 0x5C, (128, 256)),
)
VSUBPS = _instruction(
    "vsubps",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x5C, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x5C, (128, 256)),
)
VSUBSD = _instruction(
    "vsubsd",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 3, 0, 0x5C, (128,)),
)
VSUBSS = _instruction(
    "vsubss",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_VEX, _MAP_1, 2, 0, 0x5C, (128,)),
)
VSUBPH = _instruction(
    "vsubph",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 0, 0, 0x5C, (128, 256, 512)),
)
VSUBSH = _instruction(
    "vsubsh",
    _NDS,
    _OPERANDS_16,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_5, 2, 0, 0x5C, (128,)),
)
VXORPS = _instruction(
    "vxorps",
    _ZERO,
    _OPERANDS_20,
    _IMMEDIATES_0,
    _encoding(_EVEX, _MAP_1, 0, 0, 0x57, (512,)),
    _encoding(_VEX, _MAP_1, 0, 0, 0x57, (128, 256)),
)

# Family-aligned facts are consumed with strict zip at descriptor construction.
# The binding validates the mnemonic, operand/immediate shape, prefix, and width.
AVX2_UNIFORM_SHIFT_IMMEDIATE = (
    VPSLLW,
    VPSLLD,
    VPSLLQ,
    VPSRAW,
    VPSRAD,
    VPSRLW,
    VPSRLD,
    VPSRLQ,
)
AVX2_UNIFORM_SHIFT_COUNT = (
    VPSLLW_COUNT,
    VPSLLD_COUNT,
    VPSLLQ_COUNT,
    VPSRAW_COUNT,
    VPSRAD_COUNT,
    VPSRLW_COUNT,
    VPSRLD_COUNT,
    VPSRLQ_COUNT,
)
AVX512_UNIFORM_SHIFT_IMMEDIATE = (*AVX2_UNIFORM_SHIFT_IMMEDIATE, VPSRAQ)

AVX2_INTEGER_BINARY = (
    VPADDB,
    VPADDW,
    VPADDD,
    VPADDQ,
    VPSUBB,
    VPSUBW,
    VPSUBD,
    VPSUBQ,
    VPMULLW,
    VPMULLD,
    VPMINSB,
    VPMINSW,
    VPMINSD,
    VPMAXSB,
    VPMAXSW,
    VPMAXSD,
    VPMINUB,
    VPMINUW,
    VPMINUD,
    VPMAXUB,
    VPMAXUW,
    VPMAXUD,
    VPSLLVD,
    VPSLLVQ,
    VPSRAVD,
    VPSRLVD,
    VPSRLVQ,
)
AVX512VL_INTEGER_BINARY = (
    VPMULLQ,
    VPMINSQ,
    VPMAXSQ,
    VPMINUQ,
    VPMAXUQ,
    VPSLLVW,
    VPSRAVW,
    VPSRLVW,
    VPSRAVQ,
)
AVX512_INTEGER_BINARY = (*AVX2_INTEGER_BINARY, *AVX512VL_INTEGER_BINARY)
AVX2_FLOAT_BINARY = (
    VADDPS,
    VADDPD,
    VSUBPS,
    VSUBPD,
    VMULPS,
    VMULPD,
    VDIVPS,
    VDIVPD,
)
AVX2_SCALAR_FLOAT_BINARY = (
    VADDSS,
    VADDSD,
    VSUBSS,
    VSUBSD,
    VMULSS,
    VMULSD,
    VDIVSS,
    VDIVSD,
)
AVX2_BITWISE = (VPAND, VPOR, VPXOR)
AVX512_BITWISE = (VPANDD, VPORD, VPXORD)
AVX2_LANE = (
    (VPEXTRB, VPINSRB),
    (VPEXTRW, VPINSRW),
    (VPEXTRD, VPINSRD),
    (VPEXTRQ, VPINSRQ),
)
AVX2_INTEGER_COMPARE = (
    (VPCMPEQB, VPCMPGTB),
    (VPCMPEQW, VPCMPGTW),
    (VPCMPEQD, VPCMPGTD),
    (VPCMPEQQ, VPCMPGTQ),
)
AVX512_INTEGER_COMPARE = (
    (VPCMPB, VPCMPUB),
    (VPCMPW, VPCMPUW),
    (VPCMPD, VPCMPUD),
    (VPCMPQ, VPCMPUQ),
)
AVX2_FLOAT_COMPARE = (VCMPPS_VECTOR_BINARY, VCMPPD_VECTOR_BINARY)
AVX512_FLOAT_COMPARE = (VCMPPS_MASK_COMPARE, VCMPPD_MASK_COMPARE)
AVX512_FP16_FLOAT_BINARY = (VADDPH, VSUBPH, VMULPH, VDIVPH)
AVX512_FP16_SCALAR_FLOAT_BINARY = (VADDSH, VSUBSH, VMULSH, VDIVSH)
AVX512_FP16_FLOAT_EXTREMA = (VMINPH, VMAXPH)
AVX512_FP16_SCALAR_FLOAT_EXTREMA = (VMINSH, VMAXSH)
AVX512_FP16_FLOAT_FMA = VFMADD231PH
AVX512_FP16_SCALAR_FLOAT_FMA = VFMADD231SH
AVX512_FP16_FLOAT_COMPARE = VCMPPH
AVX512_FP16_SCALAR_FLOAT_COMPARE = VCMPSH
AVX512_FP16_CONVERSIONS = (
    VCVTPH2PSX,
    VCVTPS2PHX,
    VCVTSH2SS,
    VCVTSS2SH,
)
AVX_NE_CONVERT_MEMORY = (
    VBCSTNEBF162PS_MEMORY,
    VBCSTNESH2PS_MEMORY,
    VCVTNEEBF162PS_MEMORY,
    VCVTNEEPH2PS_MEMORY,
    VCVTNEOBF162PS_MEMORY,
    VCVTNEOPH2PS_MEMORY,
)
AVX2_FLOAT_EXTREMA = (VMINPS, VMAXPS, VMINPD, VMAXPD)
AVX2_SCALAR_FLOAT_EXTREMA = (VMINSS, VMAXSS, VMINSD, VMAXSD)
AVX2_FLOAT_FMA = (VFMADD231PS, VFMADD231PD)
AVX2_SCALAR_FLOAT_FMA = (VFMADD231SS, VFMADD231SD)
AVX2_SIMD_BROADCAST = (
    VPBROADCASTB_VECTOR_UNARY,
    VPBROADCASTW_VECTOR_UNARY,
    VPBROADCASTQ_VECTOR_UNARY,
)
AVX512_GPR_BROADCAST = (
    VPBROADCASTB_FROM_GPR32,
    VPBROADCASTW_FROM_GPR32,
    VPBROADCASTD_FROM_GPR32,
    VPBROADCASTQ_FROM_GPR64,
)
AVX512_SELECT = (VPBLENDMB, VPBLENDMW, VPBLENDMD, VPBLENDMQ, VBLENDMPS, VBLENDMPD)
AVX512_PREDICATE_CONVERSION = (
    (VPMOVQ2M, VPMOVM2Q),
    (VPMOVD2M, VPMOVM2D),
    (VPMOVW2M, VPMOVM2W),
    (VPMOVB2M, VPMOVM2B),
    (VPMOVB2M, VPMOVM2B),
    (VPMOVB2M, VPMOVM2B),
)
AVX_VNNI_INT8 = (
    VPDPBSSD,
    VPDPBSSDS,
    VPDPBSUD,
    VPDPBSUDS,
    VPDPBUUD,
    VPDPBUUDS,
)
VECTOR_MEMORY = (
    VMOVDQU32_LOAD,
    VMOVDQU32_INDEXED_LOAD,
    VMOVDQU32_STORE_SOURCE,
    VMOVDQU32_INDEXED_STORE_SOURCE,
)
XMM_QWORD_MEMORY = (
    VMOVSD_LOAD,
    VMOVSD_INDEXED_LOAD,
    VMOVSD_STORE_SOURCE,
    VMOVSD_INDEXED_STORE_SOURCE,
)
