# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact native-encoding identities shared by x86 descriptor views."""

from __future__ import annotations

from enum import IntEnum


class X86EncodingFormat(IntEnum):
    """Target-owned expansion recipes stored in Descriptor.encoding_format_id."""

    NONE = 0
    RM_REG = 1
    REG_RM = 2
    REG_RM_IMM32 = 3
    RM_IMM8 = 4
    SELECT = 5
    COMPARE = 6
    MOV_IMMEDIATE = 7
    MEMORY_REG_RM = 8
    MEMORY_RM_REG = 9
    LEA = 10
    DIRECT_BRANCH = 11
    RM_IMM32 = 12
    RM_GROUP = 13
    CONDITIONAL_SUBTRACT = 14


# Scalar legacy opcodes occupy at most two bytes and every two-byte opcode in
# the initial surface begins with 0x0F. The high bit remains available for the
# one width override needed by a 32-bit write whose semantic result is GPR64.
X86_ENCODING_ID_FORCE_32_BIT = 1 << 15
X86_ENCODING_ID_OPCODE_MASK = X86_ENCODING_ID_FORCE_32_BIT - 1


def x86_legacy_opcode(*opcode_bytes: int, force_32_bit: bool = False) -> int:
    """Packs a one- or 0x0F-prefixed two-byte scalar opcode."""

    if len(opcode_bytes) == 1:
        encoding_id = opcode_bytes[0]
    elif len(opcode_bytes) == 2 and opcode_bytes[0] == 0x0F:
        encoding_id = (opcode_bytes[0] << 8) | opcode_bytes[1]
    else:
        raise ValueError("x86 scalar legacy opcode must have one byte or a 0x0F prefix")
    if any(opcode_byte < 0 or opcode_byte > 0xFF for opcode_byte in opcode_bytes):
        raise ValueError("x86 opcode bytes must fit in u8")
    if force_32_bit:
        encoding_id |= X86_ENCODING_ID_FORCE_32_BIT
    return encoding_id


def x86_modrm_group_opcode(opcode: int, extension: int) -> int:
    """Packs a one-byte opcode and its three-bit ModRM opcode extension."""

    if opcode < 0 or opcode > 0xFF:
        raise ValueError("x86 group opcode must fit in u8")
    if extension < 0 or extension > 7:
        raise ValueError("x86 ModRM group extension must fit in three bits")
    return opcode | (extension << 8)


__all__ = [
    "X86EncodingFormat",
    "X86_ENCODING_ID_FORCE_32_BIT",
    "X86_ENCODING_ID_OPCODE_MASK",
    "x86_legacy_opcode",
    "x86_modrm_group_opcode",
]
