# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native encoding facts consumed by target/emit/native/x86/encoding.h."""

from enum import IntEnum, IntFlag

from loom.target.low_descriptors import Descriptor, OperandRole


class Form(IntEnum):
    BINARY_RM_R = 1
    BINARY_R_RM = 2
    MULTIPLY_HIGH = 3
    MULTIPLY_IMMEDIATE = 4
    BINARY_IMMEDIATE = 5
    SHIFT_IMMEDIATE = 6
    SHIFT_COUNT = 7
    MOVE = 8
    SELECT = 9
    SUBTRACT_IF_UGE = 10
    COMPARE = 11
    COMPARE_IMMEDIATE = 12
    CONSTANT = 13
    LOAD = 14
    STORE = 15
    ADDRESS_ADD = 16
    ADDRESS_DISPLACEMENT = 17
    ADDRESS_SCALE = 18
    ADDRESS_ADD_SCALE = 19


class Flag(IntFlag):
    REX_W = 1 << 12
    OPERAND_16 = 1 << 13
    BYTE = 1 << 14
    INDEXED = 1 << 15


def encoding(opcode: int, *, extension: int = 0, flags: int = 0) -> int:
    """Packs immutable native instruction fields into a descriptor encoding ID."""
    if (
        not (0 < opcode <= 0xFF or 0x0F00 <= opcode <= 0x0FFF)
        or not 0 <= extension <= 7
    ):
        raise ValueError("invalid native opcode or ModRM extension")
    if flags & ((1 << 12) - 1):
        raise ValueError("encoding flags overlap opcode fields")
    return (opcode & 0xFF) | ((opcode > 0xFF) << 8) | (extension << 9) | flags


def validate_descriptor_encoding(descriptor: Descriptor) -> None:
    """Establishes the fixed native operand carrier bounds during generation."""
    if not descriptor.encoding_format_id:
        return
    Form(descriptor.encoding_format_id)
    inputs = sum(
        o.role
        in (
            OperandRole.OPERAND,
            OperandRole.OPERAND_RESULT,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        )
        for o in descriptor.operands
    )
    results = sum(
        o.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT)
        for o in descriptor.operands
    )
    if inputs > 3 or results > 1 or len(descriptor.immediates) > 2:
        raise ValueError(f"{descriptor.key}: native operand carrier overflow")
    if len(descriptor.immediates) == 2 and tuple(
        immediate.field_name for immediate in descriptor.immediates
    ) != ("disp32", "scale"):
        raise ValueError(f"{descriptor.key}: native addressing requires disp32, scale")
