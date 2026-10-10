# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native encoding facts consumed by target/emit/native/x86/encoding.h."""

from enum import IntEnum, IntFlag

from loom.target.arch.x86.vector_encoding import (
    VECTOR_ENCODING_FORMAT_MARKER,
    validate_vector_encoding_recipe,
    vector_encoding_address_base_input,
)
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
    # Values 20..26 are structural control and ABI forms in the C encoder.
    ADDRESS_PC_RELATIVE = 27


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
    is_vector = bool(descriptor.encoding_format_id & VECTOR_ENCODING_FORMAT_MARKER)
    if is_vector:
        try:
            validate_vector_encoding_recipe(descriptor.encoding_format_id)
        except ValueError as error:
            raise ValueError(f"{descriptor.key}: {error}") from error
        if not descriptor.encoding_id:
            raise ValueError(f"{descriptor.key}: invalid native vector encoding")
        # Map zero denotes the compact EVEX map-5/map-6 states. The typed
        # VectorEncoding constructor has already rejected every unsupported
        # architectural map before producing this packed ID.
    else:
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
    if (
        inputs > (5 if is_vector else 3)
        or results > 1
        or len(descriptor.immediates) > 2
    ):
        raise ValueError(f"{descriptor.key}: native operand carrier overflow")
    if len(descriptor.immediates) == 2 and tuple(
        immediate.field_name for immediate in descriptor.immediates
    ) != ("disp32", "scale"):
        raise ValueError(f"{descriptor.key}: native addressing requires disp32, scale")
    if is_vector:
        try:
            base_index = vector_encoding_address_base_input(
                descriptor.encoding_format_id
            )
        except ValueError as error:
            raise ValueError(f"{descriptor.key}: {error}") from error
        if base_index is not None:
            input_operands = tuple(
                operand
                for operand in descriptor.operands
                if operand.role
                in (
                    OperandRole.OPERAND,
                    OperandRole.OPERAND_RESULT,
                    OperandRole.PREDICATE,
                    OperandRole.RESOURCE,
                )
            )
            address_count = 2 if len(descriptor.immediates) == 2 else 1
            address_operands = input_operands[base_index : base_index + address_count]
            if len(address_operands) != address_count or any(
                operand.role != OperandRole.RESOURCE
                or {alternative.reg_class for alternative in operand.reg_alts}
                != {"x86.gpr64"}
                for operand in address_operands
            ):
                raise ValueError(
                    f"{descriptor.key}: vector memory address must use adjacent "
                    "GPR64 resources"
                )
