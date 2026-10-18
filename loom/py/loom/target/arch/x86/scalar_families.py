# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable x86 scalar-family rows shared by descriptor and contract generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarReadMemoryFamily:
    """An integer operation with a destructive GPR/read-memory encoding."""

    source_operation: str
    descriptor_operation: str
    mnemonic: str
    semantic: str
    opcode: int
    commutative: bool = True


@dataclass(frozen=True, slots=True)
class ScalarVariableShiftFamily:
    """A source shift and its non-destructive BMI2 machine form."""

    source_operation: str
    descriptor_operation: str
    semantic: str
    mandatory_prefix: int


@dataclass(frozen=True, slots=True)
class ScalarRotationFamily:
    """A source rotation and its legacy destructive machine form."""

    source_operation: str
    mnemonic: str
    semantic: str
    opcode_extension: int


X86_SCALAR_READ_MEMORY_FAMILIES = (
    ScalarReadMemoryFamily("addi", "add", "add", "add", 0x03),
    ScalarReadMemoryFamily("subi", "sub", "sub", "sub", 0x2B, False),
    ScalarReadMemoryFamily("muli", "imul", "imul", "mul", 0x0FAF),
    ScalarReadMemoryFamily("andi", "and", "and", "and", 0x23),
    ScalarReadMemoryFamily("ori", "or", "or", "or", 0x0B),
    ScalarReadMemoryFamily("xori", "xor", "xor", "xor", 0x33),
)

X86_SCALAR_VARIABLE_SHIFT_FAMILIES = (
    ScalarVariableShiftFamily("shli", "shlx", "shl", 1),
    ScalarVariableShiftFamily("shrsi", "sarx", "shrs", 2),
    ScalarVariableShiftFamily("shrui", "shrx", "shru", 3),
)

X86_SCALAR_ROTATION_FAMILIES = (
    ScalarRotationFamily("rotli", "rol", "rotl", 0),
    ScalarRotationFamily("rotri", "ror", "rotr", 1),
)


def validate_scalar_families() -> None:
    if {
        (family.source_operation, family.semantic, family.opcode_extension)
        for family in X86_SCALAR_ROTATION_FAMILIES
    } != {("rotli", "rotl", 0), ("rotri", "rotr", 1)}:
        raise ValueError("x86 scalar rotation rows must cover left and right")


validate_scalar_families()
