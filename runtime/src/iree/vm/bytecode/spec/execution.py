# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime executable-subset and verification-form declarations.

The ISA specification describes every legal instruction in a version. During
the incremental runtime bring-up, this table separately names the instructions
that the interpreter can execute and the handwritten verification form that
owns their module-load checks.
"""

from __future__ import annotations

import dataclasses
import re

_FORM_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutableInstruction:
    """One instruction implemented by the current interpreter."""

    mnemonic: str
    verification_form: str

    def __post_init__(self) -> None:
        if not _FORM_PATTERN.fullmatch(self.verification_form):
            raise ValueError(f"invalid verification form {self.verification_form!r}")


EXECUTABLE_INSTRUCTIONS = (
    ExecutableInstruction("control.block", "CONTROL_BLOCK"),
    ExecutableInstruction("control.return", "CONTROL_RETURN"),
    ExecutableInstruction("constant.zero", "CONSTANT_ZERO"),
    ExecutableInstruction("constant.s16", "CONSTANT_S16"),
    ExecutableInstruction("constant.i32", "CONSTANT_I32"),
    ExecutableInstruction("constant.i64", "CONSTANT_I64"),
    ExecutableInstruction("constant.pool.load.i32", "CONSTANT_POOL_LOAD_I32"),
    ExecutableInstruction("constant.pool.load.i64", "CONSTANT_POOL_LOAD_I64"),
    ExecutableInstruction("value.copy", "VALUE_UNARY_4"),
    ExecutableInstruction("value.select", "VALUE_SELECT"),
    ExecutableInstruction("global.value.immutable.load", "GLOBAL_VALUE_IMMUTABLE_LOAD"),
    ExecutableInstruction(
        "global.value.immutable.store", "GLOBAL_VALUE_IMMUTABLE_STORE"
    ),
    ExecutableInstruction("global.value.mutable.load", "GLOBAL_VALUE_MUTABLE_LOAD"),
    ExecutableInstruction("global.value.mutable.store", "GLOBAL_VALUE_MUTABLE_STORE"),
    ExecutableInstruction("integer.add.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.add.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.sub.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.sub.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.mul.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.mul.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.neg.i32", "VALUE_UNARY_4"),
    ExecutableInstruction("integer.neg.i64", "VALUE_UNARY_4"),
    ExecutableInstruction("integer.abs.s32", "VALUE_UNARY_4"),
    ExecutableInstruction("integer.abs.s64", "VALUE_UNARY_4"),
    ExecutableInstruction("integer.min.s32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.min.s64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.min.u32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.min.u64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.max.s32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.max.s64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.max.u32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.max.u64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.and.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.and.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.or.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.or.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.xor.i32", "VALUE_BINARY_4"),
    ExecutableInstruction("integer.xor.i64", "VALUE_BINARY_4"),
    ExecutableInstruction("buffer.rodata.load", "BUFFER_RODATA_LOAD"),
    ExecutableInstruction("conversion.integer", "CONVERSION_INTEGER"),
    ExecutableInstruction("conversion.float.extend", "CONVERSION_FLOAT_EXTEND"),
    ExecutableInstruction("conversion.float.to.integer", "CONVERSION_FLOAT_TO_INTEGER"),
)
