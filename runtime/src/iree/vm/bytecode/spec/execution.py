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
    ExecutableInstruction("control.yield.s32", "CONTROL_YIELD_S32"),
    ExecutableInstruction("control.branch.s16", "CONTROL_BRANCH_S16"),
    ExecutableInstruction("control.branch.s32", "CONTROL_BRANCH_S32"),
    ExecutableInstruction("control.branch.if.s16", "CONTROL_BRANCH_CONDITIONAL_S16"),
    ExecutableInstruction("control.branch.if.s32", "CONTROL_BRANCH_CONDITIONAL_S32"),
    ExecutableInstruction(
        "control.branch.unless.s16", "CONTROL_BRANCH_CONDITIONAL_S16"
    ),
    ExecutableInstruction(
        "control.branch.unless.s32", "CONTROL_BRANCH_CONDITIONAL_S32"
    ),
    ExecutableInstruction("control.switch", "CONTROL_SWITCH"),
    ExecutableInstruction("control.assert", "CONTROL_ASSERT"),
    ExecutableInstruction("control.fail", "CONTROL_FAIL"),
    ExecutableInstruction("value.abi.argument.load", "VALUE_ABI_ARGUMENT_LOAD"),
    ExecutableInstruction("value.abi.result.store", "VALUE_ABI_RESULT_STORE"),
    ExecutableInstruction("ref.abi.argument.load.borrow", "REF_ABI_ARGUMENT_LOAD"),
    ExecutableInstruction("ref.abi.argument.load.move", "REF_ABI_ARGUMENT_LOAD"),
    ExecutableInstruction("ref.abi.result.store.move", "REF_ABI_RESULT_STORE"),
    ExecutableInstruction("func.abi.argument.load", "FUNC_ABI_ARGUMENT_LOAD"),
    ExecutableInstruction("func.abi.result.store", "FUNC_ABI_RESULT_STORE"),
    ExecutableInstruction("constant.zero", "CONSTANT_ZERO"),
    ExecutableInstruction("constant.s16", "CONSTANT_S16"),
    ExecutableInstruction("constant.i32", "CONSTANT_I32"),
    ExecutableInstruction("constant.i64", "CONSTANT_I64"),
    ExecutableInstruction("constant.pool.load.i32", "CONSTANT_POOL_LOAD_I32"),
    ExecutableInstruction("constant.pool.load.i64", "CONSTANT_POOL_LOAD_I64"),
    ExecutableInstruction("value.copy", "VALUE_UNARY_4"),
    ExecutableInstruction("value.select", "VALUE_SELECT"),
    ExecutableInstruction("func.null", "FUNC_NULL"),
    ExecutableInstruction("func.compare.null", "FUNC_COMPARE_NULL"),
    ExecutableInstruction("func.copy", "FUNC_COPY"),
    ExecutableInstruction("func.address", "FUNC_ADDRESS"),
    ExecutableInstruction("func.import.resolved", "FUNC_IMPORT_RESOLVED"),
    ExecutableInstruction("func.stack.load", "FUNC_STACK_TRANSFER"),
    ExecutableInstruction("func.stack.store", "FUNC_STACK_TRANSFER"),
    ExecutableInstruction("global.value.immutable.load", "GLOBAL_VALUE_IMMUTABLE_LOAD"),
    ExecutableInstruction(
        "global.value.immutable.store", "GLOBAL_VALUE_IMMUTABLE_STORE"
    ),
    ExecutableInstruction("global.value.mutable.load", "GLOBAL_VALUE_MUTABLE_LOAD"),
    ExecutableInstruction("global.value.mutable.store", "GLOBAL_VALUE_MUTABLE_STORE"),
    ExecutableInstruction(
        "global.ref.immutable.load.borrow", "GLOBAL_REF_IMMUTABLE_LOAD"
    ),
    ExecutableInstruction(
        "global.ref.immutable.store.move", "GLOBAL_REF_IMMUTABLE_STORE"
    ),
    ExecutableInstruction("global.ref.mutable.load.retain", "GLOBAL_REF_MUTABLE_LOAD"),
    ExecutableInstruction("global.ref.mutable.store.move", "GLOBAL_REF_MUTABLE_STORE"),
    ExecutableInstruction("global.func.immutable.load", "GLOBAL_FUNC_IMMUTABLE_LOAD"),
    ExecutableInstruction("global.func.immutable.store", "GLOBAL_FUNC_IMMUTABLE_STORE"),
    ExecutableInstruction("global.func.mutable.load", "GLOBAL_FUNC_MUTABLE_LOAD"),
    ExecutableInstruction("global.func.mutable.store", "GLOBAL_FUNC_MUTABLE_STORE"),
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
    ExecutableInstruction("integer.compare.i32", "INTEGER_COMPARE"),
    ExecutableInstruction("integer.compare.i64", "INTEGER_COMPARE"),
    ExecutableInstruction("integer.lea.i32", "INTEGER_LEA"),
    ExecutableInstruction("integer.lea.i64", "INTEGER_LEA"),
    ExecutableInstruction("integer.ceildiv.pow2.u32", "INTEGER_CEILDIV_POW2_U32"),
    ExecutableInstruction("integer.ceildiv.pow2.u64", "INTEGER_CEILDIV_POW2_U64"),
    ExecutableInstruction("float.add.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.add.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.sub.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.sub.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.mul.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.mul.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.div.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.div.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.rem.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.rem.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.neg.f32", "VALUE_UNARY_4"),
    ExecutableInstruction("float.neg.f64", "VALUE_UNARY_4"),
    ExecutableInstruction("float.abs.f32", "VALUE_UNARY_4"),
    ExecutableInstruction("float.abs.f64", "VALUE_UNARY_4"),
    ExecutableInstruction("float.minmax.f32", "FLOAT_MINMAX"),
    ExecutableInstruction("float.minmax.f64", "FLOAT_MINMAX"),
    ExecutableInstruction("float.compare.f32", "FLOAT_COMPARE"),
    ExecutableInstruction("float.compare.f64", "FLOAT_COMPARE"),
    ExecutableInstruction("float.classify.f32", "FLOAT_CLASSIFY"),
    ExecutableInstruction("float.classify.f64", "FLOAT_CLASSIFY"),
    ExecutableInstruction("float.clamp.f32", "FLOAT_CLAMP"),
    ExecutableInstruction("float.clamp.f64", "FLOAT_CLAMP"),
    ExecutableInstruction("float.copysign.f32", "VALUE_BINARY_4"),
    ExecutableInstruction("float.copysign.f64", "VALUE_BINARY_4"),
    ExecutableInstruction("float.math.unary.f32", "FLOAT_MATH_UNARY"),
    ExecutableInstruction("float.math.unary.f64", "FLOAT_MATH_UNARY"),
    ExecutableInstruction("float.math.binary.f32", "FLOAT_MATH_BINARY"),
    ExecutableInstruction("float.math.binary.f64", "FLOAT_MATH_BINARY"),
    ExecutableInstruction("float.math.ternary.f32", "FLOAT_MATH_TERNARY"),
    ExecutableInstruction("float.math.ternary.f64", "FLOAT_MATH_TERNARY"),
    ExecutableInstruction("ref.null", "REF_CLEAR"),
    ExecutableInstruction("ref.compare.null", "REF_COMPARE_NULL"),
    ExecutableInstruction("ref.compare.eq", "REF_COMPARE_EQ"),
    ExecutableInstruction("ref.retain", "REF_RETAIN"),
    ExecutableInstruction("ref.move", "REF_MOVE"),
    ExecutableInstruction("ref.discard", "REF_CLEAR"),
    ExecutableInstruction("ref.stack.load.retain", "REF_STACK_TRANSFER"),
    ExecutableInstruction("ref.stack.load.move", "REF_STACK_TRANSFER"),
    ExecutableInstruction("ref.stack.store.retain", "REF_STACK_TRANSFER"),
    ExecutableInstruction("ref.stack.store.move", "REF_STACK_TRANSFER"),
    ExecutableInstruction("ref.stack.discard", "REF_STACK_DISCARD"),
    ExecutableInstruction("buffer.rodata.load", "BUFFER_RODATA_LOAD"),
    ExecutableInstruction("conversion.integer", "CONVERSION_INTEGER"),
    ExecutableInstruction("conversion.float.extend", "CONVERSION_FLOAT_EXTEND"),
    ExecutableInstruction("conversion.float.to.integer", "CONVERSION_FLOAT_TO_INTEGER"),
    ExecutableInstruction("stack.load", "STACK_LOAD"),
    ExecutableInstruction("stack.store", "STACK_STORE"),
    ExecutableInstruction("stack.load.indexed", "STACK_LOAD_INDEXED"),
    ExecutableInstruction("stack.store.indexed", "STACK_STORE_INDEXED"),
    ExecutableInstruction("stack.fill", "STACK_FILL"),
    ExecutableInstruction("stack.copy", "STACK_COPY"),
    ExecutableInstruction("stack.compare", "STACK_COMPARE"),
    ExecutableInstruction("stack.copy.rodata", "STACK_COPY_RODATA"),
    ExecutableInstruction("stack.const.s16.i32", "STACK_CONST_S16_I32"),
    ExecutableInstruction("stack.const.s16.i64", "STACK_CONST_S16_I64"),
    ExecutableInstruction("stack.pack.i32.u16.x2", "STACK_PACK_I32"),
    ExecutableInstruction("stack.pack.i32.u16.x4", "STACK_PACK_I32"),
    ExecutableInstruction("stack.pack.i32.u16.x8", "STACK_PACK_I32"),
    ExecutableInstruction("stack.pack.i64.u32.x2", "STACK_PACK_I64"),
    ExecutableInstruction("stack.pack.i64.u32.x4", "STACK_PACK_I64"),
    ExecutableInstruction("stack.pack.i64.u32.x8", "STACK_PACK_I64"),
)
