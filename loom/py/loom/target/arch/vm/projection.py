# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Maps Loom source semantics to authoritative VM instruction objects."""

from __future__ import annotations

from typing import NamedTuple

from iree.vm.bytecode.spec.isa import (
    Instruction,
    IntegerBinaryOperation,
    IntegerBinarySemantics,
)
from iree.vm.bytecode.spec.specification import SPECIFICATION

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dsl import Op
from loom.scalar_type import ScalarTypeKind


class SourceLowering(NamedTuple):
    source_op: Op
    source_op_kind: int
    scalar_type: ScalarTypeKind
    instruction: Instruction


_SOURCE_OP_BY_OPERATION = {
    IntegerBinaryOperation.ADD: scalar_arithmetic.scalar_addi,
    IntegerBinaryOperation.SUB: scalar_arithmetic.scalar_subi,
    IntegerBinaryOperation.MUL: scalar_arithmetic.scalar_muli,
}

_SCALAR_TYPE_BY_BIT_WIDTH = {
    32: ScalarTypeKind.I32,
    64: ScalarTypeKind.I64,
}

_SCALAR_OP_KIND_BY_ID = {
    id(op): (op.group.dialect_id << 8) | op_index
    for op_index, op in enumerate(ALL_SCALAR_OPS)
}

SOURCE_LOWERINGS = tuple(
    SourceLowering(
        _SOURCE_OP_BY_OPERATION[instruction.semantics.operation],
        _SCALAR_OP_KIND_BY_ID[
            id(_SOURCE_OP_BY_OPERATION[instruction.semantics.operation])
        ],
        _SCALAR_TYPE_BY_BIT_WIDTH[instruction.semantics.bit_width],
        instruction,
    )
    for instruction in SPECIFICATION.instructions
    if isinstance(instruction.semantics, IntegerBinarySemantics)
)
