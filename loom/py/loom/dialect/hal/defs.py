# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source operations projected from target-independent HAL methods."""

from loom.assembly import TreeRefs
from loom.dialect.hal.methods import ALL_HAL_METHODS
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64_ARRAY,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    Op,
    Operand,
    OperandTree,
)

hal_ops = Dialect("hal", dialect_id=0x07, doc="Host hardware orchestration.")

ALL_HAL_OPS = tuple(
    Op(
        f"hal.{method.name}",
        group=hal_ops,
        doc=method.doc,
        operands=[Operand("arguments", ANY, variadic=True, doc="SSA leaves in semantic field order.")],
        attrs=[AttrDef("shape", ATTR_TYPE_I64_ARRAY, doc="Static record, range and optional grouping of the operands.")],
        constraints=[OperandTree("arguments", "shape")],
        traits=[UNKNOWN_EFFECTS],
        verify="loom_hal_method_verify",
        builder_name=method.name.replace(".", "_"),
        format=[TreeRefs("arguments", "shape")],
        examples=method.examples,
    )
    for method in ALL_HAL_METHODS
)
