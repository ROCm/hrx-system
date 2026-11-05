# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Communication-group operation definitions."""

from loom.assembly import ARROW, COLON, Ref, ResultType, TypeOf
from loom.dsl import (
    GROUP,
    INDEX,
    Dialect,
    Op,
    Operand,
    Result,
    TypeDef,
    value_duplication_op,
)

group_ops = Dialect(
    "group",
    dialect_id=0x22,
    doc="Shaped participant domains shared by pipelines and target layers.",
)

group_create = Op(
    "group.create",
    group=group_ops,
    doc=("Create a distinct rank-one communication domain with SSA-defined cardinality. The result identifies participants; it contains no participant values, channels, or physical resources."),
    operands=[Operand("cardinality", INDEX, doc="Number of group lanes.")],
    results=[
        Result(
            "result",
            GROUP,
            allocates=True,
            doc="Fresh communication-domain identity.",
        )
    ],
    verify="loom_group_create_verify",
    format=[
        Ref("cardinality"),
        COLON,
        TypeOf("cardinality"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%workers = group.create %worker_count : index -> group<[%worker_count]>"],
)

group_duplicate = value_duplication_op(
    "group.duplicate",
    group=group_ops,
    carrier=GROUP,
    doc=("Create another local owner of a communication group and its dependent dimensions. The duplicate names the same participant domain through a distinct SSA ownership identity."),
    examples=["%copy_count, %copy = group.duplicate %count, %source : index, group<[%copy_count]>"],
)

ALL_GROUP_TYPES: tuple[TypeDef, ...] = ()
ALL_GROUP_OPS: tuple[Op, ...] = (group_create, group_duplicate)
