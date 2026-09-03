# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group dialect type and operation definitions."""

from loom.assembly import ARROW, COLON, Ref, ResultType, TypeOf
from loom.dsl import ANY, INDEX, Dialect, Op, Operand, Result, TypeDef

group_ops = Dialect(
    "group",
    dialect_id=0x22,
    doc="Named SSA scheduling groups shared by pipelines and target layers.",
)

group_type = TypeDef(
    "group",
    doc=("Opaque scheduling identity whose cardinality is supplied by the operation that creates it. Distinct group values remain distinct even when their cardinalities are equal."),
)

group_create = Op(
    "group.create",
    group=group_ops,
    doc=(
        "Create a distinct scheduling group with SSA-defined cardinality. "
        "The cardinality may be specialized from workload or target queries; "
        "consumers use ordinary value facts when an exact count is required."
    ),
    operands=[Operand("cardinality", INDEX, doc="Number of group lanes.")],
    results=[
        Result(
            "result",
            ANY,
            allocates=True,
            doc="Fresh scheduling identity.",
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
    examples=["%workers = group.create %worker_count : index -> group"],
)

ALL_GROUP_TYPES: tuple[TypeDef, ...] = (group_type,)
ALL_GROUP_OPS: tuple[Op, ...] = (group_create,)
