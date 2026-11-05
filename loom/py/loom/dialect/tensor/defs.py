# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tensor operation definitions."""

from loom.dsl import TENSOR, Dialect, Op, TypeDef, value_duplication_op

tensor_ops = Dialect(
    "tensor",
    dialect_id=0x04,
    doc="Logical tensor values and their structural operations.",
)

tensor_duplicate = value_duplication_op(
    "tensor.duplicate",
    group=tensor_ops,
    carrier=TENSOR,
    doc=("Create another local owner of a tensor and its dependent dimensions or encoding. The duplicate carries the same elements through a distinct SSA ownership identity."),
    examples=["%copy_m, %copy = tensor.duplicate %m, %source : index, tensor<[%copy_m]xf32>"],
)

ALL_TENSOR_TYPES: tuple[TypeDef, ...] = ()
ALL_TENSOR_OPS: tuple[Op, ...] = (tensor_duplicate,)
