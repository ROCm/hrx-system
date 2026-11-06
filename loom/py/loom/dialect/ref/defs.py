# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Managed object handle operations."""

from loom.dsl import (
    MANAGED_REFERENCE,
    OBSERVABLE_EFFECT,
    Dialect,
    Op,
    value_duplication_op,
)

ref_ops = Dialect(
    "ref",
    dialect_id=0x26,
    doc="Lifetime and identity operations on managed object handles.",
)

ref_duplicate = value_duplication_op(
    "ref.duplicate",
    group=ref_ops,
    carrier=MANAGED_REFERENCE,
    traits=[OBSERVABLE_EFFECT],
    doc=(
        "Create an independent handle to the same managed payload, preserving "
        "its type and null state. Duplicating ref adds an owner that must be "
        "released or transferred independently. Duplicating ptr preserves its "
        "borrowed mode and any protection needed through its final use. The "
        "source remains valid; live source and result lifetimes are independent."
    ),
    examples=[
        "%copy = ref.duplicate %owner : ref<@Object>",
        "%saved = ref.duplicate %pointer : ptr<@Object>",
    ],
)

ALL_REF_OPS: tuple[Op, ...] = (ref_duplicate,)
