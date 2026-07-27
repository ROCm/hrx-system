# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vector dialect: register-lane operations over vector values."""

from __future__ import annotations

from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.combining import CombiningKind
from loom.dialect.vector.defs import (
    ALL_VECTOR_OPS,
    VECTOR_OP_CATEGORIES,
    VECTOR_OP_CATEGORY_GROUPS,
    FloatDot4F8Kind,
    IntegerDot4Kind,
    IntegerDot8I4Kind,
    QuantizeNaN,
    QuantizeTie,
    VectorFragmentRole,
    vector_ops,
)

__all__ = [
    "vector_ops",
    "AtomicKind",
    "AtomicOrdering",
    "AtomicScope",
    "CombiningKind",
    "FloatDot4F8Kind",
    "IntegerDot4Kind",
    "IntegerDot8I4Kind",
    "QuantizeNaN",
    "QuantizeTie",
    "VectorFragmentRole",
    "VECTOR_OP_CATEGORIES",
    "VECTOR_OP_CATEGORY_GROUPS",
    "ALL_VECTOR_OPS",
]
