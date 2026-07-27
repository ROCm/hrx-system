# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect: logical view operations."""

from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.view.defs import (
    ALL_VIEW_OPS,
    view_atomic_cmpxchg,
    view_atomic_reduce,
    view_atomic_rmw,
    view_load,
    view_ops,
    view_prefetch,
    view_refine,
    view_store,
    view_subview,
)

__all__ = [
    "view_ops",
    "AtomicKind",
    "AtomicOrdering",
    "AtomicScope",
    "ALL_VIEW_OPS",
    "view_subview",
    "view_refine",
    "view_load",
    "view_store",
    "view_atomic_reduce",
    "view_atomic_rmw",
    "view_atomic_cmpxchg",
    "view_prefetch",
]
