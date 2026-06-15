# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer dialect: opaque storage roots and typed view construction."""

from loom.dialect.buffer.defs import (
    ALL_BUFFER_OPS,
    buffer_alloca,
    buffer_assume_memory_space,
    buffer_ops,
    buffer_view,
)
from loom.dialect.memory import MemorySpace

__all__ = [
    "buffer_ops",
    "ALL_BUFFER_OPS",
    "MemorySpace",
    "buffer_alloca",
    "buffer_assume_memory_space",
    "buffer_view",
]
