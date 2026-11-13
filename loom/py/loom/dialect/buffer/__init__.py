# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer dialect: opaque storage roots and typed view construction."""

from loom.dialect.buffer.defs import (
    ALL_BUFFER_OPS,
    BufferAddressPredicate,
    buffer_alloca,
    buffer_alloca_init,
    buffer_assume_memory_space,
    buffer_compare,
    buffer_compare_address,
    buffer_compare_null,
    buffer_constant,
    buffer_copy,
    buffer_duplicate,
    buffer_fence,
    buffer_fill,
    buffer_length,
    buffer_load_i8_u,
    buffer_null,
    buffer_ops,
    buffer_pack,
    buffer_store_i8,
    buffer_view,
)
from loom.dialect.memory import MemorySpace

__all__ = [
    "buffer_ops",
    "ALL_BUFFER_OPS",
    "BufferAddressPredicate",
    "MemorySpace",
    "buffer_alloca",
    "buffer_alloca_init",
    "buffer_assume_memory_space",
    "buffer_compare",
    "buffer_compare_address",
    "buffer_compare_null",
    "buffer_constant",
    "buffer_copy",
    "buffer_duplicate",
    "buffer_fill",
    "buffer_fence",
    "buffer_length",
    "buffer_load_i8_u",
    "buffer_null",
    "buffer_pack",
    "buffer_store_i8",
    "buffer_view",
]
