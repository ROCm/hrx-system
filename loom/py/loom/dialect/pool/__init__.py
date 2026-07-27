# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pool dialect: block-managed device memory pool ops.

Pools are pre-allocated contiguous regions of device memory divided
into fixed-size blocks. The pool type carries a single parameter —
block size in bytes — and is passed as a function argument. Multiple
pools with different block sizes can coexist (KV cache pages vs MoE
expert weights).

Pool ops provide the interface between block-managed device memory
and the tile compute world:

  Device-side (run in kernels):
    pool.load   — read a page as a typed tile
    pool.store  — write tile data into the pool

  Host-side (scheduling/orchestration):
    pool.pin    — atomically increment block pin count
    pool.unpin  — atomically decrement block pin count
    pool.buffer — extract raw device buffer handle

Page addressing uses i32 page IDs with a dynamic offset-typed byte stride:
byte_offset = page_id * page_bytes. No indirection, no page table.
The encoding is on the tiles, not the pool — pool ops are
encoding-agnostic byte stores.
"""

from loom.dialect.pool.defs import (
    ALL_POOL_OPS,
    pool_buffer,
    pool_load,
    pool_ops,
    pool_pin,
    pool_store,
    pool_unpin,
)

__all__ = [
    "pool_ops",
    "ALL_POOL_OPS",
    "pool_load",
    "pool_store",
    "pool_pin",
    "pool_unpin",
    "pool_buffer",
]
