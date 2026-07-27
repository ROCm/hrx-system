# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pool dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ANY,
    INTEGER,
    OFFSET,
    POOL,
    PURE,
    TILE,
    Dialect,
    Op,
    Operand,
    Reads,
    Result,
    Writes,
)

# ============================================================================
# Group
# ============================================================================

pool_ops = Dialect("pool", dialect_id=0x0A, doc="Block-managed device memory pool ops.")

# ============================================================================
# pool.load — read a page from the pool as a typed tile
# ============================================================================

pool_load = Op(
    name="pool.load",
    group=pool_ops,
    doc="Read a page from the pool as a typed tile.",
    operands=[
        Operand("pool", POOL, doc="The pool to load from."),
        Operand("page_id", INTEGER, doc="Page index within the pool."),
        Operand("page_bytes", OFFSET, doc="Page stride in bytes."),
    ],
    results=[Result("result", TILE, doc="Typed tile view of the page.")],
    effects=[Reads("pool")],
    format=[
        Ref("pool"),
        COMMA,
        Ref("page_id"),
        COMMA,
        Ref("page_bytes"),
        COLON,
        TypeOf("pool"),
        COMMA,
        TypeOf("page_id"),
        COMMA,
        TypeOf("page_bytes"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tile = pool.load %pool, %pid, %pb : pool<[%BS]>, i32, offset -> tile<[16, 128]xf16>",
    ],
)

# ============================================================================
# pool.store — write tile data into the pool
# ============================================================================

pool_store = Op(
    name="pool.store",
    group=pool_ops,
    doc="Write tile data into the pool at a page offset.",
    operands=[
        Operand("pool", POOL, doc="The pool to store into."),
        Operand("page_id", INTEGER, doc="Page index within the pool."),
        Operand("page_bytes", OFFSET, doc="Page stride in bytes."),
        Operand("offset_in_page", OFFSET, doc="Byte offset within the page."),
        Operand("data", TILE, doc="Tile data to write."),
    ],
    results=[],
    effects=[Writes("pool")],
    format=[
        Ref("pool"),
        COMMA,
        Ref("page_id"),
        COMMA,
        Ref("page_bytes"),
        COMMA,
        Ref("offset_in_page"),
        COMMA,
        Ref("data"),
        COLON,
        TypeOf("pool"),
        COMMA,
        TypeOf("page_id"),
        COMMA,
        TypeOf("page_bytes"),
        COMMA,
        TypeOf("offset_in_page"),
        COMMA,
        TypeOf("data"),
    ],
    examples=[
        "pool.store %pool, %pid, %pb, %off, %data : pool<[%BS]>, i32, offset, offset, tile<[16, 128]xf16>",
    ],
)

# ============================================================================
# pool.pin — atomically increment block pin count
# ============================================================================

pool_pin = Op(
    name="pool.pin",
    group=pool_ops,
    doc="Atomically increment the pin count for a block.",
    operands=[
        Operand("pool", POOL, doc="The pool containing the block."),
        Operand("block_id", INTEGER, doc="Block index to pin."),
    ],
    results=[],
    effects=[Writes("pool")],
    format=[
        Ref("pool"),
        COMMA,
        Ref("block_id"),
        COLON,
        TypeOf("pool"),
        COMMA,
        TypeOf("block_id"),
    ],
    examples=[
        "pool.pin %pool, %bid : pool<[%BS]>, i32",
    ],
)

# ============================================================================
# pool.unpin — atomically decrement block pin count
# ============================================================================

pool_unpin = Op(
    name="pool.unpin",
    group=pool_ops,
    doc="Atomically decrement the pin count for a block.",
    operands=[
        Operand("pool", POOL, doc="The pool containing the block."),
        Operand("block_id", INTEGER, doc="Block index to unpin."),
    ],
    results=[],
    effects=[Writes("pool")],
    format=[
        Ref("pool"),
        COMMA,
        Ref("block_id"),
        COLON,
        TypeOf("pool"),
        COMMA,
        TypeOf("block_id"),
    ],
    examples=[
        "pool.unpin %pool, %bid : pool<[%BS]>, i32",
    ],
)

# ============================================================================
# pool.buffer — extract raw device buffer handle
# ============================================================================

pool_buffer = Op(
    name="pool.buffer",
    group=pool_ops,
    doc="Extract the raw device buffer handle from a pool.",
    operands=[
        Operand("pool", POOL, doc="The pool to extract the buffer from."),
    ],
    results=[Result("buffer", ANY, doc="Raw device buffer handle.")],
    traits=[PURE],
    format=[
        Ref("pool"),
        COLON,
        TypeOf("pool"),
        ARROW,
        ResultType("buffer"),
    ],
    examples=[
        "%buf = pool.buffer %pool : pool<[%BS]> -> hal.buffer",
    ],
)

# ============================================================================
# Registry: all pool ops in declaration order
# ============================================================================

ALL_POOL_OPS: tuple[Op, ...] = (
    pool_load,
    pool_store,
    pool_pin,
    pool_unpin,
    pool_buffer,
)
