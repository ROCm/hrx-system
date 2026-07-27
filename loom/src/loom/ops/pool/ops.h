// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_POOL_OPS_H_
#define LOOM_OPS_POOL_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_POOL_LOAD = LOOM_OP_KIND(LOOM_DIALECT_POOL, 0),
  LOOM_OP_POOL_STORE = LOOM_OP_KIND(LOOM_DIALECT_POOL, 1),
  LOOM_OP_POOL_PIN = LOOM_OP_KIND(LOOM_DIALECT_POOL, 2),
  LOOM_OP_POOL_UNPIN = LOOM_OP_KIND(LOOM_DIALECT_POOL, 3),
  LOOM_OP_POOL_BUFFER = LOOM_OP_KIND(LOOM_DIALECT_POOL, 4),
  LOOM_OP_POOL_COUNT_ = 5,
};

// LOOM_OP_POOL_LOAD: Read a page from the pool as a typed tile.
// %tile = pool.load %pool, %pid, %pb : pool<[%BS]>, i32, offset -> tile<[16, 128]xf16>
LOOM_DEFINE_ISA(loom_pool_load_isa, LOOM_OP_POOL_LOAD)
LOOM_DEFINE_OPERAND(loom_pool_load_pool, 0)
LOOM_DEFINE_OPERAND(loom_pool_load_page_id, 1)
LOOM_DEFINE_OPERAND(loom_pool_load_page_bytes, 2)
LOOM_DEFINE_RESULT(loom_pool_load_result, 0)
iree_status_t loom_pool_load_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t pool,
    loom_may_consume loom_value_id_t page_id,
    loom_may_consume loom_value_id_t page_bytes,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_POOL_STORE: Write tile data into the pool at a page offset.
// pool.store %pool, %pid, %pb, %off, %data : pool<[%BS]>, i32, offset, offset, tile<[16, 128]xf16>
LOOM_DEFINE_ISA(loom_pool_store_isa, LOOM_OP_POOL_STORE)
LOOM_DEFINE_OPERAND(loom_pool_store_pool, 0)
LOOM_DEFINE_OPERAND(loom_pool_store_page_id, 1)
LOOM_DEFINE_OPERAND(loom_pool_store_page_bytes, 2)
LOOM_DEFINE_OPERAND(loom_pool_store_offset_in_page, 3)
LOOM_DEFINE_OPERAND(loom_pool_store_data, 4)
iree_status_t loom_pool_store_build(
    loom_builder_t* builder,
    loom_value_id_t pool,
    loom_value_id_t page_id,
    loom_value_id_t page_bytes,
    loom_value_id_t offset_in_page,
    loom_value_id_t data,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_POOL_PIN: Atomically increment the pin count for a block.
// pool.pin %pool, %bid : pool<[%BS]>, i32
LOOM_DEFINE_ISA(loom_pool_pin_isa, LOOM_OP_POOL_PIN)
LOOM_DEFINE_OPERAND(loom_pool_pin_pool, 0)
LOOM_DEFINE_OPERAND(loom_pool_pin_block_id, 1)
iree_status_t loom_pool_pin_build(
    loom_builder_t* builder,
    loom_value_id_t pool,
    loom_value_id_t block_id,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_POOL_UNPIN: Atomically decrement the pin count for a block.
// pool.unpin %pool, %bid : pool<[%BS]>, i32
LOOM_DEFINE_ISA(loom_pool_unpin_isa, LOOM_OP_POOL_UNPIN)
LOOM_DEFINE_OPERAND(loom_pool_unpin_pool, 0)
LOOM_DEFINE_OPERAND(loom_pool_unpin_block_id, 1)
iree_status_t loom_pool_unpin_build(
    loom_builder_t* builder,
    loom_value_id_t pool,
    loom_value_id_t block_id,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_POOL_BUFFER: Extract the raw device buffer handle from a pool.
// %buf = pool.buffer %pool : pool<[%BS]> -> hal.buffer
LOOM_DEFINE_ISA(loom_pool_buffer_isa, LOOM_OP_POOL_BUFFER)
LOOM_DEFINE_OPERAND(loom_pool_buffer_pool, 0)
LOOM_DEFINE_RESULT(loom_pool_buffer_buffer, 0)
iree_status_t loom_pool_buffer_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t pool,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the pool dialect.
const loom_op_vtable_t* const* loom_pool_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the pool dialect.
const loom_op_semantics_t* loom_pool_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a pool op kind, or empty metadata.
loom_op_semantics_t loom_pool_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_POOL_OPS_H_
