// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common representation of copying and moving register/storage transfers.
#ifndef LOOM_OPS_LOW_STORAGE_H_
#define LOOM_OPS_LOW_STORAGE_H_

#include "loom/ops/low/ops.h"

typedef enum loom_low_register_transfer_kind_e {
  // Operation does not structurally transfer register storage.
  LOOM_LOW_REGISTER_TRANSFER_NONE = 0,
  // Copy into a fresh register identity.
  LOOM_LOW_REGISTER_TRANSFER_COPY = 1,
  // Move into a fresh register identity.
  LOOM_LOW_REGISTER_TRANSFER_MOVE = 2,
  // Project a contiguous register subrange.
  LOOM_LOW_REGISTER_TRANSFER_SLICE = 3,
  // Compose ordered register subranges.
  LOOM_LOW_REGISTER_TRANSFER_CONCAT = 4,
} loom_low_register_transfer_kind_t;

static_assert(LOOM_OP_LOW_MOVE - LOOM_OP_LOW_COPY ==
                  LOOM_LOW_REGISTER_TRANSFER_MOVE -
                      LOOM_LOW_REGISTER_TRANSFER_COPY,
              "low register transfers must follow semantic kind order");
static_assert(LOOM_OP_LOW_SLICE - LOOM_OP_LOW_COPY ==
                  LOOM_LOW_REGISTER_TRANSFER_SLICE -
                      LOOM_LOW_REGISTER_TRANSFER_COPY,
              "low register transfers must follow semantic kind order");
static_assert(LOOM_OP_LOW_CONCAT - LOOM_OP_LOW_COPY ==
                  LOOM_LOW_REGISTER_TRANSFER_CONCAT -
                      LOOM_LOW_REGISTER_TRANSFER_COPY,
              "low register transfers must follow semantic kind order");

// Classifies the structural register transfer represented by |op|.
static inline loom_low_register_transfer_kind_t loom_low_register_transfer_kind(
    const loom_op_t* op) {
  if (op->kind < LOOM_OP_LOW_COPY || op->kind > LOOM_OP_LOW_CONCAT) {
    return LOOM_LOW_REGISTER_TRANSFER_NONE;
  }
  return (loom_low_register_transfer_kind_t)(LOOM_LOW_REGISTER_TRANSFER_COPY +
                                             op->kind - LOOM_OP_LOW_COPY);
}

static inline bool loom_low_storage_store_isa(const loom_op_t* op) {
  return loom_low_spill_isa(op) || loom_low_spill_move_isa(op);
}

static inline bool loom_low_storage_load_isa(const loom_op_t* op) {
  return loom_low_reload_isa(op) || loom_low_reload_move_isa(op);
}

static inline bool loom_low_storage_transfer_moves(const loom_op_t* op) {
  return loom_low_spill_move_isa(op) || loom_low_reload_move_isa(op);
}

// The transfer family shares its operand, result, and offset layout.
static inline loom_value_id_t loom_low_storage_store_value(
    const loom_op_t* op) {
  return loom_op_const_operands(op)[0];
}

static inline loom_value_id_t loom_low_storage_store_storage(
    const loom_op_t* op) {
  return loom_op_const_operands(op)[1];
}

static inline loom_value_id_t loom_low_storage_load_storage(
    const loom_op_t* op) {
  return loom_op_const_operands(op)[0];
}

static inline loom_value_id_t loom_low_storage_load_result(
    const loom_op_t* op) {
  return loom_op_const_results(op)[0];
}

static inline int64_t loom_low_storage_transfer_offset(const loom_op_t* op) {
  return loom_attr_as_i64(loom_op_const_attrs(op)[0]);
}

#endif  // LOOM_OPS_LOW_STORAGE_H_
