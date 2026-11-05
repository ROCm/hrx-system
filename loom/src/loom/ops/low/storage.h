// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common representation of copying and moving register/storage transfers.
#ifndef LOOM_OPS_LOW_STORAGE_H_
#define LOOM_OPS_LOW_STORAGE_H_

#include "loom/ops/low/ops.h"

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
