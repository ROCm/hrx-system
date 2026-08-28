// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function.h"

#include "loom/ops/low/ops.h"

bool loom_low_function_def_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op);
}

loom_symbol_ref_t loom_low_function_callee(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_callee(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_callee(function_op);
  }
  return loom_symbol_ref_null();
}

loom_symbol_ref_t loom_low_function_target(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_target(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_target(function_op);
  }
  return loom_symbol_ref_null();
}

uint8_t loom_low_function_allocation(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_allocation(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_allocation(function_op);
  }
  return 0;
}

uint8_t loom_low_function_schedule(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_schedule(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_schedule(function_op);
  }
  return 0;
}

const loom_value_id_t* loom_low_function_result_ids(
    const loom_op_t* function_op, uint16_t* out_count) {
  *out_count = 0;
  if (loom_low_func_def_isa(function_op)) {
    const loom_value_slice_t results = loom_low_func_def_results(function_op);
    *out_count = results.count;
    return results.values;
  }
  return NULL;
}

loom_named_attr_slice_t loom_low_function_abi_layout(
    const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_abi_layout(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_abi_layout(function_op);
  }
  return loom_named_attr_slice_empty();
}

uint16_t loom_low_function_abi_layout_attr_index(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_abi_layout_ATTR_INDEX;
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_abi_layout_ATTR_INDEX;
  }
  return LOOM_ATTR_INDEX_NONE;
}

loom_region_t* loom_low_function_body(loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_body(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_body(function_op);
  }
  return NULL;
}

const loom_region_t* loom_low_function_const_body(
    const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_body(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_body(function_op);
  }
  return NULL;
}
