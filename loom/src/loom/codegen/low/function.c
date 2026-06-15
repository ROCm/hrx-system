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
