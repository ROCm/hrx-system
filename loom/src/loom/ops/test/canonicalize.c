// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonicalization implementations for the test dialect.
// These are hand-written (not generated) and linked into the test
// dialect library so the vtable function pointers resolve.

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

// test.addi canonicalization: addi(x, 0) → x, addi(0, x) → x.
iree_status_t loom_test_addi_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_test_addi_lhs(op);
  loom_value_id_t rhs = loom_test_addi_rhs(op);

  // addi(x, 0) → x.
  loom_value_t* rhs_value = loom_module_value(rewriter->module, rhs);
  if (!loom_value_is_block_arg(rhs_value)) {
    loom_op_t* rhs_def = loom_value_def_op(rhs_value);
    if (rhs_def && loom_test_constant_isa(rhs_def)) {
      int64_t value = loom_attr_as_i64(loom_op_attrs(rhs_def)[0]);
      if (value == 0) {
        return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &lhs, 1);
      }
    }
  }

  // addi(0, x) → x (commutative).
  loom_value_t* lhs_value = loom_module_value(rewriter->module, lhs);
  if (!loom_value_is_block_arg(lhs_value)) {
    loom_op_t* lhs_def = loom_value_def_op(lhs_value);
    if (lhs_def && loom_test_constant_isa(lhs_def)) {
      int64_t value = loom_attr_as_i64(loom_op_attrs(lhs_def)[0]);
      if (value == 0) {
        return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &rhs, 1);
      }
    }
  }

  return iree_ok_status();
}

// test.counter canonicalization:
//   value < 0  → return IREE_STATUS_INTERNAL (error path testing).
//   value > 0  → replace with counter(value - 1) (multi-step testing).
//   value == 0 → no change (fixed point).
iree_status_t loom_test_counter_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  int64_t value = loom_test_counter_value(op);
  if (value < 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "test.counter canonicalize error sentinel (value=%" PRId64 ")", value);
  }
  if (value == 0) return iree_ok_status();

  // Replace with a new counter op whose value is decremented by 1.
  loom_value_id_t old_result = loom_test_counter_result(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, old_result);
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_test_counter_build(
      &rewriter->builder, value - 1, result_type, op->location, &replacement));
  loom_value_id_t new_result = loom_test_counter_result(replacement);
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &new_result, 1);
}
