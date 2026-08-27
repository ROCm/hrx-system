// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

iree_status_t loom_test_condition_refines_positive_materialize(
    loom_rewriter_t* rewriter, const loom_op_t* condition_op,
    loom_value_id_t source, bool assumed_truth,
    loom_value_id_t* out_refined_value) {
  IREE_ASSERT(assumed_truth);
  loom_predicate_t* predicate = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &rewriter->module->arena, 1, sizeof(*predicate), (void**)&predicate));
  *predicate = (loom_predicate_t){
      .kind = LOOM_PREDICATE_GT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {source, 0, 0},
  };
  loom_type_t result_type = loom_module_value_type(rewriter->module, source);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(
      &rewriter->builder, &source, 1, predicate, 1, &result_type, 1,
      condition_op->location, &assume_op));
  *out_refined_value = loom_index_assume_results(assume_op).values[0];
  return iree_ok_status();
}
