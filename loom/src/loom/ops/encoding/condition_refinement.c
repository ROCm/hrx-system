// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Branch-edge refinement materializers for encoding predicates.

#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/rewrite/rewriter.h"

iree_status_t loom_encoding_isa_materialize_refinement(
    loom_rewriter_t* rewriter, const loom_op_t* condition_op,
    loom_value_id_t source, bool assumed_truth,
    loom_value_id_t* out_refined_value) {
  IREE_ASSERT(assumed_truth);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, source);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_assume_spec_build(
      &rewriter->builder, source, loom_encoding_isa_spec(condition_op),
      result_type, condition_op->location, &assume_op));
  *out_refined_value = loom_encoding_assume_spec_result(assume_op);
  return iree_ok_status();
}

iree_status_t loom_encoding_matches_materialize_refinement(
    loom_rewriter_t* rewriter, const loom_op_t* condition_op,
    loom_value_id_t source, bool assumed_truth,
    loom_value_id_t* out_refined_value) {
  IREE_ASSERT(assumed_truth);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, source);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_assume_match_build(
      &rewriter->builder, loom_encoding_matches_requirements(condition_op),
      source, result_type, condition_op->location, &assume_op));
  *out_refined_value = loom_encoding_assume_match_result(assume_op);
  return iree_ok_status();
}
