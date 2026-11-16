// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/index_bounds.h"

#include "loom/ops/index/ops.h"

iree_status_t loom_vector_assume_axis_index(loom_rewriter_t* rewriter,
                                            loom_type_t vector_type,
                                            uint8_t axis, int64_t extent,
                                            loom_value_id_t index,
                                            loom_location_id_t location,
                                            loom_value_id_t* out_index) {
  *out_index = index;
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    return iree_ok_status();
  }
  const int64_t maximum =
      loom_type_dim_static_size_at(vector_type, axis) - extent;
  const loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, index);
  if (facts.range_lo >= 0 && facts.range_hi <= maximum) {
    return iree_ok_status();
  }
  const loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {index, 0, maximum},
  };
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(&rewriter->builder, &index, 1,
                                               &range, 1, &index_type, 1,
                                               location, &assume_op));
  *out_index = loom_index_assume_results(assume_op).values[0];
  return iree_ok_status();
}
