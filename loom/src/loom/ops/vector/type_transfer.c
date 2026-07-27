// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/ops.h"
#include "loom/rewrite/type_propagation.h"

iree_status_t loom_vector_transpose_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  (void)module;
  loom_type_t source_type =
      loom_type_transfer_value_type(context, loom_vector_transpose_source(op));
  loom_type_t result_type =
      loom_type_transfer_value_type(context, loom_vector_transpose_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  if (permutation.kind != LOOM_ATTR_I64_ARRAY ||
      (permutation.count > 0 && !permutation.i64_array)) {
    return iree_ok_status();
  }
  if (source_rank != result_rank || permutation.count != source_rank) {
    return iree_ok_status();
  }

  uint32_t seen_axes = 0;
  uint64_t result_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  uint64_t source_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t result_axis = 0; result_axis < result_rank; ++result_axis) {
    int64_t source_axis_i64 = permutation.i64_array[result_axis];
    if (source_axis_i64 < 0 || source_axis_i64 >= source_rank) {
      return iree_ok_status();
    }
    uint8_t source_axis = (uint8_t)source_axis_i64;
    uint32_t axis_bit = 1u << source_axis;
    if (iree_all_bits_set(seen_axes, axis_bit)) return iree_ok_status();
    seen_axes |= axis_bit;

    result_dimensions[result_axis] = loom_type_dim(source_type, source_axis);
    source_dimensions[source_axis] = loom_type_dim(result_type, result_axis);
  }

  IREE_RETURN_IF_ERROR(loom_type_transfer_seed_shape_dims(
      context, loom_vector_transpose_result(op), result_dimensions,
      result_rank));
  return loom_type_transfer_seed_shape_dims(context,
                                            loom_vector_transpose_source(op),
                                            source_dimensions, source_rank);
}
