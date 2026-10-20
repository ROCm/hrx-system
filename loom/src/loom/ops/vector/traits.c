// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/vector/ops.h"

loom_trait_flags_t loom_vector_extract_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  loom_trait_flags_t traits = op->traits & ~LOOM_TRAIT_SAFE_TO_SPECULATE;
  if (loom_vector_extract_indices(op).count != 0) {
    return traits;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(op));
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(op);
  const uint8_t source_rank = loom_type_rank(source_type);
  if (!loom_type_is_vector(source_type) || !loom_type_is_scalar(result_type) ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != source_rank ||
      (source_rank != 0 && static_indices.i64_array == NULL)) {
    return traits;
  }

  for (uint8_t axis = 0; axis < source_rank; ++axis) {
    const int64_t index = static_indices.i64_array[axis];
    if (index < 0 || loom_type_dim_is_dynamic_at(source_type, axis) ||
        index >= loom_type_dim_static_size_at(source_type, axis)) {
      return traits;
    }
  }
  return traits | LOOM_TRAIT_SAFE_TO_SPECULATE;
}
