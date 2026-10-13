// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/value_tree.h"

bool loom_value_tree_validate(const int64_t* shape, uint16_t shape_count,
                              uint16_t value_count, uint16_t result_count) {
  if (!shape || !shape_count) {
    return false;
  }
  uint32_t pending_count = 1;
  uint32_t leaf_count = 0;
  uint32_t output_count = 0;
  for (uint32_t i = 0; i < shape_count; ++i) {
    if (!pending_count || shape[i] < 0) {
      return false;
    }
    uint64_t token = (uint64_t)shape[i];
    uint32_t kind = token & 3;
    uint64_t child_count = token >> 2;
    if (token == LOOM_VALUE_TREE_RESULT) {
      ++output_count;
      child_count = 0;
    }
    if (kind < LOOM_VALUE_TREE_RECORD && child_count) {
      return false;
    }
    // Bound before narrowing or adding, even for malicious i64-sized counts.
    uint32_t remaining_count = shape_count - i - 1;
    if (child_count > remaining_count) {
      return false;
    }
    pending_count += (uint32_t)child_count;
    --pending_count;
    if (pending_count > remaining_count) {
      return false;
    }
    leaf_count += token == LOOM_VALUE_TREE_VALUE;
  }
  return pending_count == 0 && leaf_count == value_count &&
         output_count == result_count;
}
