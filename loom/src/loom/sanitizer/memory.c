// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/memory.h"

#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

bool loom_sanitizer_query_view_memory_space(
    const loom_rewriter_t* rewriter, loom_value_id_t view,
    loom_value_fact_memory_space_t* out_memory_space) {
  *out_memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!rewriter->fact_table) return false;
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, view), &reference)) {
    return false;
  }
  *out_memory_space = reference.memory_space;
  return true;
}

bool loom_sanitizer_access_memory_space_is_observable(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return false;
  }
  return false;
}

bool loom_sanitizer_access_view_is_observable(const loom_rewriter_t* rewriter,
                                              loom_value_id_t view) {
  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_query_view_memory_space(rewriter, view, &memory_space)) {
    return true;
  }
  return loom_sanitizer_access_memory_space_is_observable(memory_space);
}
