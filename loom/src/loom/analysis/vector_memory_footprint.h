// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// View memory footprint legality analysis.
//
// This is the production proof gate for view-based scalar and vector memory
// ops. It keeps cheap structural checks in the ordinary verifier and proves
// dynamic footprint inequalities with value facts, symbolic expressions, and
// view region summaries immediately before lowering out of Loom IR.

#ifndef LOOM_ANALYSIS_VECTOR_MEMORY_FOOTPRINT_H_
#define LOOM_ANALYSIS_VECTOR_MEMORY_FOOTPRINT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vector_memory_footprint_options_t {
  // Scratch arena used for symbolic expressions.
  iree_arena_allocator_t* arena;
  // Caller-owned value facts for |function|.
  const loom_value_fact_table_t* fact_table;
  // Structured diagnostic emitter for user footprint proof failures.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before suppressing further diagnostics.
  // Zero means no limit.
  uint32_t max_errors;
} loom_vector_memory_footprint_options_t;

typedef struct loom_vector_memory_footprint_result_t {
  // Number of memory ops checked.
  uint32_t checked_op_count;
  // Number of memory ops skipped because no lane accesses memory.
  uint32_t skipped_op_count;
  // Number of footprint proof errors emitted.
  uint32_t error_count;
} loom_vector_memory_footprint_result_t;

// Verifies all scalar and vector view memory footprints in |function|.
//
// User IR proof failures are counted in |out_result| and emitted through
// |options->emitter|. Infrastructure failures such as fact table allocation
// failures are returned as status.
iree_status_t loom_vector_memory_footprint_verify_function(
    loom_module_t* module, loom_func_like_t function,
    const loom_vector_memory_footprint_options_t* options,
    loom_vector_memory_footprint_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_VECTOR_MEMORY_FOOTPRINT_H_
