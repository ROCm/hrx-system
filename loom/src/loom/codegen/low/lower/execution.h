// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source execution multiplicity shared by lowering selection and reporting.

#ifndef LOOM_CODEGEN_LOW_LOWER_EXECUTION_H_
#define LOOM_CODEGEN_LOW_LOWER_EXECUTION_H_

#include "iree/base/api.h"
#include "loom/util/cfg_execution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;

// Cached execution counts for the immutable source-function body. The owning
// lowering context retains this state until function lowering completes.
typedef struct loom_low_lower_execution_counts_t {
  // Shared loop/control result indexed by source body block ordinal.
  loom_cfg_execution_counts_t blocks;
  // True after analysis has completed, including an unknown result.
  bool initialized;
} loom_low_lower_execution_counts_t;

// Returns retained exact block execution counts, or NULL when the source CFG
// contains unmodeled paths, dynamic trip counts, or arithmetic overflow. The
// first query consumes the fact owner's loop structure and recurrence facts;
// subsequent queries only borrow function-lifetime storage. A NULL result is
// uncertainty about execution frequency, not an unsupported source program.
iree_status_t loom_low_lower_source_block_execution_counts(
    loom_low_lower_context_t* context, const uint64_t** out_counts);

// Returns the execution count for one source-function block when exact.
// Branch-local blocks are unknown while blocks after proven reconvergence can
// retain the multiplier contributed by enclosing fixed-trip loops.
iree_status_t loom_low_lower_source_block_execution_count(
    loom_low_lower_context_t* context, uint16_t block_index,
    uint64_t* out_count, bool* out_exact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_EXECUTION_H_
