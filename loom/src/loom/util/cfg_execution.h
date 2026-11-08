// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution-frequency uncertainty over retained CFG control dependence.

#ifndef LOOM_UTIL_CFG_EXECUTION_H_
#define LOOM_UTIL_CFG_EXECUTION_H_

#include "iree/base/bitmap.h"
#include "loom/util/cfg_control.h"
#include "loom/util/cfg_loop_nest.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |block_index| selects a control alternative whose
// execution frequency is modeled independently by the caller.
typedef bool (*loom_cfg_execution_selector_is_modeled_fn_t)(
    const void* user_data, uint16_t block_index);

typedef struct loom_cfg_execution_selector_model_t {
  // Selector classification callback, or NULL when no selector is modeled.
  loom_cfg_execution_selector_is_modeled_fn_t is_modeled;
  // Caller-owned payload passed to |is_modeled|.
  const void* user_data;
} loom_cfg_execution_selector_model_t;

// Classifies blocks whose execution count depends on an unmodeled selector.
//
// Control alternatives affect the path from their target up to, but excluding,
// the selector's immediate postdominator. Consequently a reconverged block is
// exact again when its enclosing execution count is exact. Modeled selectors,
// such as a fixed-trip loop header, do not introduce uncertainty; their caller
// accounts for the corresponding multiplicity separately.
//
// The returned bitmap has one bit per CFG block and is owned by |arena|.
// Unreachable blocks are unmarked. Classification consumes the retained
// compressed control graph in O(B+C+I) time for B blocks, C control components,
// and I component inputs. O(C) temporary storage is released before return.
iree_status_t loom_cfg_execution_classify_unmodeled_blocks(
    const loom_cfg_control_t* control,
    loom_cfg_execution_selector_model_t selector_model,
    iree_arena_allocator_t* arena, iree_bitmap_t* out_unmodeled_blocks);

// Execution multiplicity retained for one immutable CFG. Unknown alternatives
// do not discard counts at blocks before their split or after reconvergence.
typedef struct loom_cfg_execution_counts_t {
  // Loop-derived count per block, or NULL when the requested exactness cannot
  // be established from the available loop and control information.
  const uint64_t* values;
  // Blocks controlled by an unmodeled selector. An empty bitmap means every
  // retained count is exact; unreachable blocks have exact count zero.
  iree_bitmap_t unmodeled_blocks;
} loom_cfg_execution_counts_t;

// Combines exact loop trip counts with retained control dependence. storage
// has graph->block_count entries and outlives the result. NULL control requests
// only the single-path proof, avoiding partial-count analysis when unused.
// Otherwise branch-local counts remain unknown while reconverged blocks retain
// their exact enclosing multiplier. No CFG/loop/control discovery is repeated.
// Status carries allocation failure for the uncertainty bitmap.
iree_status_t loom_cfg_execution_counts_calculate(
    const loom_cfg_loop_nest_t* loops, const uint64_t* trip_counts,
    const loom_cfg_control_t* control, iree_arena_allocator_t* arena,
    uint64_t* storage, loom_cfg_execution_counts_t* out_counts);

// Queries retained source information without revisiting control flow.
static inline bool loom_cfg_execution_count_is_exact(
    const loom_cfg_execution_counts_t* counts, uint16_t block_index) {
  return counts->values &&
         (!counts->unmodeled_blocks.bit_count ||
          !iree_bitmap_test(counts->unmodeled_blocks, block_index));
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_CFG_EXECUTION_H_
