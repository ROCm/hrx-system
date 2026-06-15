// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IR walker: iterative DFS traversal of the region nesting tree.
//
// Walks all operations in a region (and recursively in nested regions)
// in dominance order, providing nesting context to the callback at
// each operation. The context includes the containing block, region,
// parent op, and nesting depth — the information that parent pointers
// on the IR structs would provide, threaded through the walk for free.
//
// For structured regions (no LOOM_REGION_INSTANCE_FLAG_CFG), dominance
// order is:
//   - Ops within a block are visited in block order.
//   - For multi-block regions, the entry block (block 0) is visited
//     first, then sibling blocks. Entry block dominates all siblings
//     in structured control flow.
//   - Nested regions are visited between the ops of the parent block
//     (pre-order: parent op before children; post-order: children
//     before parent op).
//
// The walker is iterative (no recursion) with an arena-allocated
// stack. Stack growth uses iree_arena_grow_array for O(1) amortized
// push.
//
// Usage:
//
//   static iree_status_t my_visitor(void* user_data, loom_op_t* op,
//                                   const loom_walk_context_t* context,
//                                   loom_walk_result_t* out_result) {
//     *out_result = LOOM_WALK_CONTINUE;
//     // ... process op using context->block, context->parent_op ...
//     return iree_ok_status();
//   }
//
//   loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
//   IREE_RETURN_IF_ERROR(loom_walk_function(
//       module, function, LOOM_WALK_PRE_ORDER,
//       (loom_walk_callback_t){my_visitor, &my_state},
//       &arena, &walk_result));
//   if (walk_result == LOOM_WALK_ABORT) { /* ... */ }

#ifndef LOOM_UTIL_WALK_H_
#define LOOM_UTIL_WALK_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Walk types
//===----------------------------------------------------------------------===//

// Traversal order for the walk.
typedef enum loom_walk_order_e {
  // Visit the op, then descend into its nested regions.
  LOOM_WALK_PRE_ORDER = 0,
  // Descend into nested regions, then visit the op.
  LOOM_WALK_POST_ORDER = 1,
} loom_walk_order_t;

// Walk control result returned by the callback via out parameter.
typedef enum loom_walk_result_e {
  // Continue walking. For pre-order, descend into nested regions.
  LOOM_WALK_CONTINUE = 0,
  // Skip the current op's nested regions (pre-order only).
  // In post-order this is treated as CONTINUE since children have
  // already been visited.
  LOOM_WALK_SKIP = 1,
  // Abort the walk entirely. The walk function returns with
  // *out_result set to LOOM_WALK_ABORT.
  LOOM_WALK_ABORT = 2,
} loom_walk_result_t;

// Nesting context provided to the walk callback. Contains the
// information needed for dominance reasoning without requiring
// separate parent pointer lookups.
typedef struct loom_walk_context_t {
  // Block containing the current op.
  loom_block_t* block;
  // Region containing the block.
  loom_region_t* region;
  // Op whose region contains the block. NULL for the top-level
  // region passed to loom_walk_region.
  loom_op_t* parent_op;
  // Region nesting depth. 0 for the top-level region.
  uint16_t depth;
} loom_walk_context_t;

//===----------------------------------------------------------------------===//
// Walk callback
//===----------------------------------------------------------------------===//

// Walk callback function. |user_data| is the first parameter for
// register placement. Returns iree_status_t for infrastructure
// failures (OOM, invalid IR). Walk control is via |out_result|,
// which must always be written.
typedef iree_status_t (*loom_walk_fn_t)(void* user_data, loom_op_t* op,
                                        const loom_walk_context_t* context,
                                        loom_walk_result_t* out_result);

// Callback bundle: function pointer and user data.
typedef struct loom_walk_callback_t {
  loom_walk_fn_t fn;
  void* user_data;
} loom_walk_callback_t;

//===----------------------------------------------------------------------===//
// Walk functions
//===----------------------------------------------------------------------===//

// Walks all ops in |region| and its nested regions in the specified
// order, invoking |callback| for each live op.
//
// On success, |*out_result| is LOOM_WALK_CONTINUE if the walk
// completed normally, or LOOM_WALK_ABORT if the callback aborted.
// LOOM_WALK_SKIP is consumed internally and never returned here.
//
// The walker uses an arena-allocated stack that grows as needed.
// All stack memory is allocated from |arena| and freed when the
// arena is reset/deinitialized.
iree_status_t loom_walk_region(const loom_module_t* module,
                               loom_region_t* region, loom_walk_order_t order,
                               loom_walk_callback_t callback,
                               iree_arena_allocator_t* arena,
                               loom_walk_result_t* out_result);

// Walks all ops in |function|'s root regions. Regions are visited in physical
// region order; nested regions are handled by loom_walk_region. Returns
// iree_ok_status() with *out_result = LOOM_WALK_CONTINUE if the function has no
// regions.
iree_status_t loom_walk_function(const loom_module_t* module,
                                 loom_func_like_t function,
                                 loom_walk_order_t order,
                                 loom_walk_callback_t callback,
                                 iree_arena_allocator_t* arena,
                                 loom_walk_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_WALK_H_
