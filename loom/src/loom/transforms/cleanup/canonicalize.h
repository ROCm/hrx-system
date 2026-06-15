// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CANONICALIZE_H_
#define LOOM_TRANSFORMS_CANONICALIZE_H_

#include "loom/pass/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_canonicalizer_state_t loom_canonicalizer_state_t;

//===----------------------------------------------------------------------===//
// Canonicalizer driver
//===----------------------------------------------------------------------===//

// Default maximum number of canonicalizer fixed-point iterations.
#define LOOM_CANONICALIZER_DEFAULT_MAX_ITERATIONS 10

// Canonicalizer driver options. Zero-initialized options use defaults.
typedef struct loom_canonicalizer_options_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;

  // Optional seed facts cloned into the driver-owned fact table before the
  // initial function analysis. Extension payloads are re-interned, so the seed
  // table may come from a different fact context.
  const loom_value_fact_table_t* seed_facts;
} loom_canonicalizer_options_t;

// Summary of one canonicalizer function run.
typedef struct loom_canonicalizer_result_t {
  // True if the driver changed IR by erasing, replacing, moving, creating, or
  // otherwise mutating an operation/value.
  bool changed;

  // True if incremental fact recomputation changed at least one stored value
  // fact during rewriting.
  bool facts_changed;

  // True if a rewrite changed at least one value type.
  bool types_changed;

  // Conservative boundary invalidation bit. True when summaries derived from
  // this function's externally visible values may need recomputation.
  bool boundary_maybe_changed;

  // Number of ops modified by canonicalization.
  int64_t ops_modified;
} loom_canonicalizer_result_t;

// Stateful canonicalizer that can be driven by a pass or by whole-program
// refinement. The driver owns a resettable scratch arena for one function run;
// the caller owns |parent_arena| and the IR module arena.
typedef struct loom_canonicalizer_t {
  // Module being transformed.
  loom_module_t* module;

  // Caller-owned reusable value-fact storage used for region analysis.
  loom_pass_value_fact_owner_t* value_facts;

  // Parent arena whose block pool backs the resettable scratch arena.
  iree_arena_allocator_t* parent_arena;

  // Reset before each region run; owns the rewriter worklist and
  // symbolic-expression scratch state for the most recent run.
  iree_arena_allocator_t scratch_arena;

  // True after scratch_arena has been initialized and before deinitialize.
  bool scratch_arena_initialized;

  // Private driver state allocated from parent_arena. Holds the active rewriter
  // and any future canonicalizer-local state without exposing implementation
  // details through this header.
  loom_canonicalizer_state_t* state;
} loom_canonicalizer_t;

// Initializes a canonicalizer over |module|. |parent_arena| is not used for
// bulk scratch allocations directly; its block pool backs a nested arena that
// is reset for each run.
iree_status_t loom_canonicalizer_initialize(
    loom_module_t* module, iree_arena_allocator_t* parent_arena,
    loom_pass_value_fact_owner_t* value_facts,
    loom_canonicalizer_t* out_canonicalizer);

// Releases transient worklist state and returns scratch blocks to the parent
// arena's block pool. Does not modify the module.
void loom_canonicalizer_deinitialize(loom_canonicalizer_t* canonicalizer);

// Runs canonicalization on an explicit region tree. |function| supplies the
// logical function context for value-fact inference and may be empty for
// detached regions. |parent_op| owns the root entry block arguments when
// provided.
iree_status_t loom_canonicalizer_run_region(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result);

// Runs canonicalization on a function-like op's root regions. Non-body regions
// run first, then their facts seed body-region canonicalization.
iree_status_t loom_canonicalizer_run_function(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result);

// Returns the fact table from the most recent run. The table is invalidated by
// the next run and by loom_canonicalizer_deinitialize.
const loom_value_fact_table_t* loom_canonicalizer_fact_table(
    const loom_canonicalizer_t* canonicalizer);

//===----------------------------------------------------------------------===//
// Pass wrapper
//===----------------------------------------------------------------------===//

// Returns immutable metadata for the canonicalize pass.
const loom_pass_info_t* loom_canonicalize_pass_info(void);

// Creates canonicalize pass state from a textual option dictionary.
iree_status_t loom_canonicalize_create(loom_pass_t* pass,
                                       iree_string_view_t options);

// Canonicalize pass.
//
// Iterates all ops in a function, calling each op's vtable canonicalize
// function (if non-NULL) through the rewriter. The rewriter's worklist
// tracks what needs revisiting after each transformation. Iterates
// until fixed point or max iterations.
//
// Each op kind defines its own canonicalization patterns (e.g.,
// addi(x, 0) → x, neg(neg(x)) → x) as a single C function on the
// vtable. The canonicalize pass is the driver — it doesn't know what
// transformations exist, only how to invoke them.
iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CANONICALIZE_H_
