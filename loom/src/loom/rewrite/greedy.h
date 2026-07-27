// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Greedy fixed-point rewrite driver.
//
// This layer owns the generic mechanics of a rewrite session: rewriter
// lifetime, region fact setup, worklist seeding, fixed-point iteration, and
// result accounting. Passes provide the actual rewrite policy through
// callbacks, keeping dialect-specific canonicalization out of the shared
// driver.

#ifndef LOOM_REWRITE_GREEDY_H_
#define LOOM_REWRITE_GREEDY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS 10

typedef enum loom_greedy_rewrite_change_flag_bits_e {
  LOOM_GREEDY_REWRITE_CHANGE_FLAG_NONE = 0u,
  LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP = 1u << 0,
} loom_greedy_rewrite_change_flag_bits_t;
typedef uint32_t loom_greedy_rewrite_change_flags_t;

// Options for one greedy region run. Zero-initialized options use defaults.
typedef struct loom_greedy_rewrite_options_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;

  // Optional seed facts cloned into the driver-owned fact table before the
  // initial region analysis. Requires a driver with a value-fact owner.
  const loom_value_fact_table_t* seed_facts;

  // Optional constant materialization hook installed on the active rewriter.
  loom_materialize_constant_fn_t materialize_constant;
} loom_greedy_rewrite_options_t;

// Summary of one greedy rewrite run.
typedef struct loom_greedy_rewrite_result_t {
  // True if the driver changed IR by erasing, replacing, moving, creating, or
  // otherwise mutating an operation/value.
  bool changed;

  // True if incremental fact recomputation changed at least one stored value
  // fact during rewriting.
  bool facts_changed;

  // True if a rewrite changed at least one value type.
  bool types_changed;

  // Conservative boundary invalidation bit. True when summaries derived from
  // externally visible values may need recomputation.
  bool boundary_maybe_changed;

  // Number of ops modified by rewrite policy.
  int64_t ops_modified;
} loom_greedy_rewrite_result_t;

// Stateful greedy rewrite driver. The caller owns the module, scratch arena,
// and value-fact owner. The driver keeps successful run facts queryable until
// the next run, reset, or deinitialize.
typedef struct loom_greedy_rewrite_driver_t {
  // Module being transformed.
  loom_module_t* module;

  // Optional reusable value-fact storage used for region analysis.
  loom_pass_value_fact_owner_t* value_facts;

  // Caller-owned scratch arena for the active rewriter and pass callbacks.
  iree_arena_allocator_t* scratch_arena;

  // Borrowed facts from value_facts for the most recent successful run.
  loom_value_fact_table_t* latest_facts;

  // Rewriter state for the active run.
  loom_rewriter_t rewriter;

  // True when rewriter currently owns live worklist bits that must be cleared
  // before resetting scratch storage.
  bool rewriter_initialized;
} loom_greedy_rewrite_driver_t;

typedef iree_status_t (*loom_greedy_rewrite_prepare_region_fn_t)(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op);

typedef void (*loom_greedy_rewrite_cleanup_region_fn_t)(
    void* user_data, loom_greedy_rewrite_driver_t* driver);

typedef iree_status_t (*loom_greedy_rewrite_before_worklist_fn_t)(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_region_t* region, loom_greedy_rewrite_result_t* result,
    bool* out_changed);

typedef iree_status_t (*loom_greedy_rewrite_op_fn_t)(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed);

typedef void (*loom_greedy_rewrite_changed_fn_t)(
    void* user_data, loom_greedy_rewrite_driver_t* driver);

// Pass-provided hooks for a greedy rewrite run.
typedef struct loom_greedy_rewrite_callbacks_t {
  // Opaque callback state borrowed for the duration of the run.
  void* user_data;

  // Optional hook called after rewriter and facts are initialized, before the
  // first iteration seeds the worklist.
  loom_greedy_rewrite_prepare_region_fn_t prepare_region;

  // Optional hook called before returning from the run if prepare_region was
  // called. It must tolerate partially initialized user state when
  // prepare_region itself fails.
  loom_greedy_rewrite_cleanup_region_fn_t cleanup_region;

  // Optional hook called once per fixed-point iteration after worklist seeding
  // and before individual ops are popped.
  loom_greedy_rewrite_before_worklist_fn_t before_worklist;

  // Optional hook called for each popped live op.
  loom_greedy_rewrite_op_fn_t rewrite_op;

  // Optional hook called after a callback reports a rewrite change.
  loom_greedy_rewrite_changed_fn_t changed;
} loom_greedy_rewrite_callbacks_t;

// Initializes a reusable greedy driver. The scratch arena is reset by driver
// runs but remains owned by the caller.
void loom_greedy_rewrite_driver_initialize(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    loom_pass_value_fact_owner_t* value_facts,
    loom_greedy_rewrite_driver_t* out_driver);

// Clears active rewriter state, invalidates borrowed facts, and resets scratch.
void loom_greedy_rewrite_driver_reset(loom_greedy_rewrite_driver_t* driver);

// Releases active run state. Does not deinitialize the caller-owned scratch
// arena or value-fact owner.
void loom_greedy_rewrite_driver_deinitialize(
    loom_greedy_rewrite_driver_t* driver);

// Returns the fact table from the most recent successful run. The table is
// invalidated by the next driver run, reset, or deinitialize.
const loom_value_fact_table_t* loom_greedy_rewrite_driver_fact_table(
    const loom_greedy_rewrite_driver_t* driver);

// Records fact/type flags set on |rewriter| into |result|.
void loom_greedy_rewrite_result_record_rewriter_flags(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter);

// Records one IR rewrite and any fact/type flags set on |rewriter|.
void loom_greedy_rewrite_result_record_change(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter,
    loom_greedy_rewrite_change_flags_t flags);

// Runs greedy rewriting on an explicit region tree. |function| supplies the
// logical function context for value-fact inference and may be empty for
// detached regions. |parent_op| owns the root entry block arguments when
// provided.
iree_status_t loom_greedy_rewrite_run_region(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_greedy_rewrite_options_t* options,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* out_result);

// A simple kind-rooted rewrite pattern. Patterns are tried in caller-provided
// order for every matching op.
typedef struct loom_pattern_t {
  // Op kind this pattern matches.
  loom_op_kind_t root_kind;

  // Pattern callback. It reports a successful rewrite by mutating through the
  // rewriter, which sets LOOM_REWRITER_FLAG_CHANGED.
  iree_status_t (*match_and_rewrite)(const struct loom_pattern_t* pattern,
                                     loom_op_t* op, loom_rewriter_t* rewriter);
} loom_pattern_t;

typedef struct loom_rewrite_config_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;
} loom_rewrite_config_t;

// Runs simple pattern application on a function-like body region with an
// already-initialized greedy driver. This lets pass-specific drivers keep
// shared value-fact ownership while reusing the common pattern dispatch policy.
iree_status_t loom_greedy_rewrite_run_patterns(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    const loom_pattern_t* patterns, iree_host_size_t pattern_count,
    const loom_greedy_rewrite_options_t* options,
    loom_greedy_rewrite_result_t* out_result);

// Runs simple pattern application on a function-like body region.
iree_status_t loom_greedy_rewrite(iree_arena_allocator_t* arena,
                                  loom_module_t* module,
                                  loom_func_like_t function,
                                  const loom_pattern_t* patterns,
                                  iree_host_size_t pattern_count,
                                  const loom_rewrite_config_t* config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_GREEDY_H_
