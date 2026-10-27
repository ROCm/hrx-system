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
#include "loom/rewrite/action.h"
#include "loom/rewrite/pattern_registry.h"
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

  // Optional constant materialization policy installed on the active rewriter.
  loom_constant_materializer_t constant_materializer;

  // Optional consumer for relations retained before an exact fold.
  loom_rewriter_pending_exact_relations_callback_t
      pending_exact_relations_callback;

  // Borrowed function-scoped target math policy used for optional rewrites.
  const struct loom_target_math_policy_t* math_policy;

  // Resolved target bundle paired with |math_policy|.
  const struct loom_target_bundle_t* math_target_bundle;

  // Optional rooted patterns prepared once from coherent facts, then applied
  // as a batch before fixed-point rewriting. Plans are released immediately.
  const loom_rewrite_pattern_registry_t* prepared_patterns;
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
// and optional precomputed fact table. Rewrites incrementally maintain the
// borrowed facts but never acquire, clear, or otherwise own their scope.
typedef struct loom_greedy_rewrite_driver_t {
  // Module being transformed.
  loom_module_t* module;

  // Caller-owned scratch arena for the active rewriter and pass callbacks.
  iree_arena_allocator_t* scratch_arena;

  // Optional caller-computed facts incrementally maintained by rewrites.
  loom_value_fact_table_t* fact_table;

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

  // Optional hook called once per fixed-point iteration after any
  // fact-independent rewrite batch has completed and the worklist has been
  // reseeded. Pending facts are published first.
  loom_greedy_rewrite_before_worklist_fn_t before_worklist;

  // Optional fact-independent rewrite hook. When present, the driver drains a
  // complete worklist through this hook before publishing the mutations as one
  // batch and reseeding for |rewrite_op|. The callback must not observe value
  // facts; mutation APIs continue maintaining the attached fact table.
  loom_greedy_rewrite_op_fn_t rewrite_op_without_facts;

  // Optional hook called for each popped live op with pending facts published.
  // A callback that mutates and then reads derived facts must publish those
  // mutations with loom_rewriter_propagate_value_facts before reading.
  loom_greedy_rewrite_op_fn_t rewrite_op;

  // Optional hook called after a callback reports a rewrite change.
  loom_greedy_rewrite_changed_fn_t changed;
} loom_greedy_rewrite_callbacks_t;

// Initializes a reusable greedy driver. The scratch arena is reset by driver
// runs but remains owned by the caller.
void loom_greedy_rewrite_driver_initialize(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    loom_value_fact_table_t* fact_table,
    loom_greedy_rewrite_driver_t* out_driver);

// Replaces the caller-owned fact table used by subsequent runs. The driver must
// not have an active run. NULL disables incremental fact maintenance.
void loom_greedy_rewrite_driver_set_fact_table(
    loom_greedy_rewrite_driver_t* driver, loom_value_fact_table_t* fact_table);

// Clears active rewriter state and resets scratch. Borrowed facts remain valid.
void loom_greedy_rewrite_driver_reset(loom_greedy_rewrite_driver_t* driver);

// Releases active run state. Does not deinitialize the caller-owned scratch
// arena or fact table.
void loom_greedy_rewrite_driver_deinitialize(
    loom_greedy_rewrite_driver_t* driver);

// Returns the caller-owned fact table incrementally maintained by the driver.
const loom_value_fact_table_t* loom_greedy_rewrite_driver_fact_table(
    const loom_greedy_rewrite_driver_t* driver);

// Records fact/type flags set on |rewriter| into |result|.
void loom_greedy_rewrite_result_record_rewriter_flags(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter);

// Records one IR rewrite and any fact/type flags set on |rewriter|.
void loom_greedy_rewrite_result_record_change(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter,
    loom_greedy_rewrite_change_flags_t flags);

// Publishes pending facts, prepares one action per matching root, then applies
// the complete batch in reverse preorder without intervening fact publication.
// All decisions are made before the first mutation; preparation failure leaves
// IR unchanged. Action storage is released before returning. The active
// driver's next fact observer must publish the resulting mutations as usual.
iree_status_t loom_greedy_rewrite_apply_prepared(
    loom_greedy_rewrite_driver_t* driver, loom_region_t* region,
    loom_rewrite_prepare_callback_t prepare,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* result);

// Runs greedy rewriting on an explicit region tree. |function| and |parent_op|
// describe the region to policy callbacks; the driver does not rediscover or
// recompute facts for the region. Successful return publishes all pending
// equations before detaching the rewriter. On failure the borrowed table may
// contain partial incremental updates and its owner must invalidate it before
// reuse.
iree_status_t loom_greedy_rewrite_run_region(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_greedy_rewrite_options_t* options,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* out_result);

typedef struct loom_rewrite_config_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;
} loom_rewrite_config_t;

// Runs indexed pattern application on a function-like body region with an
// already-initialized greedy driver. This lets pass-specific drivers keep
// shared value-fact ownership while reusing the common pattern dispatch policy.
iree_status_t loom_greedy_rewrite_run_patterns(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    const loom_rewrite_pattern_registry_t* pattern_registry,
    void* pattern_context, const loom_greedy_rewrite_options_t* options,
    loom_greedy_rewrite_result_t* out_result);

// Runs indexed pattern application on a function-like body region.
iree_status_t loom_greedy_rewrite(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_func_like_t function,
    const loom_rewrite_pattern_registry_t* pattern_registry,
    void* pattern_context, const loom_rewrite_config_t* config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_GREEDY_H_
