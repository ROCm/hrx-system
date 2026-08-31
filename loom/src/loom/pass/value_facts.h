// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scoped value-fact ownership for pass pipeline execution.
//
// The owner is production execution state, not a target capability and not a
// test helper. It owns reusable value-id-addressed storage while making fact
// population explicit by scope: function passes compute function facts only;
// module-scope facts are requested only by module algorithms that need them.

#ifndef LOOM_PASS_VALUE_FACTS_H_
#define LOOM_PASS_VALUE_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_pass_value_fact_scope_kind_e {
  LOOM_PASS_VALUE_FACT_SCOPE_NONE = 0,
  LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION = 1,
  LOOM_PASS_VALUE_FACT_SCOPE_MODULE = 2,
  LOOM_PASS_VALUE_FACT_SCOPE_REGION = 3,
} loom_pass_value_fact_scope_kind_t;

typedef enum loom_pass_value_fact_owner_flag_bits_e {
  LOOM_PASS_VALUE_FACT_OWNER_FLAG_TABLE_INITIALIZED = 1u << 0,
} loom_pass_value_fact_owner_flag_bits_t;
typedef uint32_t loom_pass_value_fact_owner_flags_t;

// Optional deterministic counts for one observed value-fact owner interval.
// The owner checks for and updates these only at lifecycle transitions while
// lifecycle_counts is non-NULL; ordinary pass execution performs no counter
// updates.
typedef struct loom_pass_value_fact_lifecycle_counts_t {
  // Number of computed-fact acquisition requests.
  uint64_t acquisition_count;
  // Number of acquisitions served by the already-active scope.
  uint64_t cache_hit_count;
  // Number of complete scope computations attempted after cache misses.
  uint64_t recomputation_count;
  // Number of empty scopes prepared for caller-maintained facts.
  uint64_t preparation_count;
  // Number of explicit invalidation requests against an initialized table.
  uint64_t invalidation_count;
  // Number of populated scopes actually cleared for any lifecycle reason.
  uint64_t scope_clear_count;
  // Total number of SSA value entries produced by complete computations.
  uint64_t computed_value_count;
  // Total number of populated SSA value entries discarded by scope clears.
  uint64_t cleared_value_count;
} loom_pass_value_fact_lifecycle_counts_t;

typedef struct loom_pass_value_fact_scope_t {
  // Requested fact population scope.
  loom_pass_value_fact_scope_kind_t kind;

  // Function context for FUNCTION and REGION scopes. REGION scopes may leave
  // this empty when analyzing detached IR, but projected func-like regions
  // should provide the owning function so op fact inference can query it.
  loom_func_like_t function;

  // Region root for LOOM_PASS_VALUE_FACT_SCOPE_REGION.
  loom_region_t* region;

  // Op that owns the region root for LOOM_PASS_VALUE_FACT_SCOPE_REGION.
  loom_op_t* parent_op;

  // Optional immutable target facts for target-sensitive fact inference.
  const loom_target_facts_t* target_facts;
} loom_pass_value_fact_scope_t;

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_none(
    void) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_NONE;
  return scope;
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_function(
    loom_func_like_t function) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION;
  scope.function = function;
  return scope;
}

static inline loom_pass_value_fact_scope_t
loom_pass_value_fact_scope_function_for_target(
    loom_func_like_t function, const loom_target_facts_t* target_facts) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION;
  scope.function = function;
  scope.target_facts = target_facts;
  return scope;
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_region(
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_REGION;
  scope.function = function;
  scope.region = region;
  scope.parent_op = parent_op;
  return scope;
}

static inline loom_pass_value_fact_scope_t
loom_pass_value_fact_scope_region_for_target(
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op,
    const loom_target_facts_t* target_facts) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_REGION;
  scope.function = function;
  scope.region = region;
  scope.parent_op = parent_op;
  scope.target_facts = target_facts;
  return scope;
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_module(
    void) {
  loom_pass_value_fact_scope_t scope = {LOOM_PASS_VALUE_FACT_SCOPE_NONE};
  scope.kind = LOOM_PASS_VALUE_FACT_SCOPE_MODULE;
  return scope;
}

struct loom_pass_value_fact_owner_t {
  // Shared block pool used by owner arenas.
  iree_arena_block_pool_t* block_pool;
  // Arena for reusable direct-address entries and touched-value storage.
  iree_arena_allocator_t storage_arena;
  // Arena for scope-local extension payloads and inference scratch.
  iree_arena_allocator_t transient_arena;
  // Module whose value table is addressed by table entries.
  const loom_module_t* module;
  // Reusable value-id-addressed fact table.
  loom_value_fact_table_t table;
  // Active populated fact scope, or NONE when table entries are not valid.
  loom_pass_value_fact_scope_t active_scope;
  // Owner state flags.
  loom_pass_value_fact_owner_flags_t flags;
  // Optional borrowed counters updated at owner lifecycle transitions.
  loom_pass_value_fact_lifecycle_counts_t* lifecycle_counts;
};

// Initializes a dormant owner. This performs no fact-table allocation and no
// IR walk; storage is allocated only when a caller acquires facts.
void loom_pass_value_fact_owner_initialize(
    iree_arena_block_pool_t* block_pool,
    loom_pass_value_fact_owner_t* out_owner);

// Deinitializes all owner storage.
void loom_pass_value_fact_owner_deinitialize(
    loom_pass_value_fact_owner_t* owner);

// Invalidates the active scope and clears populated entries. This keeps
// reusable direct-address storage when capacity still matches the module.
void loom_pass_value_fact_owner_invalidate(loom_pass_value_fact_owner_t* owner);

// Prepares empty fact storage for |scope|. The returned table is borrowed and
// may be populated by the caller through normal fact-table APIs. Use this for
// production algorithms such as rewriting that must compute and incrementally
// maintain facts while they mutate IR. If population fails, the caller must
// invalidate the owner before returning the failure.
iree_status_t loom_pass_value_fact_owner_prepare(
    loom_pass_value_fact_owner_t* owner, const loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table);

// Acquires computed facts for |scope|. The returned table is borrowed and
// remains valid until the owner is invalidated, prepared or acquired for
// another scope, or deinitialized.
iree_status_t loom_pass_value_fact_owner_acquire(
    loom_pass_value_fact_owner_t* owner, const loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table);

// Prepares scoped fact storage through a pass invocation.
iree_status_t loom_pass_value_facts_prepare(
    loom_pass_t* pass, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table);

// Acquires scoped facts through a pass invocation.
iree_status_t loom_pass_value_facts_acquire(
    loom_pass_t* pass, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_VALUE_FACTS_H_
