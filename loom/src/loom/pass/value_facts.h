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

  // Optional target bundle for target-sensitive fact inference.
  const loom_target_bundle_t* target_bundle;
} loom_pass_value_fact_scope_t;

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_none(
    void) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_NONE,
  };
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_function(
    loom_func_like_t function) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION,
      /*.function=*/function,
  };
}

static inline loom_pass_value_fact_scope_t
loom_pass_value_fact_scope_function_for_target(
    loom_func_like_t function, const loom_target_bundle_t* target_bundle) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION,
      /*.function=*/function,
      /*.region=*/{},
      /*.parent_op=*/{},
      /*.target_bundle=*/target_bundle,
  };
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_region(
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_REGION,
      /*.function=*/function,
      /*.region=*/region,
      /*.parent_op=*/parent_op,
  };
}

static inline loom_pass_value_fact_scope_t
loom_pass_value_fact_scope_region_for_target(
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op,
    const loom_target_bundle_t* target_bundle) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_REGION,
      /*.function=*/function,
      /*.region=*/region,
      /*.parent_op=*/parent_op,
      /*.target_bundle=*/target_bundle,
  };
}

static inline loom_pass_value_fact_scope_t loom_pass_value_fact_scope_module(
    void) {
  return (loom_pass_value_fact_scope_t){
      /*.kind=*/LOOM_PASS_VALUE_FACT_SCOPE_MODULE,
  };
}

struct loom_pass_value_fact_owner_t {
  // Shared block pool used by owner arenas.
  iree_arena_block_pool_t* block_pool;
  // Arena for reusable direct-address entries and touched-value storage.
  iree_arena_allocator_t storage_arena;
  // Arena for scope-local extension payloads and inference scratch.
  iree_arena_allocator_t transient_arena;
  // Module whose value table is addressed by table entries.
  loom_module_t* module;
  // Reusable value-id-addressed fact table.
  loom_value_fact_table_t table;
  // Active populated fact scope, or NONE when table entries are not valid.
  loom_pass_value_fact_scope_t active_scope;
  // Owner state flags.
  loom_pass_value_fact_owner_flags_t flags;
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
    loom_pass_value_fact_owner_t* owner, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table);

// Acquires computed facts for |scope|. The returned table is borrowed and
// remains valid until the owner is invalidated, prepared or acquired for
// another scope, or deinitialized.
iree_status_t loom_pass_value_fact_owner_acquire(
    loom_pass_value_fact_owner_t* owner, loom_module_t* module,
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
