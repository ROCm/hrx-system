// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Metadata-first link planner.
//
// The planner consumes a provider-backed module index and produces an ordered
// live-symbol selection. Materialization and cloning remain the responsibility
// of the incremental linker sink.

#ifndef LOOM_LINK_PLANNER_H_
#define LOOM_LINK_PLANNER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_plan_t loom_link_plan_t;

// Planning mode.
typedef enum loom_link_plan_mode_e {
  // Select every non-stripped symbol in stable provider order.
  LOOM_LINK_PLAN_ARCHIVE = 0,
  // Select explicit roots or exported roots and their reachable closure.
  LOOM_LINK_PLAN_SELECTIVE = 1,
} loom_link_plan_mode_t;

// Policy for references that cannot be selected by the current index.
typedef enum loom_link_plan_unresolved_policy_e {
  // Fail planning when a required symbol cannot be selected.
  LOOM_LINK_PLAN_UNRESOLVED_ERROR = 0,
  // Leave unresolved references to later verification/materialization.
  LOOM_LINK_PLAN_UNRESOLVED_ALLOW = 1,
} loom_link_plan_unresolved_policy_t;

// Policy for check dialect testbench symbols.
typedef enum loom_link_plan_check_policy_e {
  // Keep check.case and check.benchmark symbols.
  LOOM_LINK_PLAN_CHECK_KEEP = 0,
  // Strip check.case and check.benchmark symbols before materialization.
  LOOM_LINK_PLAN_CHECK_STRIP = 1,
} loom_link_plan_check_policy_t;

// Reason a planned symbol is live.
typedef enum loom_link_plan_live_reason_e {
  // Selected because archive mode includes every linkable symbol.
  LOOM_LINK_PLAN_LIVE_ARCHIVE = 0,
  // Selected because the user or exported-root policy named it as a root.
  LOOM_LINK_PLAN_LIVE_ROOT = 1,
  // Selected because another live symbol references it.
  LOOM_LINK_PLAN_LIVE_DEPENDENCY = 2,
  // Selected because another live symbol contains a func.apply for the
  // provider's implementation contract.
  LOOM_LINK_PLAN_LIVE_PROVIDER = 3,
} loom_link_plan_live_reason_t;

// Optional strip filter. Returning true removes |symbol| from archive mode and
// rejects selective references to it unless unresolved references are allowed.
typedef bool (*loom_link_plan_strip_symbol_fn_t)(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Optional materializer for dependency scanning of indexed modules whose IR is
// not already materialized. Implementations must keep |out_module| alive until
// loom_link_plan_build returns.
typedef iree_status_t (*loom_link_plan_materialize_module_fn_t)(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module,
    const loom_module_t** out_module);

// Options controlling one planning operation.
typedef struct loom_link_plan_options_t {
  // Planning mode. Zero defaults to ARCHIVE.
  loom_link_plan_mode_t mode;
  // Explicit roots for SELECTIVE mode. Names may include a leading '@'.
  iree_string_view_list_t root_symbols;
  // Select all exported symbols as roots in SELECTIVE mode.
  bool include_exported_roots;
  // Unresolved reference handling. Zero defaults to ERROR.
  loom_link_plan_unresolved_policy_t unresolved_policy;
  // Check dialect symbol handling. Zero defaults to KEEP.
  loom_link_plan_check_policy_t check_policy;
  // Optional strip filter.
  loom_link_plan_strip_symbol_fn_t strip_symbol;
  // User data passed to strip_symbol.
  void* strip_symbol_user_data;
  // Optional on-demand materializer for unmaterialized indexed modules.
  loom_link_plan_materialize_module_fn_t materialize_module;
  // User data passed to materialize_module.
  void* materialize_module_user_data;
} loom_link_plan_options_t;

// One live symbol selection in a plan.
typedef struct loom_link_plan_symbol_t {
  // Plan-local selection ordinal.
  iree_host_size_t ordinal;
  // Index symbol ordinal selected by this entry.
  iree_host_size_t symbol_ordinal;
  // Why this symbol is live.
  loom_link_plan_live_reason_t reason;
  // Plan-local ordinal that caused this dependency, or INVALID_ORDINAL.
  iree_host_size_t cause_ordinal;
  // Root name that caused this selection when reason is ROOT.
  iree_string_view_t root_name;
} loom_link_plan_symbol_t;

// Builds a link plan from |index|.
//
// |block_pool| backs planner-owned dependency tables. The caller owns the
// returned plan and must release it with loom_link_plan_free().
iree_status_t loom_link_plan_build(const loom_link_module_index_t* index,
                                   const loom_link_plan_options_t* options,
                                   iree_arena_block_pool_t* block_pool,
                                   iree_allocator_t allocator,
                                   loom_link_plan_t** out_plan);

// Releases |plan|.
void loom_link_plan_free(loom_link_plan_t* plan);

// Returns the index this plan was built from.
const loom_link_module_index_t* loom_link_plan_index(
    const loom_link_plan_t* plan);

// Returns the number of live symbol selections.
iree_host_size_t loom_link_plan_symbol_count(const loom_link_plan_t* plan);

// Returns live symbol selection |ordinal|, or NULL if out of range.
const loom_link_plan_symbol_t* loom_link_plan_symbol_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal);

// Returns true when |symbol_ordinal| is live in |plan|.
bool loom_link_plan_contains_symbol(const loom_link_plan_t* plan,
                                    iree_host_size_t symbol_ordinal);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_PLANNER_H_
