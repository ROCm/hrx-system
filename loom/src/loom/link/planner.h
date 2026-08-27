// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Metadata-first link planner.
//
// The planner consumes a provider-backed module index and produces an ordered
// live-symbol selection. Materialization, cloning, and definition contract
// merging remain the responsibility of the incremental linker sink.

#ifndef LOOM_LINK_PLANNER_H_
#define LOOM_LINK_PLANNER_H_

#include "iree/base/api.h"
#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_plan_t loom_link_plan_t;

// Planning mode.
typedef enum loom_link_plan_mode_e {
  // Merge every non-stripped INPUT symbol in stable provider order.
  LOOM_LINK_PLAN_MERGE = 0,
  // Select explicit roots or exported input roots and their reachable closure.
  LOOM_LINK_PLAN_LINK = 1,
} loom_link_plan_mode_t;

// Policy for references that cannot be selected by the current index.
typedef enum loom_link_plan_unresolved_policy_e {
  // Fail planning when a required symbol cannot be selected.
  LOOM_LINK_PLAN_UNRESOLVED_ERROR = 0,
  // Leave unresolved references to later verification/materialization.
  LOOM_LINK_PLAN_UNRESOLVED_ALLOW = 1,
} loom_link_plan_unresolved_policy_t;

// Policy for symbols that exist only for test or benchmark tooling.
typedef enum loom_link_plan_test_symbol_policy_e {
  // Keep test-only symbols.
  LOOM_LINK_PLAN_TEST_SYMBOL_KEEP = 0,
  // Strip test-only symbols before materialization.
  LOOM_LINK_PLAN_TEST_SYMBOL_STRIP = 1,
} loom_link_plan_test_symbol_policy_t;

// Policy for projecting dependencies reached through symbol references.
typedef enum loom_link_plan_dependency_policy_e {
  // Retain complete referenced symbols for ordinary source composition.
  LOOM_LINK_PLAN_DEPENDENCY_COMPLETE = 0,
  // Preserve the exact semantic interfaces requested by each reference.
  LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS = 1,
} loom_link_plan_dependency_policy_t;

// Reason a planned symbol is live.
typedef enum loom_link_plan_live_reason_e {
  // Selected because merge mode includes every linkable INPUT symbol.
  LOOM_LINK_PLAN_LIVE_MERGE = 0,
  // Selected because the user or exported-root policy named it as a root.
  LOOM_LINK_PLAN_LIVE_ROOT = 1,
  // Selected because another live symbol references it.
  LOOM_LINK_PLAN_LIVE_DEPENDENCY = 2,
  // Selected explicitly after template specialization chose this provider.
  LOOM_LINK_PLAN_LIVE_PROVIDER = 3,
} loom_link_plan_live_reason_t;

// Optional strip filter. Returning true removes |symbol| from merge mode and
// rejects link references to it unless unresolved references are allowed.
typedef bool (*loom_link_plan_strip_symbol_fn_t)(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// One identity-resolved link root facet. The symbol's primary
// contract/definition facet is always selected with the requested facet.
typedef struct loom_link_plan_root_facet_t {
  // Index-wide symbol ordinal resolved by the requesting subsystem.
  iree_host_size_t symbol_ordinal;
  // Semantic facet requested by the linked product.
  loom_link_symbol_facet_kind_t kind;
} loom_link_plan_root_facet_t;

// Options controlling one planning operation.
typedef struct loom_link_plan_options_t {
  // Planning mode. Zero defaults to MERGE.
  loom_link_plan_mode_t mode;
  // Explicit roots for LINK mode. Names may include a leading '@'.
  iree_string_view_list_t root_symbols;
  // Select exported INPUT-provider symbols as roots in LINK mode.
  bool include_input_exports;
  // Unresolved reference handling. Zero defaults to ERROR.
  loom_link_plan_unresolved_policy_t unresolved_policy;
  // Test-only symbol handling. Zero defaults to KEEP.
  loom_link_plan_test_symbol_policy_t test_symbol_policy;
  // Optional strip filter.
  loom_link_plan_strip_symbol_fn_t strip_symbol;
  // User data passed to strip_symbol.
  void* strip_symbol_user_data;
  // Exact template providers already chosen by specialization. These are
  // additional link roots whose ordinary dependencies and nested
  // template-family demands participate in the same closure.
  struct {
    // Number of index-wide symbol ordinals.
    iree_host_size_t count;
    // Index-wide template.def/template.ukernel symbol ordinals.
    const iree_host_size_t* values;
  } selected_template_providers;
  // Identity-resolved facet roots for LINK mode.
  struct {
    // Number of root facet requests.
    iree_host_size_t count;
    // Root facet requests in caller-defined stable order.
    const loom_link_plan_root_facet_t* values;
  } root_facets;
  // Dependency projection policy. Zero defaults to complete source symbols.
  loom_link_plan_dependency_policy_t dependency_policy;
  // Select exports from specific indexed providers as roots in LINK mode.
  struct {
    // Number of index-wide provider ordinals.
    iree_host_size_t count;
    // Index-wide provider ordinals in caller-defined stable order.
    const iree_host_size_t* values;
  } root_providers;
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
  // Plan-local ordinal of the selected primary contract/definition facet.
  iree_host_size_t primary_facet_ordinal;
  // Number of selected semantic facets.
  uint16_t selected_facet_count;
  // Root name that caused this selection when reason is ROOT.
  iree_string_view_t root_name;
} loom_link_plan_symbol_t;

// One live structural symbol facet in a link plan.
typedef struct loom_link_plan_facet_t {
  // Plan-local facet ordinal.
  iree_host_size_t ordinal;
  // Plan-local symbol selection owning this facet.
  iree_host_size_t symbol_plan_ordinal;
  // Index-wide symbol ordinal owning this facet.
  iree_host_size_t symbol_ordinal;
  // Selected semantic facet.
  loom_link_symbol_facet_kind_t kind;
  // Why this facet is live.
  loom_link_plan_live_reason_t reason;
  // Plan-local facet ordinal that caused this selection, or INVALID_ORDINAL.
  iree_host_size_t cause_ordinal;
} loom_link_plan_facet_t;

// Builds a link plan from |index|.
//
// The caller owns the returned plan and must release it with
// loom_link_plan_free().
iree_status_t loom_link_plan_build(const loom_link_module_index_t* index,
                                   const loom_link_plan_options_t* options,
                                   iree_allocator_t allocator,
                                   loom_link_plan_t** out_plan);

// Releases |plan|.
void loom_link_plan_free(loom_link_plan_t* plan);

// Returns the index this plan was built from.
const loom_link_module_index_t* loom_link_plan_index(
    const loom_link_plan_t* plan);

// Returns the mode that selected |plan|.
loom_link_plan_mode_t loom_link_plan_mode(const loom_link_plan_t* plan);

// Returns the number of live symbol selections.
iree_host_size_t loom_link_plan_symbol_count(const loom_link_plan_t* plan);

// Returns live symbol selection |ordinal|, or NULL if out of range.
const loom_link_plan_symbol_t* loom_link_plan_symbol_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal);

// Returns the number of explicit structural facet selections in a link plan.
// Merge plans retain complete symbols without enumerating facets.
iree_host_size_t loom_link_plan_facet_count(const loom_link_plan_t* plan);

// Returns live facet selection |ordinal|, or NULL if out of range.
const loom_link_plan_facet_t* loom_link_plan_facet_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal);

// Returns the number of reachable template-family demands.
iree_host_size_t loom_link_plan_demanded_template_family_count(
    const loom_link_plan_t* plan);

// Returns the number of reachable template-demand occurrences. Repeated
// demands for the same family remain distinct.
iree_host_size_t loom_link_plan_template_demand_occurrence_count(
    const loom_link_plan_t* plan);

// Returns demanded template-family ordinal |ordinal|, or INVALID when out of
// range. Each family appears once in first-reachable order.
loom_link_template_family_ordinal_t loom_link_plan_demanded_template_family_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal);

// Returns true when |symbol_ordinal| is live in |plan|.
bool loom_link_plan_contains_symbol(const loom_link_plan_t* plan,
                                    iree_host_size_t symbol_ordinal);

// Returns true when semantic |kind| of |symbol_ordinal| is live in |plan|.
// Every valid facet of a merge-selected symbol is live.
bool loom_link_plan_contains_facet(const loom_link_plan_t* plan,
                                   iree_host_size_t symbol_ordinal,
                                   loom_link_symbol_facet_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_PLANNER_H_
