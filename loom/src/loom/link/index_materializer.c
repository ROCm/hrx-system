// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/index_materializer.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/ir/module.h"
#include "loom/link/template_candidate_loader.h"
#include "loom/transforms/symbol/template_selection.h"

typedef struct loom_link_index_selected_providers_t {
  // Exact selected index symbol ordinals in first-selection order.
  struct {
    // Allocator-owned ordinal storage sized to the index symbol count.
    iree_host_size_t* values;
    // Number of selected provider ordinals.
    iree_host_size_t count;
    // Maximum number of selected provider ordinals.
    iree_host_size_t capacity;
  } ordinals;
  // Packed membership indexed by index-wide symbol ordinal.
  struct {
    // Allocator-owned membership words.
    uint64_t* values;
    // Number of symbol ordinals represented by values.
    iree_host_size_t symbol_count;
  } membership;
  // Host allocator owning both dense arrays.
  iree_allocator_t allocator;
} loom_link_index_selected_providers_t;

static iree_status_t loom_link_index_selected_providers_initialize(
    iree_host_size_t symbol_count, iree_allocator_t allocator,
    loom_link_index_selected_providers_t* out_selected) {
  *out_selected = (loom_link_index_selected_providers_t){
      .ordinals = {.capacity = symbol_count},
      .membership = {.symbol_count = symbol_count},
      .allocator = allocator,
  };
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        allocator, symbol_count, sizeof(*out_selected->ordinals.values),
        (void**)&out_selected->ordinals.values));
  }
  iree_host_size_t rounded_symbol_count = 0;
  if (!iree_host_size_checked_add(symbol_count, 63, &rounded_symbol_count)) {
    iree_allocator_free(allocator, out_selected->ordinals.values);
    *out_selected = (loom_link_index_selected_providers_t){0};
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected provider bitmap size overflow");
  }
  const iree_host_size_t word_count = rounded_symbol_count / 64;
  if (word_count != 0) {
    iree_status_t status = iree_allocator_malloc_array(
        allocator, word_count, sizeof(*out_selected->membership.values),
        (void**)&out_selected->membership.values);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, out_selected->ordinals.values);
      *out_selected = (loom_link_index_selected_providers_t){0};
      return status;
    }
  }
  return iree_ok_status();
}

static void loom_link_index_selected_providers_deinitialize(
    loom_link_index_selected_providers_t* selected) {
  iree_allocator_free(selected->allocator, selected->membership.values);
  iree_allocator_free(selected->allocator, selected->ordinals.values);
  *selected = (loom_link_index_selected_providers_t){0};
}

static bool loom_link_index_selected_providers_contains(
    const loom_link_index_selected_providers_t* selected,
    iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < selected->membership.symbol_count);
  return (selected->membership.values[symbol_ordinal >> 6] &
          (UINT64_C(1) << (symbol_ordinal & 63u))) != 0;
}

static void loom_link_index_selected_providers_append(
    loom_link_index_selected_providers_t* selected,
    iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < selected->membership.symbol_count);
  IREE_ASSERT(
      !loom_link_index_selected_providers_contains(selected, symbol_ordinal));
  IREE_ASSERT(selected->ordinals.count < selected->ordinals.capacity);
  selected->membership.values[symbol_ordinal >> 6] |= UINT64_C(1)
                                                      << (symbol_ordinal & 63u);
  selected->ordinals.values[selected->ordinals.count++] = symbol_ordinal;
}

static loom_link_template_provider_membership_t
loom_link_index_selected_provider_membership(
    const loom_link_index_selected_providers_t* selected) {
  return (loom_link_template_provider_membership_t){
      .words = selected->membership.values,
      .symbol_count = selected->membership.symbol_count,
  };
}

static void loom_link_index_apply_selected_providers(
    const loom_link_index_selected_providers_t* selected,
    loom_link_plan_options_t* options) {
  options->selected_template_providers.count = selected->ordinals.count;
  options->selected_template_providers.values = selected->ordinals.values;
}

static loom_template_selection_mode_t loom_link_index_selection_mode(
    const loom_link_plan_options_t* options) {
  return options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW
             ? LOOM_TEMPLATE_SELECTION_MODE_EARLY
             : LOOM_TEMPLATE_SELECTION_MODE_FINAL;
}

static iree_status_t loom_link_index_query_candidates(
    const loom_link_module_index_t* index,
    loom_link_template_candidate_loader_t* candidate_loader,
    const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    const loom_link_index_selected_providers_t* selected,
    const loom_link_plan_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    loom_template_selection_query_result_t* out_result) {
  loom_template_provider_slice_t external_candidates =
      loom_template_provider_slice_empty();
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_loader_project(
      candidate_loader, plan, materialization,
      loom_link_index_selected_provider_membership(selected), arena,
      &external_candidates));

  loom_symbol_fact_table_t fact_table;
  loom_symbol_fact_table_initialize(&fact_table, arena);
  loom_template_provider_catalog_t catalog;
  loom_template_provider_catalog_initialize(&catalog, arena);
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_build(
      &catalog, materialization->module, &fact_table,
      external_candidates.providers, external_candidates.count));
  const loom_template_selection_query_options_t query_options = {
      .mode = loom_link_index_selection_mode(options),
      .catalog = &catalog,
      .origin_count = loom_link_module_index_symbol_count(index),
  };
  return loom_template_selection_query(materialization->module, &query_options,
                                       block_pool, arena, out_result);
}

static bool loom_link_index_append_query_selections(
    const loom_template_selection_query_result_t* query,
    loom_link_index_selected_providers_t* selected) {
  const iree_host_size_t prior_count = selected->ordinals.count;
  for (iree_host_size_t i = 0; i < query->selected_origins.count; ++i) {
    const iree_host_size_t symbol_ordinal = query->selected_origins.values[i];
    IREE_ASSERT(symbol_ordinal < selected->membership.symbol_count);
    if (!loom_link_index_selected_providers_contains(selected,
                                                     symbol_ordinal)) {
      loom_link_index_selected_providers_append(selected, symbol_ordinal);
    }
  }
  return selected->ordinals.count != prior_count;
}

static iree_status_t loom_link_index_build_plan(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* base_options,
    const loom_link_index_selected_providers_t* selected,
    iree_allocator_t allocator, loom_link_plan_t** out_plan) {
  loom_link_plan_options_t options = *base_options;
  loom_link_index_apply_selected_providers(selected, &options);
  return loom_link_plan_build(index, &options, allocator, out_plan);
}

static iree_status_t loom_link_index_materialize_selective(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_string_view_list_t output_roots,
    loom_link_index_materialization_t* out_materialization) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  loom_link_index_selected_providers_t selected;
  IREE_RETURN_IF_ERROR(loom_link_index_selected_providers_initialize(
      symbol_count, environment->allocator, &selected));
  loom_link_template_candidate_loader_t* candidate_loader = NULL;
  iree_status_t status = loom_link_template_candidate_loader_create(
      index, environment, &candidate_loader);

  loom_link_plan_t* stable_plan = NULL;
  while (iree_status_is_ok(status) && stable_plan == NULL) {
    loom_link_plan_t* plan = NULL;
    status = loom_link_index_build_plan(index, plan_options, &selected,
                                        environment->allocator, &plan);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (loom_link_plan_demanded_template_family_count(plan) == 0) {
      stable_plan = plan;
      break;
    }

    iree_arena_allocator_t iteration_arena;
    iree_arena_initialize(environment->block_pool, &iteration_arena);
    loom_link_plan_materialization_t analysis = {0};
    status =
        loom_link_plan_materialize(plan, environment, module_name, output_roots,
                                   &iteration_arena, &analysis);
    loom_template_selection_query_result_t query = {0};
    if (iree_status_is_ok(status)) {
      status = loom_link_index_query_candidates(
          index, candidate_loader, plan, &analysis, &selected, plan_options,
          environment->block_pool, &iteration_arena, &query);
    }
    const bool changed =
        iree_status_is_ok(status) &&
        loom_link_index_append_query_selections(&query, &selected);
    loom_module_free(analysis.module);
    iree_arena_deinitialize(&iteration_arena);
    if (!iree_status_is_ok(status)) {
      loom_link_plan_free(plan);
      break;
    }
    if (changed) {
      loom_link_plan_free(plan);
    } else {
      stable_plan = plan;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_arena_allocator_t final_arena;
    iree_arena_initialize(environment->block_pool, &final_arena);
    loom_link_plan_materialization_t final = {0};
    status = loom_link_plan_materialize(stable_plan, environment, module_name,
                                        output_roots, &final_arena, &final);
    if (iree_status_is_ok(status)) {
      out_materialization->plan = stable_plan;
      out_materialization->module = final.module;
      stable_plan = NULL;
    } else {
      loom_module_free(final.module);
    }
    iree_arena_deinitialize(&final_arena);
  }

  loom_link_plan_free(stable_plan);
  loom_link_template_candidate_loader_free(candidate_loader);
  loom_link_index_selected_providers_deinitialize(&selected);
  return status;
}

static iree_status_t loom_link_index_materialize_archive(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_string_view_list_t output_roots,
    loom_link_index_materialization_t* out_materialization) {
  loom_link_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_plan_build(index, plan_options, environment->allocator, &plan));
  iree_arena_allocator_t arena;
  iree_arena_initialize(environment->block_pool, &arena);
  loom_link_plan_materialization_t result = {0};
  iree_status_t status = loom_link_plan_materialize(
      plan, environment, module_name, output_roots, &arena, &result);
  if (iree_status_is_ok(status)) {
    out_materialization->plan = plan;
    out_materialization->module = result.module;
    plan = NULL;
  } else {
    loom_module_free(result.module);
  }
  iree_arena_deinitialize(&arena);
  loom_link_plan_free(plan);
  return status;
}

void loom_link_index_materialization_deinitialize(
    loom_link_index_materialization_t* materialization) {
  if (materialization == NULL) {
    return;
  }
  loom_module_free(materialization->module);
  loom_link_plan_free(materialization->plan);
  *materialization = (loom_link_index_materialization_t){0};
}

iree_status_t loom_link_index_materialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_string_view_list_t output_roots,
    loom_link_index_materialization_t* out_materialization) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(plan_options);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = (loom_link_index_materialization_t){0};
  if (plan_options->selected_template_providers.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "index materialization owns template provider selection roots");
  }
  switch (plan_options->mode) {
    case LOOM_LINK_PLAN_ARCHIVE:
      return loom_link_index_materialize_archive(
          index, plan_options, environment, module_name, output_roots,
          out_materialization);
    case LOOM_LINK_PLAN_SELECTIVE:
      return loom_link_index_materialize_selective(
          index, plan_options, environment, module_name, output_roots,
          out_materialization);
  }
  IREE_ASSERT_UNREACHABLE("unknown link plan mode");
  IREE_BUILTIN_UNREACHABLE();
}
