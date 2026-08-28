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

typedef struct loom_link_index_provider_roots_t {
  // Required index symbol ordinals in first-requirement order.
  struct {
    // Allocator-owned ordinal storage sized to the index symbol count.
    iree_host_size_t* values;
    // Number of required provider ordinals.
    iree_host_size_t count;
    // Maximum number of required provider ordinals.
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
} loom_link_index_provider_roots_t;

static iree_status_t loom_link_index_provider_roots_initialize(
    iree_host_size_t symbol_count, iree_allocator_t allocator,
    loom_link_index_provider_roots_t* out_roots) {
  *out_roots = (loom_link_index_provider_roots_t){
      .ordinals = {.capacity = symbol_count},
      .membership = {.symbol_count = symbol_count},
      .allocator = allocator,
  };
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        allocator, symbol_count, sizeof(*out_roots->ordinals.values),
        (void**)&out_roots->ordinals.values));
  }
  iree_host_size_t rounded_symbol_count = 0;
  if (!iree_host_size_checked_add(symbol_count, 63, &rounded_symbol_count)) {
    iree_allocator_free(allocator, out_roots->ordinals.values);
    *out_roots = (loom_link_index_provider_roots_t){0};
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template provider root bitmap size overflow");
  }
  const iree_host_size_t word_count = rounded_symbol_count / 64;
  if (word_count != 0) {
    iree_status_t status = iree_allocator_malloc_array(
        allocator, word_count, sizeof(*out_roots->membership.values),
        (void**)&out_roots->membership.values);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, out_roots->ordinals.values);
      *out_roots = (loom_link_index_provider_roots_t){0};
      return status;
    }
  }
  return iree_ok_status();
}

static void loom_link_index_provider_roots_deinitialize(
    loom_link_index_provider_roots_t* roots) {
  iree_allocator_free(roots->allocator, roots->membership.values);
  iree_allocator_free(roots->allocator, roots->ordinals.values);
  *roots = (loom_link_index_provider_roots_t){0};
}

static bool loom_link_index_provider_roots_contains(
    const loom_link_index_provider_roots_t* roots,
    iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < roots->membership.symbol_count);
  return (roots->membership.values[symbol_ordinal >> 6] &
          (UINT64_C(1) << (symbol_ordinal & 63u))) != 0;
}

static void loom_link_index_provider_roots_append(
    loom_link_index_provider_roots_t* roots, iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < roots->membership.symbol_count);
  IREE_ASSERT(!loom_link_index_provider_roots_contains(roots, symbol_ordinal));
  IREE_ASSERT(roots->ordinals.count < roots->ordinals.capacity);
  roots->membership.values[symbol_ordinal >> 6] |= UINT64_C(1)
                                                   << (symbol_ordinal & 63u);
  roots->ordinals.values[roots->ordinals.count++] = symbol_ordinal;
}

static void loom_link_index_apply_provider_roots(
    const loom_link_index_provider_roots_t* roots,
    loom_link_plan_options_t* options) {
  options->template_provider_roots.count = roots->ordinals.count;
  options->template_provider_roots.values = roots->ordinals.values;
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
    const loom_link_index_provider_roots_t* roots,
    const loom_link_plan_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    loom_template_selection_query_result_t* out_result) {
  loom_template_provider_slice_t external_candidates =
      loom_template_provider_slice_empty();
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_loader_load(
      candidate_loader, plan, materialization, arena, &external_candidates));

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

static bool loom_link_index_append_query_roots(
    const loom_template_selection_query_result_t* query,
    loom_link_index_provider_roots_t* roots) {
  const iree_host_size_t prior_count = roots->ordinals.count;
  for (iree_host_size_t i = 0; i < query->required_origins.count; ++i) {
    const iree_host_size_t symbol_ordinal = query->required_origins.values[i];
    IREE_ASSERT(symbol_ordinal < roots->membership.symbol_count);
    if (!loom_link_index_provider_roots_contains(roots, symbol_ordinal)) {
      loom_link_index_provider_roots_append(roots, symbol_ordinal);
    }
  }
  return roots->ordinals.count != prior_count;
}

static iree_status_t loom_link_index_build_plan(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* base_options,
    const loom_link_index_provider_roots_t* roots, iree_allocator_t allocator,
    loom_link_plan_t** out_plan) {
  loom_link_plan_options_t options = *base_options;
  loom_link_index_apply_provider_roots(roots, &options);
  return loom_link_plan_build(index, &options, allocator, out_plan);
}

static iree_status_t loom_link_index_materialize_link(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_index_materialization_t* out_materialization) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  loom_link_index_provider_roots_t provider_roots;
  IREE_RETURN_IF_ERROR(loom_link_index_provider_roots_initialize(
      symbol_count, environment->allocator, &provider_roots));
  loom_link_template_candidate_loader_t* candidate_loader = NULL;
  iree_status_t status = loom_link_template_candidate_loader_allocate(
      index, environment, &candidate_loader);

  loom_link_plan_t* stable_plan = NULL;
  loom_link_plan_materialization_t stable_product = {0};
  iree_arena_allocator_t stable_arena = {0};
  iree_host_size_t queried_template_demand_count = 0;
  bool has_queried_template_demands = false;
  while (iree_status_is_ok(status) && stable_plan == NULL) {
    loom_link_plan_t* plan = NULL;
    status = loom_link_index_build_plan(index, plan_options, &provider_roots,
                                        environment->allocator, &plan);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (loom_link_plan_demanded_template_family_count(plan) == 0) {
      stable_plan = plan;
      break;
    }
    const iree_host_size_t template_demand_count =
        loom_link_plan_template_demand_occurrence_count(plan);
    // The prior query considered every provider for every demanded family.
    // When the retained closure adds no application sites, neither its
    // applications nor its provider universe changed and selection is stable.
    if (has_queried_template_demands &&
        template_demand_count == queried_template_demand_count) {
      stable_plan = plan;
      break;
    }

    iree_arena_allocator_t materialization_arena;
    iree_arena_initialize(environment->block_pool, &materialization_arena);
    loom_link_plan_materialization_t analysis = {0};
    status = loom_link_plan_materialize(plan, environment, module_name,
                                        &materialization_arena, &analysis);
    iree_arena_allocator_t query_arena;
    iree_arena_initialize(environment->block_pool, &query_arena);
    loom_template_selection_query_result_t query = {0};
    if (iree_status_is_ok(status)) {
      status = loom_link_index_query_candidates(
          index, candidate_loader, plan, &analysis, &provider_roots,
          plan_options, environment->block_pool, &query_arena, &query);
    }
    const bool changed =
        iree_status_is_ok(status) &&
        loom_link_index_append_query_roots(&query, &provider_roots);
    iree_arena_deinitialize(&query_arena);
    if (!iree_status_is_ok(status)) {
      loom_module_free(analysis.module);
      iree_arena_deinitialize(&materialization_arena);
      loom_link_plan_free(plan);
      break;
    }
    if (changed) {
      queried_template_demand_count = template_demand_count;
      has_queried_template_demands = true;
      loom_link_plan_free(plan);
    } else {
      stable_plan = plan;
      stable_product = analysis;
      analysis = (loom_link_plan_materialization_t){0};
      stable_arena = materialization_arena;
      materialization_arena = (iree_arena_allocator_t){0};
    }
    loom_module_free(analysis.module);
    iree_arena_deinitialize(&materialization_arena);
  }

  if (iree_status_is_ok(status) && stable_product.module != NULL) {
    out_materialization->plan = stable_plan;
    out_materialization->product = stable_product;
    out_materialization->arena = stable_arena;
    stable_plan = NULL;
    stable_product = (loom_link_plan_materialization_t){0};
    stable_arena = (iree_arena_allocator_t){0};
  } else if (iree_status_is_ok(status)) {
    iree_arena_initialize(environment->block_pool, &out_materialization->arena);
    status = loom_link_plan_materialize(stable_plan, environment, module_name,
                                        &out_materialization->arena,
                                        &out_materialization->product);
    if (iree_status_is_ok(status)) {
      out_materialization->plan = stable_plan;
      stable_plan = NULL;
    }
  }

  loom_link_plan_free(stable_plan);
  loom_module_free(stable_product.module);
  iree_arena_deinitialize(&stable_arena);
  loom_link_template_candidate_loader_free(candidate_loader);
  loom_link_index_provider_roots_deinitialize(&provider_roots);
  return status;
}

static iree_status_t loom_link_index_materialize_merge(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_index_materialization_t* out_materialization) {
  loom_link_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_plan_build(index, plan_options, environment->allocator, &plan));
  iree_arena_initialize(environment->block_pool, &out_materialization->arena);
  iree_status_t status = loom_link_plan_materialize(
      plan, environment, module_name, &out_materialization->arena,
      &out_materialization->product);
  if (iree_status_is_ok(status)) {
    out_materialization->plan = plan;
    plan = NULL;
  }
  loom_link_plan_free(plan);
  return status;
}

void loom_link_index_materialization_deinitialize(
    loom_link_index_materialization_t* materialization) {
  if (materialization == NULL) {
    return;
  }
  loom_module_free(materialization->product.module);
  loom_link_plan_free(materialization->plan);
  iree_arena_deinitialize(&materialization->arena);
  *materialization = (loom_link_index_materialization_t){0};
}

iree_status_t loom_link_index_materialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_index_materialization_t* out_materialization) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(plan_options);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = (loom_link_index_materialization_t){0};
  if (plan_options->template_provider_roots.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "index materialization owns template provider roots");
  }
  iree_status_t status = iree_ok_status();
  switch (plan_options->mode) {
    case LOOM_LINK_PLAN_MERGE:
      status = loom_link_index_materialize_merge(
          index, plan_options, environment, module_name, out_materialization);
      break;
    case LOOM_LINK_PLAN_LINK:
      status = loom_link_index_materialize_link(
          index, plan_options, environment, module_name, out_materialization);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("unknown link plan mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (!iree_status_is_ok(status)) {
    loom_link_index_materialization_deinitialize(out_materialization);
  }
  return status;
}
