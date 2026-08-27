// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_projection.h"

#include <stdlib.h>

#include "iree/base/assert.h"

static int loom_link_plan_projection_compare_symbols(const void* lhs_ptr,
                                                     const void* rhs_ptr) {
  const loom_link_plan_module_symbol_t* lhs =
      (const loom_link_plan_module_symbol_t*)lhs_ptr;
  const loom_link_plan_module_symbol_t* rhs =
      (const loom_link_plan_module_symbol_t*)rhs_ptr;
  if (lhs->source_symbol->module_ordinal < rhs->source_symbol->module_ordinal) {
    return -1;
  }
  if (lhs->source_symbol->module_ordinal > rhs->source_symbol->module_ordinal) {
    return 1;
  }
  if (lhs->source_symbol->module_symbol_ordinal <
      rhs->source_symbol->module_symbol_ordinal) {
    return -1;
  }
  if (lhs->source_symbol->module_symbol_ordinal >
      rhs->source_symbol->module_symbol_ordinal) {
    return 1;
  }
  return 0;
}

static bool loom_link_plan_projection_symbol_is_partial(
    const loom_link_plan_module_symbol_t* symbol) {
  IREE_ASSERT_LE(symbol->plan_symbol->selected_facet_count,
                 symbol->source_symbol->facets.schema.facet_count);
  return symbol->plan_symbol->selected_facet_count !=
         symbol->source_symbol->facets.schema.facet_count;
}

static void loom_link_plan_projection_assign_complete_symbol_ordinals(
    loom_link_plan_module_selection_t* module) {
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_link_plan_module_symbol_t* symbol = &module->symbols.values[i];
    module->symbols.values[i].materialized_symbol_ordinal =
        (uint32_t)(module->source_module->materialized_module
                       ? symbol->source_symbol->module_symbol_ordinal
                       : i);
  }
}

// Assigns the source-symbol domain consumed by the incremental linker.
//
// Complete materialized modules link sparsely from their original symbol
// tables. Selected bytecode modules are already compact in selection order.
// A module containing partial semantic projections materializes complete
// symbols first, then one source-facing symbol per partial selection. Synthetic
// projection helpers follow the complete source-symbol domain. This lets the
// selected bytecode reader populate the complete prefix without opening omitted
// facets while reserving every authored symbol name before helper names are
// chosen.
static void loom_link_plan_projection_assign_materialized_symbol_ordinals(
    const loom_link_plan_t* plan, loom_link_plan_module_selection_t* module) {
  if (loom_link_plan_mode(plan) != LOOM_LINK_PLAN_LINK) {
    module->projected_symbol_count = 0;
    loom_link_plan_projection_assign_complete_symbol_ordinals(module);
    return;
  }

  iree_host_size_t complete_symbol_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    complete_symbol_count += !loom_link_plan_projection_symbol_is_partial(
        &module->symbols.values[i]);
  }
  const iree_host_size_t partial_symbol_count =
      module->symbols.count - complete_symbol_count;
  module->projected_symbol_count =
      partial_symbol_count == 0 ? 0
                                : module->symbols.count + partial_symbol_count;

  if (!loom_link_plan_module_requires_symbol_projection(module)) {
    loom_link_plan_projection_assign_complete_symbol_ordinals(module);
    return;
  }

  uint32_t complete_ordinal = 0;
  uint32_t partial_ordinal = (uint32_t)complete_symbol_count;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_link_plan_module_symbol_t* symbol = &module->symbols.values[i];
    if (loom_link_plan_projection_symbol_is_partial(symbol)) {
      symbol->materialized_symbol_ordinal = partial_ordinal++;
    } else {
      symbol->materialized_symbol_ordinal = complete_ordinal++;
    }
  }
}

iree_status_t loom_link_plan_project_modules(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    loom_link_plan_module_projection_t* out_projection) {
  *out_projection = (loom_link_plan_module_projection_t){0};
  const iree_host_size_t symbol_count = loom_link_plan_symbol_count(plan);
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const bool projects_merged_modules =
      loom_link_plan_mode(plan) == LOOM_LINK_PLAN_MERGE;

  loom_link_plan_module_symbol_t* symbols = NULL;
  if (symbol_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, symbol_count, sizeof(*symbols), (void**)&symbols));
  }
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_plan_symbol_t* planned_symbol =
        loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* source_symbol =
        loom_link_module_index_symbol_at(index, planned_symbol->symbol_ordinal);
    symbols[i] = (loom_link_plan_module_symbol_t){
        .plan_symbol = planned_symbol,
        .source_symbol = source_symbol,
    };
  }
  if (symbol_count > 1) {
    qsort(symbols, symbol_count, sizeof(*symbols),
          loom_link_plan_projection_compare_symbols);
  }

  iree_host_size_t module_count = 0;
  if (projects_merged_modules) {
    const iree_host_size_t provider_count =
        loom_link_module_index_provider_count(index);
    for (iree_host_size_t i = 0; i < provider_count; ++i) {
      const loom_link_module_index_provider_t* provider =
          loom_link_module_index_provider_at(index, i);
      if (provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT) {
        continue;
      }
      if (!iree_host_size_checked_add(module_count, provider->module_count,
                                      &module_count)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "projected module count overflow");
      }
    }
  } else if (symbol_count > 0) {
    module_count = 1;
    for (iree_host_size_t i = 1; i < symbol_count; ++i) {
      if (symbols[i - 1].source_symbol->module_ordinal !=
          symbols[i].source_symbol->module_ordinal) {
        ++module_count;
      }
    }
  }
  if (module_count == 0) {
    return iree_ok_status();
  }

  loom_link_plan_module_selection_t* modules = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_count, sizeof(*modules), (void**)&modules));

  if (projects_merged_modules) {
    iree_host_size_t symbol_index = 0;
    iree_host_size_t module_index = 0;
    const iree_host_size_t provider_count =
        loom_link_module_index_provider_count(index);
    for (iree_host_size_t provider_ordinal = 0;
         provider_ordinal < provider_count; ++provider_ordinal) {
      const loom_link_module_index_provider_t* provider =
          loom_link_module_index_provider_at(index, provider_ordinal);
      if (provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT) {
        continue;
      }
      const iree_host_size_t module_end_ordinal =
          provider->module_start_ordinal + provider->module_count;
      for (iree_host_size_t source_module_ordinal =
               provider->module_start_ordinal;
           source_module_ordinal < module_end_ordinal;
           ++source_module_ordinal) {
        const iree_host_size_t first_symbol_index = symbol_index;
        while (symbol_index < symbol_count &&
               symbols[symbol_index].source_symbol->module_ordinal ==
                   source_module_ordinal) {
          ++symbol_index;
        }
        const iree_host_size_t module_symbol_count =
            symbol_index - first_symbol_index;
        modules[module_index++] = (loom_link_plan_module_selection_t){
            .source_module =
                loom_link_module_index_module_at(index, source_module_ordinal),
            .symbols =
                {
                    .values = module_symbol_count > 0
                                  ? symbols + first_symbol_index
                                  : NULL,
                    .count = module_symbol_count,
                },
            .projected_symbol_count = 0,
        };
      }
    }
    IREE_ASSERT_EQ(module_index, module_count);
    IREE_ASSERT_EQ(symbol_index, symbol_count);
  } else {
    iree_host_size_t module_ordinal = 0;
    iree_host_size_t first_symbol_index = 0;
    for (iree_host_size_t i = 0; i < symbol_count; ++i) {
      const bool ends_module = i + 1 == symbol_count ||
                               symbols[i].source_symbol->module_ordinal !=
                                   symbols[i + 1].source_symbol->module_ordinal;
      if (ends_module) {
        modules[module_ordinal++] = (loom_link_plan_module_selection_t){
            .source_module = loom_link_module_index_module_at(
                index, symbols[i].source_symbol->module_ordinal),
            .symbols =
                {
                    .values = symbols + first_symbol_index,
                    .count = i + 1 - first_symbol_index,
                },
            .projected_symbol_count = 0,
        };
        first_symbol_index = i + 1;
      }
    }
  }

  iree_host_size_t maximum_materialized_symbol_count = 0;
  iree_host_size_t synthetic_symbol_count = 0;
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    loom_link_plan_projection_assign_materialized_symbol_ordinals(plan,
                                                                  &modules[i]);
    const iree_host_size_t materialized_symbol_count =
        loom_link_plan_module_requires_symbol_projection(&modules[i])
            ? modules[i].projected_symbol_count
            : modules[i].symbols.count;
    maximum_materialized_symbol_count =
        iree_max(maximum_materialized_symbol_count, materialized_symbol_count);
    const iree_host_size_t module_synthetic_symbol_count =
        materialized_symbol_count - modules[i].symbols.count;
    if (!iree_host_size_checked_add(synthetic_symbol_count,
                                    module_synthetic_symbol_count,
                                    &synthetic_symbol_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "projected symbol count overflow");
    }
  }

  out_projection->maximum_materialized_symbol_count =
      maximum_materialized_symbol_count;
  out_projection->synthetic_symbol_count = synthetic_symbol_count;
  out_projection->modules.values = modules;
  out_projection->modules.count = module_count;
  out_projection->symbols.values = symbols;
  out_projection->symbols.count = symbol_count;
  return iree_ok_status();
}
