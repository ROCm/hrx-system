// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_projection.h"

#include <stdlib.h>

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

iree_status_t loom_link_plan_project_modules(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    loom_link_plan_module_projection_t* out_projection) {
  *out_projection = (loom_link_plan_module_projection_t){0};
  const iree_host_size_t symbol_count = loom_link_plan_symbol_count(plan);
  if (symbol_count == 0) {
    return iree_ok_status();
  }

  loom_link_plan_module_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(*symbols), (void**)&symbols));
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
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
  qsort(symbols, symbol_count, sizeof(*symbols),
        loom_link_plan_projection_compare_symbols);

  iree_host_size_t module_count = 1;
  for (iree_host_size_t i = 1; i < symbol_count; ++i) {
    if (symbols[i - 1].source_symbol->module_ordinal !=
        symbols[i].source_symbol->module_ordinal) {
      ++module_count;
    }
  }

  loom_link_plan_module_selection_t* modules = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_count, sizeof(*modules), (void**)&modules));

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
      };
      first_symbol_index = i + 1;
    }
  }

  out_projection->modules.values = modules;
  out_projection->modules.count = module_count;
  out_projection->symbols.values = symbols;
  out_projection->symbols.count = symbol_count;
  return iree_ok_status();
}
