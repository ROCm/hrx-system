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

static bool loom_link_plan_projection_retains_import_anchor(
    const loom_link_plan_t* plan,
    const loom_link_module_index_module_t* source_module,
    uint32_t source_symbol_ordinal) {
  const iree_host_size_t symbol_ordinal =
      source_module->symbol_start_ordinal + source_symbol_ordinal;
  return loom_link_plan_contains_symbol(plan, symbol_ordinal) &&
         !loom_link_plan_symbol_imports_resolved(plan, symbol_ordinal);
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
  if (loom_link_plan_mode(plan) != LOOM_LINK_PLAN_SELECTIVE) {
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
  const bool projects_archive_modules =
      loom_link_plan_mode(plan) == LOOM_LINK_PLAN_ARCHIVE;

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
  if (projects_archive_modules) {
    module_count = loom_link_module_index_module_count(index);
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

  if (projects_archive_modules) {
    iree_host_size_t symbol_ordinal = 0;
    for (iree_host_size_t module_ordinal = 0; module_ordinal < module_count;
         ++module_ordinal) {
      const iree_host_size_t first_symbol_ordinal = symbol_ordinal;
      while (symbol_ordinal < symbol_count &&
             symbols[symbol_ordinal].source_symbol->module_ordinal ==
                 module_ordinal) {
        ++symbol_ordinal;
      }
      const iree_host_size_t module_symbol_count =
          symbol_ordinal - first_symbol_ordinal;
      modules[module_ordinal] = (loom_link_plan_module_selection_t){
          .source_module =
              loom_link_module_index_module_at(index, module_ordinal),
          .symbols =
              {
                  .values = module_symbol_count > 0
                                ? symbols + first_symbol_ordinal
                                : NULL,
                  .count = module_symbol_count,
              },
          .provider_imports = {0},
          .provider_import_anchors = {0},
          .projected_symbol_count = 0,
      };
    }
    IREE_ASSERT_EQ(symbol_ordinal, symbol_count);
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
            .provider_imports = {0},
            .provider_import_anchors = {0},
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

  iree_host_size_t provider_import_count = 0;
  iree_host_size_t provider_import_anchor_count = 0;
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    const loom_link_module_index_module_t* source_module =
        modules[i].source_module;
    for (uint32_t import_ordinal = 0;
         import_ordinal < source_module->provider_imports.count;
         ++import_ordinal) {
      const loom_link_module_index_provider_import_t provider_import =
          loom_link_module_index_provider_import_at(index, source_module,
                                                    import_ordinal);
      iree_host_size_t retained_anchor_count = 0;
      for (iree_host_size_t anchor_ordinal = 0;
           anchor_ordinal < provider_import.anchor_count; ++anchor_ordinal) {
        const uint32_t source_symbol_ordinal =
            loom_link_module_index_provider_import_anchor_at(
                index, source_module, import_ordinal, anchor_ordinal);
        retained_anchor_count +=
            loom_link_plan_projection_retains_import_anchor(
                plan, source_module, source_symbol_ordinal);
      }
      if (retained_anchor_count == 0) {
        continue;
      }
      if (!iree_host_size_checked_add(provider_import_count, 1,
                                      &provider_import_count) ||
          !iree_host_size_checked_add(provider_import_anchor_count,
                                      retained_anchor_count,
                                      &provider_import_anchor_count)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "retained provider import count overflow");
      }
    }
  }

  loom_link_plan_module_provider_import_t* provider_imports = NULL;
  if (provider_import_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, provider_import_count,
                                                   sizeof(*provider_imports),
                                                   (void**)&provider_imports));
  }
  uint32_t* provider_import_anchors = NULL;
  if (provider_import_anchor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, provider_import_anchor_count, sizeof(*provider_import_anchors),
        (void**)&provider_import_anchors));
  }

  iree_host_size_t provider_import_ordinal = 0;
  iree_host_size_t provider_import_anchor_ordinal = 0;
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    loom_link_plan_module_selection_t* module = &modules[i];
    const iree_host_size_t module_import_start = provider_import_ordinal;
    const iree_host_size_t module_import_anchor_start =
        provider_import_anchor_ordinal;
    for (uint32_t source_import_ordinal = 0;
         source_import_ordinal < module->source_module->provider_imports.count;
         ++source_import_ordinal) {
      const loom_link_module_index_provider_import_t source_import =
          loom_link_module_index_provider_import_at(
              index, module->source_module, source_import_ordinal);
      const iree_host_size_t import_anchor_start =
          provider_import_anchor_ordinal;
      for (iree_host_size_t source_anchor_ordinal = 0;
           source_anchor_ordinal < source_import.anchor_count;
           ++source_anchor_ordinal) {
        const uint32_t source_symbol_ordinal =
            loom_link_module_index_provider_import_anchor_at(
                index, module->source_module, source_import_ordinal,
                source_anchor_ordinal);
        if (loom_link_plan_projection_retains_import_anchor(
                plan, module->source_module, source_symbol_ordinal)) {
          provider_import_anchors[provider_import_anchor_ordinal++] =
              source_symbol_ordinal;
        }
      }
      const iree_host_size_t retained_anchor_count =
          provider_import_anchor_ordinal - import_anchor_start;
      if (retained_anchor_count == 0) {
        continue;
      }
      provider_imports[provider_import_ordinal++] =
          (loom_link_plan_module_provider_import_t){
              .source_import_ordinal = source_import_ordinal,
              .anchors =
                  {
                      .first = (uint32_t)(import_anchor_start -
                                          module_import_anchor_start),
                      .count = (uint32_t)retained_anchor_count,
                  },
          };
    }
    const iree_host_size_t module_import_count =
        provider_import_ordinal - module_import_start;
    module->provider_imports.values =
        module_import_count > 0 ? provider_imports + module_import_start : NULL;
    module->provider_imports.count = module_import_count;
    const iree_host_size_t module_import_anchor_count =
        provider_import_anchor_ordinal - module_import_anchor_start;
    module->provider_import_anchors.values =
        module_import_anchor_count > 0
            ? provider_import_anchors + module_import_anchor_start
            : NULL;
    module->provider_import_anchors.count = module_import_anchor_count;
  }

  out_projection->maximum_materialized_symbol_count =
      maximum_materialized_symbol_count;
  out_projection->synthetic_symbol_count = synthetic_symbol_count;
  out_projection->modules.values = modules;
  out_projection->modules.count = module_count;
  out_projection->symbols.values = symbols;
  out_projection->symbols.count = symbol_count;
  out_projection->provider_imports.values = provider_imports;
  out_projection->provider_imports.count = provider_import_count;
  out_projection->provider_import_anchors.values = provider_import_anchors;
  out_projection->provider_import_anchors.count = provider_import_anchor_count;
  return iree_ok_status();
}

static uint32_t loom_link_plan_projection_find_compact_symbol_ordinal(
    const loom_link_plan_module_selection_t* module,
    uint32_t source_symbol_ordinal) {
  iree_host_size_t lower_bound = 0;
  iree_host_size_t upper_bound = module->symbols.count;
  while (lower_bound < upper_bound) {
    const iree_host_size_t middle =
        lower_bound + (upper_bound - lower_bound) / 2;
    const iree_host_size_t candidate =
        module->symbols.values[middle].source_symbol->module_symbol_ordinal;
    if (candidate < source_symbol_ordinal) {
      lower_bound = middle + 1;
    } else if (candidate > source_symbol_ordinal) {
      upper_bound = middle;
    } else {
      return module->symbols.values[middle].materialized_symbol_ordinal;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "retained provider import anchor is absent from its module selection");
  return 0;
}

iree_status_t loom_link_plan_project_linker_imports(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* module_projection,
    iree_arena_allocator_t* arena,
    loom_link_plan_linker_import_projection_t* out_projection) {
  *out_projection = (loom_link_plan_linker_import_projection_t){0};

  loom_linker_source_provider_import_list_t* module_imports = NULL;
  if (module_projection->modules.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module_projection->modules.count, sizeof(*module_imports),
        (void**)&module_imports));
  }
  loom_linker_source_provider_import_t* provider_imports = NULL;
  if (module_projection->provider_imports.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module_projection->provider_imports.count,
        sizeof(*provider_imports), (void**)&provider_imports));
  }
  uint32_t* provider_import_anchors = NULL;
  if (module_projection->provider_import_anchors.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module_projection->provider_import_anchors.count,
        sizeof(*provider_import_anchors), (void**)&provider_import_anchors));
  }

  iree_host_size_t import_ordinal = 0;
  iree_host_size_t anchor_ordinal = 0;
  for (iree_host_size_t module_ordinal = 0;
       module_ordinal < module_projection->modules.count; ++module_ordinal) {
    const loom_link_plan_module_selection_t* module =
        &module_projection->modules.values[module_ordinal];
    const iree_host_size_t module_import_start = import_ordinal;
    for (iree_host_size_t module_import_ordinal = 0;
         module_import_ordinal < module->provider_imports.count;
         ++module_import_ordinal) {
      const loom_link_plan_module_provider_import_t* projected_import =
          &module->provider_imports.values[module_import_ordinal];
      const loom_link_module_index_provider_import_t source_import =
          loom_link_module_index_provider_import_at(
              index, module->source_module,
              projected_import->source_import_ordinal);
      const iree_host_size_t import_anchor_start = anchor_ordinal;
      for (iree_host_size_t i = 0; i < projected_import->anchors.count; ++i) {
        const uint32_t source_symbol_ordinal =
            module->provider_import_anchors
                .values[projected_import->anchors.first + i];
        provider_import_anchors[anchor_ordinal++] =
            module->source_module->materialized_module &&
                    !loom_link_plan_module_requires_symbol_projection(module)
                ? source_symbol_ordinal
                : loom_link_plan_projection_find_compact_symbol_ordinal(
                      module, source_symbol_ordinal);
      }
      provider_imports[import_ordinal++] =
          (loom_linker_source_provider_import_t){
              .provider = source_import.provider,
              .anchors =
                  {
                      .count = projected_import->anchors.count,
                      .ordinals = provider_import_anchors + import_anchor_start,
                  },
              .comments =
                  {
                      .count = source_import.comments.count,
                      .values = source_import.comments.values,
                  },
              .leading_blank_line = source_import.leading_blank_line,
          };
    }
    module_imports[module_ordinal] =
        (loom_linker_source_provider_import_list_t){
            .count = import_ordinal - module_import_start,
            .values = import_ordinal == module_import_start
                          ? NULL
                          : provider_imports + module_import_start,
        };
  }

  out_projection->modules.values = module_imports;
  out_projection->modules.count = module_projection->modules.count;
  out_projection->provider_imports.values = provider_imports;
  out_projection->provider_imports.count =
      module_projection->provider_imports.count;
  out_projection->provider_import_anchors.values = provider_import_anchors;
  out_projection->provider_import_anchors.count =
      module_projection->provider_import_anchors.count;
  return iree_ok_status();
}
