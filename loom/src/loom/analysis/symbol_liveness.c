// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_liveness.h"

#include <string.h>

#include "loom/ir/module.h"

typedef struct loom_symbol_liveness_worklist_t {
  // Symbol ids still waiting for dependency expansion.
  loom_symbol_id_t* entries;

  // Number of queued symbol ids.
  iree_host_size_t count;

  // Capacity of entries.
  iree_host_size_t capacity;
} loom_symbol_liveness_worklist_t;

typedef struct loom_symbol_liveness_state_t {
  // Module being analyzed.
  const loom_module_t* module;

  // Concrete symbol reference table for module.
  const loom_symbol_reference_table_t* references;

  // Analysis options with NULL-safe defaults.
  loom_symbol_liveness_options_t options;

  // Arena receiving result and worklist storage.
  iree_arena_allocator_t* arena;

  // Mutable live-symbol bytes.
  uint8_t* live_symbols;

  // Pending reachable symbols whose bodies still need traversal.
  loom_symbol_liveness_worklist_t worklist;

  // Number of concrete dependency edges traversed from live symbols.
  uint32_t concrete_edge_count;

  // Number of contributor-added symbol edges.
  uint32_t contributed_edge_count;

  // True when at least one contributor has an active visit callback.
  bool has_contributors;
} loom_symbol_liveness_state_t;

static iree_status_t loom_symbol_liveness_worklist_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t initial_capacity,
    loom_symbol_liveness_worklist_t* worklist) {
  worklist->count = 0;
  worklist->capacity = iree_max(initial_capacity, (iree_host_size_t)16);
  return iree_arena_allocate_array(arena, worklist->capacity,
                                   sizeof(*worklist->entries),
                                   (void**)&worklist->entries);
}

static iree_status_t loom_symbol_liveness_mark_symbol_id_impl(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t symbol_id,
    bool contributed) {
  if (symbol_id >= state->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "live symbol id %u is outside the module symbol table",
        (uint32_t)symbol_id);
  }
  if (contributed) ++state->contributed_edge_count;
  if (state->live_symbols[symbol_id]) return iree_ok_status();
  if (state->worklist.count >= state->worklist.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->worklist.count, state->worklist.count + 1,
        sizeof(*state->worklist.entries), &state->worklist.capacity,
        (void**)&state->worklist.entries));
  }
  state->live_symbols[symbol_id] = 1;
  state->worklist.entries[state->worklist.count++] = symbol_id;
  return iree_ok_status();
}

iree_status_t loom_symbol_liveness_mark_symbol_id(
    loom_symbol_liveness_contributor_context_t* context,
    loom_symbol_id_t symbol_id) {
  if (!context || !context->engine_state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness contributor context is NULL");
  }
  return loom_symbol_liveness_mark_symbol_id_impl(
      (loom_symbol_liveness_state_t*)context->engine_state, symbol_id,
      /*contributed=*/true);
}

iree_status_t loom_symbol_liveness_mark_symbol_ref(
    loom_symbol_liveness_contributor_context_t* context,
    loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0) {
    return iree_ok_status();
  }
  return loom_symbol_liveness_mark_symbol_id(context, ref.symbol_id);
}

static iree_status_t loom_symbol_liveness_mark_concrete_symbol_id(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t symbol_id) {
  return loom_symbol_liveness_mark_symbol_id_impl(state, symbol_id,
                                                  /*contributed=*/false);
}

static iree_status_t loom_symbol_liveness_seed_roots(
    loom_symbol_liveness_state_t* state) {
  for (iree_host_size_t i = 0; i < state->options.root_symbol_ids.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_concrete_symbol_id(
        state, state->options.root_symbol_ids.values[i]));
  }
  if (state->options.root_query) {
    const loom_symbol_t* symbol = NULL;
    loom_module_for_each_symbol(state->module, symbol) {
      if (!symbol->defining_op) continue;
      loom_symbol_id_t symbol_id =
          (loom_symbol_id_t)(symbol - state->module->symbols.entries);
      if (!state->options.root_query(state->options.root_query_user_data,
                                     state->module, symbol_id, symbol)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_symbol_liveness_mark_concrete_symbol_id(state, symbol_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_mark_module_root_edges(
    loom_symbol_liveness_state_t* state) {
  loom_symbol_reference_occurrence_id_t edge_id =
      state->references->first_module_occurrence_id;
  while (edge_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* edge =
        &state->references->occurrences[edge_id];
    if (!loom_symbol_reference_occurrence_is_dependency(edge)) {
      edge_id = edge->next_outgoing_occurrence_id;
      continue;
    }
    ++state->concrete_edge_count;
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_concrete_symbol_id(
        state, edge->target_symbol_id));
    edge_id = edge->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_visit_contributors(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t source_symbol_id,
    const loom_symbol_t* source_symbol, const loom_template_demand_t* demand) {
  if (!state->has_contributors) {
    return iree_ok_status();
  }
  loom_symbol_liveness_contributor_context_t context = {
      .module = state->module,
      .references = state->references,
      .arena = state->arena,
      .source_symbol_id = source_symbol_id,
      .source_symbol = source_symbol,
      .engine_state = state,
  };
  for (iree_host_size_t i = 0; i < state->options.contributor_count; ++i) {
    const loom_symbol_liveness_contributor_t* contributor =
        &state->options.contributors[i];
    if (!contributor->visit_template_demand) {
      continue;
    }
    IREE_RETURN_IF_ERROR(contributor->visit_template_demand(
        contributor->user_data, &context, demand));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_traverse_symbol(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t symbol_id) {
  if (symbol_id >= state->references->symbol_count) return iree_ok_status();

  loom_symbol_reference_occurrence_id_t edge_id =
      state->references->symbols[symbol_id].first_outgoing_occurrence_id;
  while (edge_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* edge =
        &state->references->occurrences[edge_id];
    if (!loom_symbol_reference_occurrence_is_dependency(edge)) {
      edge_id = edge->next_outgoing_occurrence_id;
      continue;
    }
    ++state->concrete_edge_count;
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_concrete_symbol_id(
        state, edge->target_symbol_id));
    edge_id = edge->next_outgoing_occurrence_id;
  }

  if (!state->has_contributors) return iree_ok_status();
  const loom_symbol_t* symbol = &state->module->symbols.entries[symbol_id];
  loom_template_demand_id_t demand_id =
      state->references->symbols[symbol_id].first_template_demand_id;
  while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
    const loom_template_demand_t* demand =
        &state->references->template_demands.values[demand_id];
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_visit_contributors(
        state, symbol_id, symbol, demand));
    demand_id = demand->next_source_demand_id;
  }
  return iree_ok_status();
}

static bool loom_symbol_liveness_options_have_contributors(
    const loom_symbol_liveness_options_t* options) {
  if (!options || !options->contributors || options->contributor_count == 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->contributor_count; ++i) {
    if (options->contributors[i].visit_template_demand) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_symbol_liveness_validate(
    const loom_module_t* module,
    const loom_symbol_reference_table_t* references,
    iree_arena_allocator_t* arena, loom_symbol_liveness_t* out_liveness) {
  if (!out_liveness) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness output is NULL");
  }
  *out_liveness = (loom_symbol_liveness_t){0};
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness module is NULL");
  }
  if (!references) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness reference table is NULL");
  }
  if (references->module != module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness reference table is for a "
                            "different module");
  }
  if (references->symbol_count != module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness reference table has %" PRIhsz
                            " symbols but module has %" PRIhsz " symbols",
                            references->symbol_count, module->symbols.count);
  }
  if (references->symbol_count > 0 && !references->symbols) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness reference symbols are NULL");
  }
  if (references->occurrence_count > 0 && !references->occurrences) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness reference occurrences are NULL");
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness arena is NULL");
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_liveness_compute(
    const loom_module_t* module,
    const loom_symbol_reference_table_t* references,
    const loom_symbol_liveness_options_t* options,
    iree_arena_allocator_t* arena, loom_symbol_liveness_t* out_liveness) {
  IREE_RETURN_IF_ERROR(
      loom_symbol_liveness_validate(module, references, arena, out_liveness));

  loom_symbol_liveness_state_t state = {
      .module = module,
      .references = references,
      .options = options ? *options : (loom_symbol_liveness_options_t){0},
      .arena = arena,
      .has_contributors =
          loom_symbol_liveness_options_have_contributors(options),
  };
  if (module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*state.live_symbols),
        (void**)&state.live_symbols));
    memset(state.live_symbols, 0,
           module->symbols.count * sizeof(*state.live_symbols));
  }
  IREE_RETURN_IF_ERROR(loom_symbol_liveness_worklist_initialize(
      arena, module->symbols.count, &state.worklist));
  IREE_RETURN_IF_ERROR(loom_symbol_liveness_seed_roots(&state));
  if (iree_any_bit_set(state.options.flags,
                       LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES)) {
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_module_root_edges(&state));
  }
  while (state.worklist.count > 0) {
    const loom_symbol_id_t symbol_id =
        state.worklist.entries[--state.worklist.count];
    IREE_RETURN_IF_ERROR(
        loom_symbol_liveness_traverse_symbol(&state, symbol_id));
  }

  *out_liveness = (loom_symbol_liveness_t){
      .module = module,
      .references = references,
      .live_symbols = state.live_symbols,
      .symbol_count = module->symbols.count,
      .concrete_edge_count = state.concrete_edge_count,
      .contributed_edge_count = state.contributed_edge_count,
  };
  return iree_ok_status();
}

bool loom_symbol_liveness_is_live(const loom_symbol_liveness_t* liveness,
                                  loom_symbol_id_t symbol_id) {
  return liveness && symbol_id < liveness->symbol_count &&
         liveness->live_symbols && liveness->live_symbols[symbol_id] != 0;
}
