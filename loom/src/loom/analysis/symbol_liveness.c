// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_liveness.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

typedef struct loom_symbol_liveness_worklist_t {
  // Symbol ids still waiting for defining-op traversal.
  loom_symbol_id_t* entries;

  // Number of queued symbol ids.
  iree_host_size_t count;

  // Capacity of entries.
  iree_host_size_t capacity;
} loom_symbol_liveness_worklist_t;

typedef struct loom_symbol_liveness_state_t {
  // Module being analyzed.
  const loom_module_t* module;

  // Concrete symbol dependency table for module.
  const loom_symbol_dependency_table_t* dependencies;

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
  if (!state->options.root_query) return iree_ok_status();
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
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_mark_module_root_edges(
    loom_symbol_liveness_state_t* state) {
  loom_symbol_dependency_edge_id_t edge_id =
      state->dependencies->first_module_edge_id;
  while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
    const loom_symbol_dependency_edge_t* edge =
        &state->dependencies->edges[edge_id];
    ++state->concrete_edge_count;
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_concrete_symbol_id(
        state, edge->target_symbol_id));
    edge_id = edge->next_outgoing_edge_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_visit_contributors(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t source_symbol_id,
    const loom_symbol_t* source_symbol, const loom_op_t* op) {
  if (!state->has_contributors) {
    return iree_ok_status();
  }
  loom_symbol_liveness_contributor_context_t context = {
      .module = state->module,
      .dependencies = state->dependencies,
      .arena = state->arena,
      .source_symbol_id = source_symbol_id,
      .source_symbol = source_symbol,
      .engine_state = state,
  };
  for (iree_host_size_t i = 0; i < state->options.contributor_count; ++i) {
    const loom_symbol_liveness_contributor_t* contributor =
        &state->options.contributors[i];
    if (!contributor->visit_op) continue;
    IREE_RETURN_IF_ERROR(
        contributor->visit_op(contributor->user_data, &context, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_scan_op(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t source_symbol_id,
    const loom_symbol_t* source_symbol, const loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }

  loom_symbol_ref_t nested_symbol_ref = loom_symbol_ref_null();
  if (loom_op_defining_symbol_ref(state->module, op, &nested_symbol_ref) &&
      nested_symbol_ref.symbol_id != source_symbol_id) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_symbol_liveness_visit_contributors(
      state, source_symbol_id, source_symbol, op));

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    const loom_region_t* region = regions[i];
    if (!region) continue;
    const loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      const loom_op_t* nested_op = NULL;
      loom_block_for_each_op(block, nested_op) {
        IREE_RETURN_IF_ERROR(loom_symbol_liveness_scan_op(
            state, source_symbol_id, source_symbol, nested_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_liveness_traverse_symbol(
    loom_symbol_liveness_state_t* state, loom_symbol_id_t symbol_id) {
  if (symbol_id >= state->dependencies->symbol_count) return iree_ok_status();

  loom_symbol_dependency_edge_id_t edge_id =
      state->dependencies->symbols[symbol_id].first_outgoing_edge_id;
  while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
    const loom_symbol_dependency_edge_t* edge =
        &state->dependencies->edges[edge_id];
    ++state->concrete_edge_count;
    IREE_RETURN_IF_ERROR(loom_symbol_liveness_mark_concrete_symbol_id(
        state, edge->target_symbol_id));
    edge_id = edge->next_outgoing_edge_id;
  }

  if (!state->has_contributors) return iree_ok_status();
  const loom_symbol_t* symbol = &state->module->symbols.entries[symbol_id];
  IREE_RETURN_IF_ERROR(loom_symbol_liveness_scan_op(state, symbol_id, symbol,
                                                    symbol->defining_op));
  return iree_ok_status();
}

static bool loom_symbol_liveness_options_have_contributors(
    const loom_symbol_liveness_options_t* options) {
  if (!options || !options->contributors || options->contributor_count == 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->contributor_count; ++i) {
    if (options->contributors[i].visit_op) return true;
  }
  return false;
}

static iree_status_t loom_symbol_liveness_validate(
    const loom_module_t* module,
    const loom_symbol_dependency_table_t* dependencies,
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
  if (!dependencies) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness dependency table is NULL");
  }
  if (dependencies->module != module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness dependency table is for a "
                            "different module");
  }
  if (dependencies->symbol_count != module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness dependency table has %" PRIhsz
                            " symbols but module has %" PRIhsz " symbols",
                            dependencies->symbol_count, module->symbols.count);
  }
  if (dependencies->symbol_count > 0 && !dependencies->symbols) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness dependency symbols are NULL");
  }
  if (dependencies->edge_count > 0 && !dependencies->edges) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness dependency edges are NULL");
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol liveness arena is NULL");
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_liveness_compute(
    const loom_module_t* module,
    const loom_symbol_dependency_table_t* dependencies,
    const loom_symbol_liveness_options_t* options,
    iree_arena_allocator_t* arena, loom_symbol_liveness_t* out_liveness) {
  IREE_RETURN_IF_ERROR(
      loom_symbol_liveness_validate(module, dependencies, arena, out_liveness));

  loom_symbol_liveness_state_t state = {
      .module = module,
      .dependencies = dependencies,
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
      .dependencies = dependencies,
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
