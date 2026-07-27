// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/planner.h"

#include <string.h>

#include "loom/analysis/symbol_dependencies.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/walk.h"

typedef struct loom_link_plan_module_state_t {
  // True once dependency_table has been built.
  bool dependencies_built;
  // True once module-root dependency edges have been scanned.
  bool module_edges_scanned;
  // Materialized provider module used for dependency scans.
  const loom_module_t* materialized_module;
  // Dependency table for a materialized provider module.
  loom_symbol_dependency_table_t dependency_table;
} loom_link_plan_module_state_t;

struct loom_link_plan_t {
  // Provider index this plan selects from.
  const loom_link_module_index_t* index;
  // Host allocator for growable arrays.
  iree_allocator_t allocator;
  // Arena backing dependency tables.
  iree_arena_allocator_t arena;
  // Live symbol selections in stable plan order.
  loom_link_plan_symbol_t* symbols;
  // Number of live symbol selections.
  iree_host_size_t symbol_count;
  // Allocated live symbol capacity.
  iree_host_size_t symbol_capacity;
  // Selected bitset indexed by module-index symbol ordinal.
  uint8_t* selected_symbols;
  // Number of entries in selected_symbols.
  iree_host_size_t selected_symbol_count;
  // Per-module dependency-analysis cache.
  loom_link_plan_module_state_t* module_states;
  // Number of entries in module_states.
  iree_host_size_t module_state_count;
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_link_plan_normalize_symbol_name(
    iree_string_view_t name) {
  if (iree_string_view_starts_with_char(name, '@')) {
    return iree_string_view_remove_prefix(name, 1);
  }
  return name;
}

static iree_status_t loom_link_plan_reserve_symbols(loom_link_plan_t* plan,
                                                    iree_host_size_t count) {
  if (count <= plan->symbol_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      plan->allocator, count, sizeof(*plan->symbols), &plan->symbol_capacity,
      (void**)&plan->symbols);
}

static bool loom_link_plan_symbol_is_stripped(
    const loom_link_plan_options_t* options, const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* symbol) {
  const loom_link_symbol_flags_t check_flags =
      LOOM_LINK_SYMBOL_FLAG_CHECK_CASE | LOOM_LINK_SYMBOL_FLAG_CHECK_BENCHMARK;
  if (options && options->check_policy == LOOM_LINK_PLAN_CHECK_STRIP &&
      iree_any_bit_set(symbol->flags, check_flags)) {
    return true;
  }
  return options && options->strip_symbol &&
         options->strip_symbol(options->strip_symbol_user_data, plan->index,
                               symbol);
}

static bool loom_link_plan_symbol_is_concrete_global(
    const loom_link_module_index_symbol_t* symbol) {
  if (!symbol || symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                          LOOM_LINK_SYMBOL_FLAG_IMPORT)) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    return false;
  }
  return iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY);
}

static bool loom_link_plan_symbol_is_declaration_like(
    const loom_link_module_index_symbol_t* symbol) {
  return symbol &&
         iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                             LOOM_LINK_SYMBOL_FLAG_IMPORT);
}

static bool loom_link_plan_symbol_has_body(
    const loom_link_module_index_symbol_t* symbol) {
  return symbol &&
         iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY);
}

static bool loom_link_plan_symbol_is_func_provider(
    const loom_link_module_index_symbol_t* symbol) {
  return symbol && (symbol->kind == LOOM_SYMBOL_FUNC_TEMPLATE ||
                    symbol->kind == LOOM_SYMBOL_FUNC_UKERNEL);
}

// Finds the first concrete body that can fill a selected declaration anchor.
// The body may be private: a declaration gives the linked output a global
// anchor, and the materialized linker can replace that anchor with a same-name
// private definition from a later provider.
static const loom_link_module_index_symbol_t*
loom_link_plan_find_concrete_duplicate_for_declaration(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* symbol) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* duplicate =
        loom_link_module_index_symbol_at(plan->index, i);
    if (duplicate == symbol ||
        !iree_string_view_equal(symbol->name, duplicate->name) ||
        loom_link_plan_symbol_is_declaration_like(duplicate) ||
        !loom_link_plan_symbol_has_body(duplicate)) {
      continue;
    }
    return duplicate;
  }
  return NULL;
}

static iree_status_t loom_link_plan_select_symbol(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_live_reason_t reason, iree_host_size_t cause_ordinal,
    iree_string_view_t root_name, iree_host_size_t* out_plan_ordinal) {
  if (out_plan_ordinal) {
    *out_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  if (!symbol || symbol->ordinal >= plan->selected_symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot select stale link-index symbol");
  }
  if (loom_link_plan_symbol_is_stripped(options, plan, symbol)) {
    if (options &&
        options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "required symbol '@%.*s' was stripped",
                            (int)symbol->name.size, symbol->name.data);
  }
  if (plan->selected_symbols[symbol->ordinal]) {
    if (out_plan_ordinal) {
      for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
        if (plan->symbols[i].symbol_ordinal == symbol->ordinal) {
          *out_plan_ordinal = i;
          break;
        }
      }
    }
    return iree_ok_status();
  }

  iree_host_size_t plan_ordinal = plan->symbol_count;
  IREE_RETURN_IF_ERROR(loom_link_plan_reserve_symbols(plan, plan_ordinal + 1));
  plan->symbols[plan_ordinal] = (loom_link_plan_symbol_t){
      .ordinal = plan_ordinal,
      .symbol_ordinal = symbol->ordinal,
      .reason = reason,
      .cause_ordinal = cause_ordinal,
      .root_name = root_name,
  };
  plan->selected_symbols[symbol->ordinal] = 1;
  plan->symbol_count = plan_ordinal + 1;
  if (out_plan_ordinal) {
    *out_plan_ordinal = plan_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_global_reference(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* referenced_symbol,
    loom_link_plan_live_reason_t reason, iree_host_size_t cause_ordinal,
    iree_string_view_t root_name, iree_host_size_t* out_plan_ordinal) {
  if (out_plan_ordinal) {
    *out_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  const loom_link_module_index_symbol_t* selected_symbol =
      loom_link_module_index_lookup_global(plan->index,
                                           referenced_symbol->name);
  if (!selected_symbol) {
    if (options &&
        options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "unresolved global dependency '@%.*s'",
        (int)referenced_symbol->name.size, referenced_symbol->name.data);
  }

  iree_host_size_t selected_plan_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  IREE_RETURN_IF_ERROR(loom_link_plan_select_symbol(
      plan, options, selected_symbol, reason, cause_ordinal, root_name,
      &selected_plan_ordinal));
  if (out_plan_ordinal) {
    *out_plan_ordinal = selected_plan_ordinal;
  }

  if (!loom_link_plan_symbol_is_declaration_like(selected_symbol)) {
    return iree_ok_status();
  }
  const loom_link_module_index_symbol_t* concrete_symbol =
      loom_link_plan_find_concrete_duplicate_for_declaration(plan,
                                                             selected_symbol);
  if (!concrete_symbol) {
    return iree_ok_status();
  }
  return loom_link_plan_select_symbol(
      plan, options, concrete_symbol, LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      selected_plan_ordinal, iree_string_view_empty(),
      /*out_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_build_module_dependencies(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_host_size_t module_ordinal,
    loom_link_plan_module_state_t** out_state) {
  *out_state = NULL;
  if (module_ordinal >= plan->module_state_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module ordinal %" PRIhsz " is out of range",
                            module_ordinal);
  }
  loom_link_plan_module_state_t* state = &plan->module_states[module_ordinal];
  if (state->dependencies_built) {
    *out_state = state;
    return iree_ok_status();
  }

  const loom_link_module_index_module_t* module =
      loom_link_module_index_module_at(plan->index, module_ordinal);
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module ordinal %" PRIhsz " is missing",
                            module_ordinal);
  }

  const loom_module_t* dependency_module = module->materialized_module;
  if (!dependency_module && options && options->materialize_module) {
    IREE_RETURN_IF_ERROR(
        options->materialize_module(options->materialize_module_user_data,
                                    plan->index, module, &dependency_module));
  }
  if (!dependency_module) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selective dependency closure for module %" PRIhsz
        " requires materialized IR or serialized symbol-use metadata",
        module_ordinal);
  }
  if (dependency_module->symbols.count != module->symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialized module '%.*s' has %" PRIhsz
                            " symbols but index metadata has %" PRIhsz,
                            (int)module->name.size, module->name.data,
                            dependency_module->symbols.count,
                            module->symbol_count);
  }
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(
      dependency_module, &plan->arena, &state->dependency_table));
  state->materialized_module = dependency_module;
  state->dependencies_built = true;
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_dependency_target(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module,
    loom_symbol_id_t target_symbol_id, iree_host_size_t cause_ordinal) {
  if (target_symbol_id >= module->symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dependency target symbol id %u is out of range for module '%.*s'",
        (unsigned)target_symbol_id, (int)module->name.size, module->name.data);
  }

  const loom_link_module_index_symbol_t* local_target =
      loom_link_module_index_symbol_at(
          plan->index, module->symbol_start_ordinal + target_symbol_id);
  if (!local_target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency target index record is missing");
  }

  const loom_link_module_index_symbol_t* selected_target = local_target;
  if (local_target->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return loom_link_plan_select_global_reference(
        plan, options, local_target, LOOM_LINK_PLAN_LIVE_DEPENDENCY,
        cause_ordinal, iree_string_view_empty(),
        /*out_plan_ordinal=*/NULL);
  }

  return loom_link_plan_select_symbol(
      plan, options, selected_target, LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      cause_ordinal, iree_string_view_empty(), /*out_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_scan_module_edges(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module,
    loom_link_plan_module_state_t* state, iree_host_size_t cause_ordinal) {
  if (state->module_edges_scanned) {
    return iree_ok_status();
  }
  state->module_edges_scanned = true;
  loom_symbol_dependency_edge_id_t edge_id =
      state->dependency_table.first_module_edge_id;
  while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
    const loom_symbol_dependency_edge_t* edge =
        &state->dependency_table.edges[edge_id];
    IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
        plan, options, module, edge->target_symbol_id, cause_ordinal));
    edge_id = edge->next_outgoing_edge_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_symbol_dependencies(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_host_size_t plan_ordinal) {
  const loom_link_plan_symbol_t* planned_symbol = &plan->symbols[plan_ordinal];
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index,
                                       planned_symbol->symbol_ordinal);
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(plan->index, symbol);
  if (!symbol || !module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "planned symbol record is stale");
  }
  if (!iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY)) {
    return iree_ok_status();
  }

  loom_link_plan_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_link_plan_build_module_dependencies(
      plan, options, module->ordinal, &state));
  IREE_RETURN_IF_ERROR(loom_link_plan_scan_module_edges(plan, options, module,
                                                        state, plan_ordinal));

  if (symbol->module_symbol_ordinal >= state->dependency_table.symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "planned symbol ordinal %" PRIhsz
                            " exceeds dependency table size",
                            symbol->module_symbol_ordinal);
  }
  const loom_symbol_dependency_symbol_edges_t* edges =
      &state->dependency_table.symbols[symbol->module_symbol_ordinal];
  loom_symbol_dependency_edge_id_t edge_id = edges->first_outgoing_edge_id;
  while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
    const loom_symbol_dependency_edge_t* edge =
        &state->dependency_table.edges[edge_id];
    IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
        plan, options, module, edge->target_symbol_id, plan_ordinal));
    edge_id = edge->next_outgoing_edge_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_provider_contract(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_string_view_t contract, iree_host_size_t cause_ordinal) {
  if (iree_string_view_is_empty(contract)) return iree_ok_status();
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* provider =
        loom_link_module_index_symbol_at(plan->index, i);
    if (!loom_link_plan_symbol_is_func_provider(provider) ||
        !iree_string_view_equal(provider->provider_contract, contract)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_select_symbol(
        plan, options, provider, LOOM_LINK_PLAN_LIVE_PROVIDER, cause_ordinal,
        iree_string_view_empty(), /*out_plan_ordinal=*/NULL));
  }
  return iree_ok_status();
}

typedef struct loom_link_plan_apply_dependency_walk_t {
  // Active plan being expanded.
  loom_link_plan_t* plan;

  // Planning options controlling selection and strip behavior.
  const loom_link_plan_options_t* options;

  // Module whose function body is being scanned.
  const loom_module_t* module;

  // Plan ordinal of the function containing the apply site.
  iree_host_size_t cause_ordinal;
} loom_link_plan_apply_dependency_walk_t;

static iree_status_t loom_link_plan_visit_apply_dependency(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_func_apply_isa(op)) return iree_ok_status();

  loom_link_plan_apply_dependency_walk_t* walk =
      (loom_link_plan_apply_dependency_walk_t*)user_data;
  loom_string_id_t contract_id = loom_func_apply_contract(op);
  if (contract_id == LOOM_STRING_ID_INVALID ||
      contract_id >= walk->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.apply has an invalid contract string id");
  }
  return loom_link_plan_select_provider_contract(
      walk->plan, walk->options, walk->module->strings.entries[contract_id],
      walk->cause_ordinal);
}

static iree_status_t loom_link_plan_expand_apply_dependencies(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_host_size_t plan_ordinal) {
  const loom_link_plan_symbol_t* planned_symbol = &plan->symbols[plan_ordinal];
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index,
                                       planned_symbol->symbol_ordinal);
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(plan->index, symbol);
  if (!symbol || !module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "planned symbol record is stale");
  }
  if (!iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY)) {
    return iree_ok_status();
  }

  loom_link_plan_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_link_plan_build_module_dependencies(
      plan, options, module->ordinal, &state));
  const loom_module_t* dependency_module = state->materialized_module;
  if (!dependency_module) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selective apply dependency closure for module %" PRIhsz
        " requires materialized IR or serialized contract-use metadata",
        module->ordinal);
  }
  if (symbol->module_symbol_ordinal >= dependency_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "planned symbol ordinal %" PRIhsz
                            " exceeds materialized module symbol count",
                            symbol->module_symbol_ordinal);
  }
  loom_symbol_t* source_symbol =
      &dependency_module->symbols.entries[symbol->module_symbol_ordinal];
  loom_func_like_t function =
      loom_func_like_cast(dependency_module, source_symbol->defining_op);
  if (!loom_func_like_isa(function)) return iree_ok_status();

  loom_link_plan_apply_dependency_walk_t walk = {
      .plan = plan,
      .options = options,
      .module = dependency_module,
      .cause_ordinal = plan_ordinal,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(dependency_module, function, LOOM_WALK_PRE_ORDER,
                            (loom_walk_callback_t){
                                .fn = loom_link_plan_visit_apply_dependency,
                                .user_data = &walk,
                            },
                            &plan->arena, &walk_result);
}

static iree_status_t loom_link_plan_check_duplicate_globals(
    const loom_link_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index,
                                         plan->symbols[i].symbol_ordinal);
    if (!loom_link_plan_symbol_is_concrete_global(symbol)) {
      continue;
    }
    for (iree_host_size_t j = i + 1; j < plan->symbol_count; ++j) {
      const loom_link_module_index_symbol_t* duplicate =
          loom_link_module_index_symbol_at(plan->index,
                                           plan->symbols[j].symbol_ordinal);
      if (!loom_link_plan_symbol_is_concrete_global(duplicate)) {
        continue;
      }
      if (iree_string_view_equal(symbol->name, duplicate->name)) {
        return loom_link_module_index_duplicate_global_status(
            plan->index, symbol, duplicate);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_archive(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, i);
    if (loom_link_plan_symbol_is_stripped(options, plan, symbol)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_select_symbol(
        plan, options, symbol, LOOM_LINK_PLAN_LIVE_ARCHIVE,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, iree_string_view_empty(),
        /*out_plan_ordinal=*/NULL));
  }
  return loom_link_plan_check_duplicate_globals(plan);
}

static iree_status_t loom_link_plan_select_root(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_string_view_t root_name) {
  iree_string_view_t normalized_name =
      loom_link_plan_normalize_symbol_name(root_name);
  if (iree_string_view_is_empty(normalized_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root symbol name must not be empty");
  }
  const loom_link_module_index_symbol_t* root =
      loom_link_module_index_lookup_global(plan->index, normalized_name);
  if (root) {
    return loom_link_plan_select_global_reference(
        plan, options, root, LOOM_LINK_PLAN_LIVE_ROOT,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, root_name,
        /*out_plan_ordinal=*/NULL);
  }

  const loom_link_module_index_symbol_t* private_root = NULL;
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, i);
    if (symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_PRIVATE ||
        !iree_string_view_equal(symbol->name, normalized_name)) {
      continue;
    }
    if (private_root) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "root symbol '@%.*s' is private in multiple modules",
          (int)normalized_name.size, normalized_name.data);
    }
    private_root = symbol;
  }
  if (!private_root) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "root symbol '@%.*s' was not found",
                            (int)normalized_name.size, normalized_name.data);
  }
  return loom_link_plan_select_symbol(
      plan, options, private_root, LOOM_LINK_PLAN_LIVE_ROOT,
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, root_name,
      /*out_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_select_exported_roots(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, i);
    if (!iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_select_symbol(
        plan, options, symbol, LOOM_LINK_PLAN_LIVE_ROOT,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, symbol->name,
        /*out_plan_ordinal=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_roots(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  if (options && options->include_exported_roots) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_exported_roots(plan, options));
  }
  const iree_host_size_t root_count = options ? options->root_symbols.count : 0;
  if (root_count > 0 && (!options || !options->root_symbols.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_root(
        plan, options, options->root_symbols.values[i]));
  }
  if (plan->symbol_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selective link planning requires at least one root");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_selective(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  IREE_RETURN_IF_ERROR(loom_link_plan_select_roots(plan, options));
  for (iree_host_size_t i = 0; i < plan->symbol_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_link_plan_expand_symbol_dependencies(plan, options, i));
    IREE_RETURN_IF_ERROR(
        loom_link_plan_expand_apply_dependencies(plan, options, i));
  }
  return loom_link_plan_check_duplicate_globals(plan);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_link_plan_build(const loom_link_module_index_t* index,
                                   const loom_link_plan_options_t* options,
                                   iree_arena_block_pool_t* block_pool,
                                   iree_allocator_t allocator,
                                   loom_link_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;

  loom_link_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->index = index;
  plan->allocator = allocator;
  iree_arena_initialize(block_pool, &plan->arena);

  iree_status_t status = iree_ok_status();
  plan->selected_symbol_count = loom_link_module_index_symbol_count(index);
  if (iree_status_is_ok(status) && plan->selected_symbol_count > 0) {
    status = iree_allocator_malloc_array(allocator, plan->selected_symbol_count,
                                         sizeof(*plan->selected_symbols),
                                         (void**)&plan->selected_symbols);
    if (iree_status_is_ok(status)) {
      memset(plan->selected_symbols, 0,
             plan->selected_symbol_count * sizeof(*plan->selected_symbols));
    }
  }

  plan->module_state_count = loom_link_module_index_module_count(index);
  if (iree_status_is_ok(status) && plan->module_state_count > 0) {
    status = iree_allocator_malloc_array(allocator, plan->module_state_count,
                                         sizeof(*plan->module_states),
                                         (void**)&plan->module_states);
    if (iree_status_is_ok(status)) {
      memset(plan->module_states, 0,
             plan->module_state_count * sizeof(*plan->module_states));
    }
  }

  loom_link_plan_mode_t mode = options ? options->mode : LOOM_LINK_PLAN_ARCHIVE;
  if (iree_status_is_ok(status)) {
    switch (mode) {
      case LOOM_LINK_PLAN_ARCHIVE:
        status = loom_link_plan_select_archive(plan, options);
        break;
      case LOOM_LINK_PLAN_SELECTIVE:
        status = loom_link_plan_select_selective(plan, options);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown link plan mode %u", (unsigned)mode);
        break;
    }
  }

  if (!iree_status_is_ok(status)) {
    loom_link_plan_free(plan);
    return status;
  }
  *out_plan = plan;
  return iree_ok_status();
}

void loom_link_plan_free(loom_link_plan_t* plan) {
  if (!plan) return;
  iree_allocator_free(plan->allocator, plan->module_states);
  iree_allocator_free(plan->allocator, plan->selected_symbols);
  iree_allocator_free(plan->allocator, plan->symbols);
  iree_arena_deinitialize(&plan->arena);
  iree_allocator_free(plan->allocator, plan);
}

const loom_link_module_index_t* loom_link_plan_index(
    const loom_link_plan_t* plan) {
  return plan ? plan->index : NULL;
}

iree_host_size_t loom_link_plan_symbol_count(const loom_link_plan_t* plan) {
  return plan ? plan->symbol_count : 0;
}

const loom_link_plan_symbol_t* loom_link_plan_symbol_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal) {
  if (!plan || ordinal >= plan->symbol_count) return NULL;
  return &plan->symbols[ordinal];
}

bool loom_link_plan_contains_symbol(const loom_link_plan_t* plan,
                                    iree_host_size_t symbol_ordinal) {
  return plan && symbol_ordinal < plan->selected_symbol_count &&
         plan->selected_symbols[symbol_ordinal];
}
