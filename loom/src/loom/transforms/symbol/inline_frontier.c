// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_frontier.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/provider.h"
#include "loom/transforms/symbol/inline_execution.h"
#include "loom/transforms/symbol/inline_preflight.h"
#include "loom/transforms/symbol/inline_profitability.h"
#include "loom/util/walk.h"

typedef struct loom_inline_body_cost_context_t {
  const loom_module_t* module;
  uint32_t cost;
} loom_inline_body_cost_context_t;

typedef struct loom_inline_frontier_symbol_t {
  // Number of final body instances under the current legal frontier.
  uint64_t materialization_count;
  // Instances introduced only because this body is currently non-leaf.
  uint64_t leaf_forced_materialization_count;
  // Cost after expansions required by the current legal frontier.
  uint32_t body_cost;
  // Number of authored call operations removed by the current frontier.
  uint32_t planned_call_removals;
  // Whether the current frontier preserves this symbol definition.
  bool definition_survives;
  // Whether the expanded body contains any retained native call.
  bool has_retained_call;
  // Whether a required child expansion carries a retained call into the body.
  bool has_propagated_call;
  // Whether the expanded body owns an entry resource.
  bool has_resources;
  // Whether the expanded body owns any explicit storage.
  bool has_storage;
  // Whether the expanded body owns workgroup storage.
  bool has_workgroup_storage;
} loom_inline_frontier_symbol_t;

static uint32_t loom_inline_type_materialization_cost(loom_type_t type) {
  uint64_t element_count = 1;
  if (loom_type_is_vector(type) || loom_type_is_tile(type) ||
      loom_type_is_tensor(type)) {
    if (!loom_type_static_element_count(type, &element_count)) {
      return UINT32_MAX;
    }
  } else if (!loom_type_is_scalar(type)) {
    return 1;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (element_bit_count <= 0) {
    return UINT32_MAX;
  }
  const uint64_t words_per_element = ((uint64_t)element_bit_count + 31u) / 32u;
  return element_count > UINT32_MAX / words_per_element
             ? UINT32_MAX
             : (uint32_t)(element_count * words_per_element);
}

static uint32_t loom_inline_op_materialization_cost(const loom_module_t* module,
                                                    loom_op_t* op) {
  // Calls are replaced as one unit when their body expands, and terminator
  // payloads disappear or become control-flow forwarding during composition.
  // Charging either payload here would count it again through the producers.
  if (loom_call_like_isa(loom_call_like_cast(module, op)) ||
      loom_op_has_trait(module, op, LOOM_TRAIT_TERMINATOR)) {
    return 1;
  }
  uint32_t cost = 1;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    cost = iree_max(
        cost, loom_inline_type_materialization_cost(
                  loom_module_value_type(module, loom_op_operands(op)[i])));
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    cost = iree_max(
        cost, loom_inline_type_materialization_cost(
                  loom_module_value_type(module, loom_op_results(op)[i])));
  }
  return cost;
}

static iree_status_t loom_inline_accumulate_body_cost(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_inline_body_cost_context_t* cost_context = user_data;
  const uint32_t op_cost =
      loom_inline_op_materialization_cost(cost_context->module, op);
  cost_context->cost = op_cost > UINT32_MAX - cost_context->cost
                           ? UINT32_MAX
                           : cost_context->cost + op_cost;
  *out_result = LOOM_WALK_CONTINUE;
  return iree_ok_status();
}

iree_status_t loom_inline_compute_direct_body_cost(
    loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id,
    uint32_t* out_cost) {
  loom_inline_body_cost_context_t cost_context = {
      .module = state->module,
  };
  loom_region_t* body = loom_func_like_body(state->symbols[symbol_id].function);
  if (!body) {
    *out_cost = 0;
    return iree_ok_status();
  }
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(
      loom_walk_region(state->module, body, LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           .fn = loom_inline_accumulate_body_cost,
                           .user_data = &cost_context,
                       },
                       &result));
  *out_cost = cost_context.cost;
  return iree_ok_status();
}

static bool loom_inline_frontier_is_native_edge(
    const loom_inline_plan_entry_t* entry) {
  if (entry->execution_scope != 0) {
    return false;
  }
  const loom_call_like_kind_t kind = loom_call_like_kind(entry->call);
  return kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL;
}

static bool loom_inline_frontier_can_require(
    loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  if (entry->action == LOOM_INLINE_PLAN_ACTION_ERROR ||
      entry->effective_policy == LOOM_INLINE_POLICY_NOINLINE) {
    return false;
  }
  if (((state->options.defer_callable_exits &&
        loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_SEMANTIC) ||
       loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_LOW_INTERNAL) &&
      state->symbols[entry->target_symbol_id].body_has_callable_exit) {
    return false;
  }
  return loom_inline_preflight_entry(state, entry) == LOOM_INLINE_BLOCKER_NONE;
}

static uint64_t loom_inline_frontier_saturating_add(uint64_t lhs,
                                                    uint64_t rhs) {
  return rhs > UINT64_MAX - lhs ? UINT64_MAX : lhs + rhs;
}

static uint64_t loom_inline_frontier_saturating_multiply(uint64_t lhs,
                                                         uint64_t rhs) {
  return lhs != 0 && rhs > UINT64_MAX / lhs ? UINT64_MAX : lhs * rhs;
}

static void loom_inline_frontier_simulate(
    loom_inline_callables_plan_t* state, const uint8_t* selected_entries,
    const uint8_t* can_require_entries, uint8_t* required_entries,
    uint8_t* leaf_forced_entries, const uint32_t* direct_body_costs,
    loom_inline_frontier_symbol_t* symbols) {
  const iree_host_size_t symbol_count = state->module->symbols.count;
  memset(required_entries, 0, state->entry_count * sizeof(*required_entries));
  memset(leaf_forced_entries, 0,
         state->entry_count * sizeof(*leaf_forced_entries));
  memset(symbols, 0, symbol_count * sizeof(*symbols));

  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    const loom_inline_symbol_info_t* direct = &state->symbols[symbol_id];
    loom_inline_frontier_symbol_t* frontier = &symbols[symbol_id];
    frontier->body_cost = direct_body_costs[symbol_id];
    frontier->has_resources = direct->body_has_resources;
    frontier->has_storage = direct->body_has_storage;
    frontier->has_workgroup_storage = direct->body_has_workgroup_storage;
  }

  // SCCs are in callee-before-caller order, so each source sees the complete
  // expanded facts and body cost of every acyclic required child.
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const loom_symbol_id_t source_symbol_id = component->nodes[node_index];
      loom_inline_frontier_symbol_t* source = &symbols[source_symbol_id];
      for (uint32_t entry_index =
               state->first_graph_by_source[source_symbol_id];
           entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
           entry_index = state->entries[entry_index].next_graph_from_source) {
        const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
        if (entry->target_symbol_id >= symbol_count) {
          continue;
        }
        const loom_inline_frontier_symbol_t* target =
            &symbols[entry->target_symbol_id];
        bool required = selected_entries[entry_index] != 0 ||
                        entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED ||
                        entry->effective_policy == LOOM_INLINE_POLICY_INLINE;
        const bool can_require = can_require_entries[entry_index] != 0;
        if (!required && can_require &&
            ((iree_any_bit_set(entry->target_policy,
                               LOOM_TARGET_CALL_POLICY_NO_RESOURCES) &&
              target->has_resources) ||
             (iree_any_bit_set(entry->target_policy,
                               LOOM_TARGET_CALL_POLICY_NO_STORAGE) &&
              target->has_storage) ||
             (iree_any_bit_set(entry->target_policy,
                               LOOM_TARGET_CALL_POLICY_NO_WORKGROUP_STORAGE) &&
              target->has_workgroup_storage))) {
          required = true;
        }
        if (!required && can_require &&
            iree_any_bit_set(entry->target_policy,
                             LOOM_TARGET_CALL_POLICY_REQUIRE_LEAF) &&
            target->has_retained_call) {
          required = true;
          leaf_forced_entries[entry_index] = 1;
        }
        required_entries[entry_index] = required;
        if (required) {
          ++symbols[entry->target_symbol_id].planned_call_removals;
        }
        if (!loom_inline_frontier_is_native_edge(entry)) {
          continue;
        }
        if (required) {
          source->body_cost = loom_inline_profitability_add_expansion(
              source->body_cost, target->body_cost);
          source->has_retained_call |= target->has_retained_call;
          source->has_propagated_call |= target->has_retained_call;
          source->has_resources |= target->has_resources;
          source->has_storage |= target->has_storage;
          source->has_workgroup_storage |= target->has_workgroup_storage;
        } else {
          source->has_retained_call = true;
        }
      }
      if (component->is_cycle && source->has_retained_call) {
        source->body_cost = UINT32_MAX;
        source->has_propagated_call = true;
      }
    }
  }

  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    loom_inline_frontier_symbol_t* symbol = &symbols[symbol_id];
    symbol->definition_survives =
        !loom_inline_symbol_can_transfer_after_removing(
            state, (loom_symbol_id_t)symbol_id, symbol->planned_call_removals);
    symbol->materialization_count = symbol->definition_survives ? 1 : 0;
  }

  // Reverse component order is caller-before-callee. Each required call in a
  // materialized caller contributes one instance of its already-expanded body.
  for (iree_host_size_t reverse_index = state->sccs.count; reverse_index > 0;
       --reverse_index) {
    const loom_scc_t* component = &state->sccs.values[reverse_index - 1];
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const loom_symbol_id_t source_symbol_id = component->nodes[node_index];
      const uint64_t source_count =
          symbols[source_symbol_id].materialization_count;
      for (uint32_t entry_index =
               state->first_graph_by_source[source_symbol_id];
           entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
           entry_index = state->entries[entry_index].next_graph_from_source) {
        const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
        if (!required_entries[entry_index] ||
            !loom_inline_frontier_is_native_edge(entry) ||
            entry->target_symbol_id >= symbol_count) {
          continue;
        }
        loom_inline_frontier_symbol_t* target =
            &symbols[entry->target_symbol_id];
        target->materialization_count = loom_inline_frontier_saturating_add(
            target->materialization_count, source_count);
        if (leaf_forced_entries[entry_index]) {
          target->leaf_forced_materialization_count =
              loom_inline_frontier_saturating_add(
                  target->leaf_forced_materialization_count, source_count);
        }
      }
    }
  }
}

iree_status_t loom_inline_select_native_frontier(
    loom_inline_callables_plan_t* state, uint8_t** out_selected_entries) {
  *out_selected_entries = NULL;
  if (!state->has_native_constraints || state->entry_count == 0) {
    return iree_ok_status();
  }
  bool has_leaf_constraint = false;
  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    if (iree_any_bit_set(state->entries[entry_index].target_policy,
                         LOOM_TARGET_CALL_POLICY_REQUIRE_LEAF)) {
      has_leaf_constraint = true;
      break;
    }
  }
  if (!has_leaf_constraint) {
    return iree_ok_status();
  }
  const iree_host_size_t symbol_count = state->module->symbols.count;
  uint8_t* selected_entries = NULL;
  uint8_t* can_require_entries = NULL;
  uint8_t* required_entries = NULL;
  uint8_t* leaf_forced_entries = NULL;
  loom_inline_frontier_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->entry_count, sizeof(*selected_entries),
      (void**)&selected_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->entry_count, sizeof(*can_require_entries),
      (void**)&can_require_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->entry_count, sizeof(*required_entries),
      (void**)&required_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->entry_count, sizeof(*leaf_forced_entries),
      (void**)&leaf_forced_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, symbol_count, sizeof(*symbols), (void**)&symbols));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, symbol_count, sizeof(*state->direct_body_costs),
      (void**)&state->direct_body_costs));
  memset(selected_entries, 0, state->entry_count * sizeof(*selected_entries));
  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    loom_inline_symbol_info_t* source = &state->symbols[symbol_id];
    if (source->incoming_call_count == 0) {
      state->direct_body_costs[symbol_id] = 0;
      continue;
    }
    loom_inline_preflight_body(state, (loom_symbol_id_t)symbol_id);
    IREE_RETURN_IF_ERROR(loom_inline_compute_direct_body_cost(
        state, (loom_symbol_id_t)symbol_id,
        &state->direct_body_costs[symbol_id]));
  }
  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    can_require_entries[entry_index] =
        loom_inline_frontier_can_require(state, &state->entries[entry_index]);
  }

  // One round may turn a parent into a leaf and expose the same choice at its
  // caller. Re-simulate only graph summaries; body walks and preflight facts
  // above are cached for the complete fixed point.
  bool changed = false;
  do {
    loom_inline_frontier_simulate(state, selected_entries, can_require_entries,
                                  required_entries, leaf_forced_entries,
                                  state->direct_body_costs, symbols);
    changed = false;

    // A candidate flattens every direct retained native call into one reusable
    // definition. The comparison deliberately does not credit any callee
    // definition that may then disappear, keeping the decision conservative.
    for (iree_host_size_t component_index = 0;
         component_index < state->sccs.count; ++component_index) {
      const loom_scc_t* component = &state->sccs.values[component_index];
      if (component->is_cycle) {
        continue;
      }
      for (iree_host_size_t node_index = 0; node_index < component->node_count;
           ++node_index) {
        const loom_symbol_id_t source_symbol_id = component->nodes[node_index];
        const loom_inline_frontier_symbol_t* source =
            &symbols[source_symbol_id];
        if (!source->has_retained_call || source->has_propagated_call ||
            source->leaf_forced_materialization_count == 0 ||
            source->body_cost == UINT32_MAX) {
          continue;
        }
        uint32_t flattened_cost = source->body_cost;
        bool has_retained_edge = false;
        bool can_flatten = true;
        for (uint32_t entry_index =
                 state->first_graph_by_source[source_symbol_id];
             entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
             entry_index = state->entries[entry_index].next_graph_from_source) {
          const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
          if (required_entries[entry_index] ||
              !loom_inline_frontier_is_native_edge(entry)) {
            continue;
          }
          has_retained_edge = true;
          if (!can_require_entries[entry_index] ||
              symbols[entry->target_symbol_id].has_retained_call) {
            can_flatten = false;
            break;
          }
          flattened_cost = loom_inline_profitability_add_expansion(
              flattened_cost, symbols[entry->target_symbol_id].body_cost);
        }
        if (!has_retained_edge || !can_flatten) {
          continue;
        }

        const uint64_t current_count = source->materialization_count;
        const uint64_t leaf_forced_count =
            source->leaf_forced_materialization_count;
        if (leaf_forced_count > current_count) {
          continue;
        }
        uint64_t flattened_count = current_count - leaf_forced_count;
        if (!source->definition_survives) {
          flattened_count =
              loom_inline_frontier_saturating_add(flattened_count, 1);
        }
        const uint64_t current_total = loom_inline_frontier_saturating_multiply(
            current_count, source->body_cost);
        const uint64_t flattened_total =
            loom_inline_frontier_saturating_multiply(flattened_count,
                                                     flattened_cost);
        if (flattened_total >= current_total) {
          continue;
        }
        for (uint32_t entry_index =
                 state->first_graph_by_source[source_symbol_id];
             entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
             entry_index = state->entries[entry_index].next_graph_from_source) {
          if (!required_entries[entry_index] &&
              loom_inline_frontier_is_native_edge(
                  &state->entries[entry_index])) {
            selected_entries[entry_index] = 1;
            changed = true;
          }
        }
      }
    }
  } while (changed);

  *out_selected_entries = selected_entries;
  return iree_ok_status();
}
