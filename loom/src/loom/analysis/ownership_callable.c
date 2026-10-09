// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership_callable.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/walk.h"

static bool loom_ownership_lifetime_policy_type_matches(
    const loom_ownership_lifetime_policy_t* policy, loom_type_t type) {
  return policy->family.type_matches(type, policy->family.user_data);
}

static bool loom_ownership_lifetime_type_has_policy_flags(
    const loom_ownership_lifetime_module_state_t* module_state,
    loom_type_t type, loom_ownership_lifetime_policy_flags_t flags) {
  const loom_ownership_lifetime_options_t* options = module_state->options;
  for (iree_host_size_t i = 0; i < options->policy_count; ++i) {
    const loom_ownership_lifetime_policy_t* policy = &options->policies[i];
    if (iree_all_bits_set(policy->flags, flags) &&
        loom_ownership_lifetime_policy_type_matches(policy, type)) {
      return true;
    }
  }
  return false;
}

static bool loom_ownership_lifetime_value_has_policy_flags(
    const loom_ownership_lifetime_module_state_t* module_state,
    loom_value_id_t value_id, loom_ownership_lifetime_policy_flags_t flags) {
  return loom_ownership_lifetime_type_has_policy_flags(
      module_state, loom_module_value_type(module_state->module, value_id),
      flags);
}

static iree_status_t loom_ownership_lifetime_initialize_summary(
    loom_ownership_lifetime_module_state_t* module_state,
    const loom_symbol_t* symbol,
    loom_ownership_lifetime_function_summary_t* summary) {
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
      !symbol->defining_op) {
    return iree_ok_status();
  }
  loom_func_like_t function =
      loom_func_like_cast(module_state->module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(function, &arg_count);
  const loom_region_descriptor_t* body_descriptor =
      loom_func_like_body_region_descriptor(module_state->module, function);
  *summary = (loom_ownership_lifetime_function_summary_t){
      .function = function,
      .body = loom_func_like_body(function),
      .body_exit_kind =
          body_descriptor ? body_descriptor->terminator : LOOM_OP_KIND_UNKNOWN,
      .arg_count = arg_count,
      .result_count = function.op->result_count,
  };
  if (summary->arg_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        module_state->options->arena, summary->arg_count,
        sizeof(*summary->arg_consumed), (void**)&summary->arg_consumed));
    memset(summary->arg_consumed, 0,
           summary->arg_count * sizeof(*summary->arg_consumed));
    if (arg_ids) {
      for (uint16_t i = 0; i < summary->arg_count; ++i) {
        summary->arg_consumed[i] =
            loom_ownership_lifetime_value_has_policy_flags(
                module_state, arg_ids[i],
                LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_ARGUMENTS);
        summary->transfers_ownership |= summary->arg_consumed[i];
      }
    }
  }
  if (summary->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        module_state->options->arena, summary->result_count,
        sizeof(*summary->result_states), (void**)&summary->result_states));
    memset(summary->result_states, 0,
           summary->result_count * sizeof(*summary->result_states));
    if (!summary->body) {
      const loom_value_id_t* result_ids = loom_op_const_results(function.op);
      for (uint16_t i = 0; i < summary->result_count; ++i) {
        bool owned = loom_ownership_lifetime_value_has_policy_flags(
            module_state, result_ids[i],
            LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_BODYLESS_RESULTS);
        summary->result_states[i] =
            owned ? LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED
                  : LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED;
        summary->transfers_ownership |= owned;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_ownership_lifetime_initialize_module_summaries(
    loom_ownership_lifetime_module_state_t* module_state) {
  module_state->summary_count = module_state->module->symbols.count;
  if (module_state->summary_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      module_state->options->arena, module_state->summary_count,
      sizeof(*module_state->summaries), (void**)&module_state->summaries));
  memset(module_state->summaries, 0,
         module_state->summary_count * sizeof(*module_state->summaries));
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_initialize_summary(
        module_state, &module_state->module->symbols.entries[i],
        &module_state->summaries[i]));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function graph
//===----------------------------------------------------------------------===//

static bool loom_ownership_lifetime_graph_callee_node(
    const loom_ownership_lifetime_graph_t* graph, loom_symbol_ref_t callee,
    iree_host_size_t* out_node) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= graph->symbol_to_node_count) {
    return false;
  }
  iree_host_size_t node = graph->symbol_to_node[callee.symbol_id];
  if (node == IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_node = node;
  return true;
}

typedef struct loom_ownership_lifetime_successor_walk_t {
  // Function graph adapter.
  const loom_ownership_lifetime_graph_t* graph;
  // Function whose structural ownership facts are being retained.
  loom_ownership_lifetime_function_summary_t* summary;
  // SCC successor visitor.
  loom_scc_successor_callback_t visitor;
} loom_ownership_lifetime_successor_walk_t;

static bool loom_ownership_lifetime_effect_is_in_domain(
    const loom_ownership_lifetime_module_state_t* state,
    loom_value_id_t value_id) {
  if (!state->materialize_options) {
    return true;
  }
  loom_type_t type = loom_module_value_type(state->module, value_id);
  for (iree_host_size_t i = 0; i < state->materialize_options->policy_count;
       ++i) {
    if (loom_ownership_lifetime_policy_type_matches(
            &state->materialize_options->policies[i], type)) {
      return true;
    }
  }
  return false;
}

static bool loom_ownership_lifetime_op_has_owned_effects(
    const loom_ownership_lifetime_module_state_t* state, const loom_op_t* op) {
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_ownership_operand_effect_t effect = {0};
    if (loom_ownership_operand_effect_at(state->module, op, i, &effect) &&
        effect.effect != LOOM_OPERAND_OWNERSHIP_BORROW &&
        loom_ownership_lifetime_effect_is_in_domain(state, effect.value_id)) {
      return true;
    }
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_ownership_result_effect_t effect = {0};
    if (loom_ownership_result_effect_at(state->module, op, i, &effect) &&
        (effect.effect == LOOM_RESULT_OWNERSHIP_FRESH ||
         effect.effect == LOOM_RESULT_OWNERSHIP_RETAINED) &&
        loom_ownership_lifetime_effect_is_in_domain(state, effect.value_id)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_ownership_lifetime_visit_successor_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_ownership_lifetime_successor_walk_t* walk =
      (loom_ownership_lifetime_successor_walk_t*)user_data;
  loom_ownership_lifetime_module_state_t* state = walk->graph->module_state;
  if (loom_func_like_isa(loom_func_like_cast(state->module, op))) {
    // A nested callable owns a separate execution scope and graph node.
    *out_result = LOOM_WALK_SKIP;
    return iree_ok_status();
  }
  if (op->region_count && !walk->summary->structured_op) {
    walk->summary->structured_op = op;
  }
  if (!walk->summary->has_owned_effects) {
    walk->summary->has_owned_effects =
        loom_ownership_lifetime_op_has_owned_effects(state, op);
  }
  loom_call_like_t call = loom_call_like_cast(state->module, op);
  if (!loom_call_like_isa(call) ||
      !loom_ownership_lifetime_call_kind_is_runtime(
          loom_call_like_kind(call))) {
    return iree_ok_status();
  }
  loom_symbol_ref_t callee = loom_call_like_callee(call);
  if (loom_symbol_ref_is_valid(callee) && callee.module_id == 0) {
    loom_ownership_lifetime_callee_t* edge = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->options->arena,
                                             sizeof(*edge), (void**)&edge));
    *edge = (loom_ownership_lifetime_callee_t){
        .summary = &state->summaries[callee.symbol_id],
        .next = walk->summary->callees,
    };
    walk->summary->callees = edge;
  }
  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_ownership_lifetime_graph_callee_node(
          walk->graph, loom_call_like_callee(call), &callee_node)) {
    return iree_ok_status();
  }
  return walk->visitor.fn(walk->visitor.user_data, callee_node);
}

static iree_status_t loom_ownership_lifetime_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t visitor) {
  const loom_ownership_lifetime_graph_t* graph =
      (const loom_ownership_lifetime_graph_t*)user_data;
  loom_ownership_lifetime_successor_walk_t walk = {
      .graph = graph,
      .summary = graph->nodes[node].summary,
      .visitor = visitor,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(
      graph->module_state->module, graph->nodes[node].summary->function,
      LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_ownership_lifetime_visit_successor_call,
                             &walk},
      &walk_result);
}

iree_status_t loom_ownership_lifetime_build_graph(
    loom_ownership_lifetime_module_state_t* module_state,
    iree_arena_allocator_t* arena, loom_ownership_lifetime_graph_t* out_graph,
    loom_scc_list_t* out_sccs) {
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module_state = module_state;
  out_graph->symbol_to_node_count = module_state->summary_count;
  if (module_state->summary_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_state->summary_count, sizeof(*out_graph->symbol_to_node),
      (void**)&out_graph->symbol_to_node));
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    out_graph->symbol_to_node[i] = IREE_HOST_SIZE_MAX;
  }

  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    if (module_state->summaries[i].body) {
      ++function_count;
    }
  }
  if (function_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, function_count,
                                                 sizeof(*out_graph->nodes),
                                                 (void**)&out_graph->nodes));
  out_graph->node_count = function_count;
  iree_host_size_t node = 0;
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    if (!module_state->summaries[i].body) {
      continue;
    }
    out_graph->nodes[node] = (loom_ownership_lifetime_graph_node_t){
        .summary = &module_state->summaries[i],
    };
    out_graph->symbol_to_node[i] = node;
    ++node;
  }

  loom_scc_graph_t scc_graph = {
      .node_count = function_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_ownership_lifetime_visit_successors, out_graph),
  };
  return loom_scc_compute(&scc_graph, NULL, arena, out_sccs);
}

const loom_op_t* loom_ownership_lifetime_cfg_boundary(
    const loom_ownership_lifetime_function_summary_t* summary) {
  if (!summary->structured_op) {
    return NULL;
  }
  if (summary->has_owned_effects || summary->transfers_ownership) {
    return summary->structured_op;
  }
  for (const loom_ownership_lifetime_callee_t* edge = summary->callees; edge;
       edge = edge->next) {
    if (edge->summary->transfers_ownership) {
      return summary->structured_op;
    }
  }
  return NULL;
}
