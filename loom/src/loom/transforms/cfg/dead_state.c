// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/dead_state.h"

#include <string.h>

#include "loom/ir/value_refs.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

enum {
  LOOM_CFG_VALUE_LIVE = 1u << 0,
  LOOM_CFG_PRODUCER_EXPANDED = 1u << 1,
};

typedef struct loom_cfg_dead_state_t {
  // Current region structure; predecessor edges supply argument definitions.
  const loom_cfg_graph_t* graph;
  // Borrowed function-local ordinal map, unchanged during the trace.
  const loom_local_value_domain_t* domain;
  // Liveness and producer-visit bits indexed by local value ordinal.
  uint8_t* values;
  // Values whose defining operation or incoming edges remain to be visited.
  loom_value_id_t* worklist;
  // Number of pending values; each local definition enters at most once.
  iree_host_size_t worklist_count;
} loom_cfg_dead_state_t;

static bool loom_cfg_dead_state_direct_branch(const loom_op_t* op) {
  return loom_cfg_br_isa(op) || loom_low_br_isa(op);
}

static void loom_cfg_dead_state_mark(loom_cfg_dead_state_t* state,
                                     loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->graph->module, value_id);
  const loom_block_t* block = loom_value_is_block_arg(value)
                                  ? loom_value_def_block(value)
                                  : loom_value_def_op(value)->parent_block;
  // The enclosing region owns captures; nested regions keep their own state.
  if (block->parent_region != state->graph->region) {
    return;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(state->domain, value_id);
  if (!(state->values[ordinal] & LOOM_CFG_VALUE_LIVE)) {
    state->values[ordinal] |= LOOM_CFG_VALUE_LIVE;
    state->worklist[state->worklist_count++] = value_id;
  }
}

static iree_status_t loom_cfg_dead_state_mark_reference(
    loom_value_id_t provider, loom_value_id_t type_carrier, void* user_data) {
  (void)type_carrier;
  loom_cfg_dead_state_mark(user_data, provider);
  return iree_ok_status();
}

static void loom_cfg_dead_state_mark_type(loom_cfg_dead_state_t* state,
                                          loom_value_id_t value_id) {
  loom_type_use_iterator_t dependencies;
  loom_module_value_type_dependencies(state->graph->module, value_id,
                                      &dependencies);
  for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
       provider != LOOM_VALUE_ID_INVALID;
       provider = loom_type_dependencies_next(&dependencies)) {
    loom_cfg_dead_state_mark(state, provider);
  }
}

static bool loom_cfg_dead_state_live(const loom_cfg_dead_state_t* state,
                                     loom_value_id_t value_id) {
  return state->values[loom_local_value_domain_ordinal(state->domain,
                                                       value_id)] &
         LOOM_CFG_VALUE_LIVE;
}

static iree_status_t loom_cfg_dead_state_seed(loom_cfg_dead_state_t* state) {
  const loom_cfg_graph_t* graph = state->graph;
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    const loom_block_t* block = graph->blocks[i].block;
    bool preserve_arguments = i == 0;
    const loom_cfg_edge_index_span_t incoming =
        loom_cfg_graph_predecessor_edges(graph, i);
    for (iree_host_size_t j = 0; j < incoming.count; ++j) {
      preserve_arguments |= !loom_cfg_dead_state_direct_branch(
          graph->edges[incoming.values[j]].terminator);
    }
    for (uint16_t j = 0; j < block->arg_count; ++j) {
      const loom_value_id_t argument = block->arg_ids[j];
      if (preserve_arguments) {
        loom_cfg_dead_state_mark(state, argument);
      }
      // Keep type binders until their carriers have been removed. The next
      // cleanup iteration can retire a binder with no remaining carrier.
      loom_cfg_dead_state_mark_type(state, argument);
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_cfg_dead_state_direct_branch(op)) {
        continue;
      }
      if (op->region_count != 0 ||
          !loom_op_is_dead_if_unused(graph->module, op)) {
        if (op->result_count != 0) {
          loom_cfg_dead_state_mark(state, loom_op_const_results(op)[0]);
        } else {
          IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
              graph->module, op, loom_cfg_dead_state_mark_reference, state));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_dead_state_trace(loom_cfg_dead_state_t* state) {
  const loom_cfg_graph_t* graph = state->graph;
  while (state->worklist_count != 0) {
    const loom_value_id_t value_id = state->worklist[--state->worklist_count];
    const loom_value_t* value = loom_module_value(graph->module, value_id);
    if (loom_value_is_block_arg(value)) {
      const loom_block_t* block = loom_value_def_block(value);
      const uint16_t argument = loom_value_def_index(value);
      const loom_cfg_edge_index_span_t incoming =
          loom_cfg_graph_predecessor_edges(graph, block->region_index);
      for (iree_host_size_t i = 0; i < incoming.count; ++i) {
        const loom_op_t* branch = graph->edges[incoming.values[i]].terminator;
        if (loom_cfg_dead_state_direct_branch(branch)) {
          loom_cfg_dead_state_mark(state,
                                   loom_op_const_operands(branch)[argument]);
        }
      }
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    const loom_value_id_t* results = loom_op_const_results(op);
    const loom_value_ordinal_t first =
        loom_local_value_domain_ordinal(state->domain, results[0]);
    if (state->values[first] & LOOM_CFG_PRODUCER_EXPANDED) {
      continue;
    }
    state->values[first] |= LOOM_CFG_PRODUCER_EXPANDED;
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_cfg_dead_state_mark(state, results[i]);
    }
    IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
        graph->module, op, loom_cfg_dead_state_mark_reference, state));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_dead_state_prune_branch(
    loom_rewriter_t* rewriter, const loom_cfg_dead_state_t* state,
    loom_op_t* branch, loom_value_id_t* arguments) {
  loom_block_t* destination = loom_op_successors(branch)[0];
  const loom_value_id_t* incoming = loom_op_const_operands(branch);
  uint16_t count = 0;
  for (uint16_t i = 0; i < destination->arg_count; ++i) {
    if (loom_cfg_dead_state_live(state, destination->arg_ids[i])) {
      arguments[count++] = incoming[i];
    }
  }
  if (count == branch->operand_count) {
    return iree_ok_status();
  }
  const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, branch);
  loom_op_t* replacement = NULL;
  const iree_status_t status =
      loom_cfg_br_isa(branch)
          ? loom_cfg_br_build(&rewriter->builder, destination, arguments, count,
                              branch->location, &replacement)
          : loom_low_br_build(&rewriter->builder, destination, arguments, count,
                              branch->location, &replacement);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return loom_rewriter_erase(rewriter, branch);
}

static iree_status_t loom_cfg_dead_state_apply(
    loom_rewriter_t* rewriter, const loom_cfg_dead_state_t* state,
    iree_arena_allocator_t* arena, uint16_t maximum_arguments,
    iree_host_size_t* out_removed_arguments,
    iree_host_size_t* out_removed_operations) {
  const loom_cfg_graph_t* graph = state->graph;
  loom_op_t** dead_ops = NULL;
  iree_host_size_t dead_count = 0;
  iree_host_size_t dead_capacity = 0;
  bool has_dead_arguments = false;
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    const loom_block_t* block = graph->blocks[i].block;
    for (uint16_t j = 0; j < block->arg_count; ++j) {
      has_dead_arguments |= !loom_cfg_dead_state_live(state, block->arg_ids[j]);
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->result_count != 0 &&
          !loom_cfg_dead_state_live(state, loom_op_const_results(op)[0])) {
        if (dead_count == dead_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              arena, dead_count, dead_count + 1, sizeof(*dead_ops),
              &dead_capacity, (void**)&dead_ops));
        }
        dead_ops[dead_count++] = op;
      }
    }
  }
  if (dead_count == 0 && !has_dead_arguments) {
    return iree_ok_status();
  }
  loom_value_id_t* arguments = NULL;
  bool* remove_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, maximum_arguments, sizeof(*arguments), (void**)&arguments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, maximum_arguments,
                                                 sizeof(*remove_arguments),
                                                 (void**)&remove_arguments));
  // Withdraw forwarding uses first so the operation set is closed. Erasing
  // it then releases every operand/attribute use of the dead block arguments.
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    loom_op_t* branch = graph->blocks[i].block->last_op;
    if (loom_cfg_dead_state_direct_branch(branch)) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_dead_state_prune_branch(rewriter, state, branch, arguments));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase_closed_set(rewriter, dead_ops, dead_count));
  *out_removed_operations = dead_count;
  for (uint16_t i = 1; i < graph->block_count; ++i) {
    loom_block_t* block = (loom_block_t*)graph->blocks[i].block;
    for (uint16_t j = 0; j < block->arg_count; ++j) {
      remove_arguments[j] = !loom_cfg_dead_state_live(state, block->arg_ids[j]);
    }
    uint16_t removed = 0;
    IREE_RETURN_IF_ERROR(loom_rewriter_remove_block_args(
        rewriter, block, remove_arguments, block->arg_count, &removed));
    *out_removed_arguments += removed;
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_eliminate_dead_state(
    loom_rewriter_t* rewriter, const loom_cfg_graph_t* graph,
    const loom_local_value_domain_t* domain,
    iree_host_size_t* out_removed_arguments,
    iree_host_size_t* out_removed_operations) {
  *out_removed_arguments = 0;
  *out_removed_operations = 0;
  if (!graph->has_cycles) {
    return iree_ok_status();
  }
  uint16_t maximum_arguments = 0;
  bool has_carried_state = false;
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    const uint16_t count = graph->blocks[i].block->arg_count;
    has_carried_state |= i != 0 && count != 0;
    maximum_arguments = iree_max(maximum_arguments, count);
  }
  if (!has_carried_state) {
    return iree_ok_status();
  }
  iree_arena_allocator_t arena;
  iree_arena_initialize(rewriter->arena->block_pool, &arena);
  loom_cfg_dead_state_t state = {.graph = graph, .domain = domain};
  iree_status_t status =
      iree_arena_allocate_array(&arena, domain->value_count,
                                sizeof(*state.values), (void**)&state.values);
  if (iree_status_is_ok(status)) {
    memset(state.values, 0, domain->value_count * sizeof(*state.values));
    status = iree_arena_allocate_array(&arena, domain->value_count,
                                       sizeof(*state.worklist),
                                       (void**)&state.worklist);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cfg_dead_state_seed(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cfg_dead_state_trace(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cfg_dead_state_apply(rewriter, &state, &arena,
                                       maximum_arguments, out_removed_arguments,
                                       out_removed_operations);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
