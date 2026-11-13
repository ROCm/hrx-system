// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/dead_state.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/callable_effects.h"
#include "loom/ops/op_defs.h"

enum {
  LOOM_CFG_VALUE_LIVE = 1u << 0,
  LOOM_CFG_PRODUCER_EXPANDED = 1u << 1,
  LOOM_CFG_VALUE_EXTERNAL_OWNER = 1u << 2,
};

typedef struct loom_dead_state_group_t {
  // Disjoint-set representative used while constructing lifetime components.
  loom_value_ordinal_t parent;
  // Next member in the component's circular list.
  loom_value_ordinal_t next;
  // Component size at a representative, used to bound union depth.
  loom_value_ordinal_t size;
} loom_dead_state_group_t;

typedef struct loom_dead_state_operation_t {
  // Operation whose classification is retained through erasure planning.
  loom_op_t* op;
  // Effect and ownership classification, independent of use counts.
  loom_dead_state_op_kind_t kind;
  // First result or consumed input, or INVALID for a value-free operation.
  loom_value_ordinal_t anchor;
} loom_dead_state_operation_t;

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
  // Connected lifetime obligations, or NULL without managed operations.
  loom_dead_state_group_t* groups;
  // Source-order operation inventory with retained effect classifications.
  loom_dead_state_operation_t* operations;
  // Number of initialized operation rows.
  iree_host_size_t operation_count;
} loom_cfg_dead_state_t;

static loom_type_t loom_dead_state_value_type(const loom_module_t* module,
                                              loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  const loom_type_t* element = loom_type_register_value_type(type);
  return element ? *element : type;
}

static bool loom_dead_state_consumes(const loom_module_t* module,
                                     const loom_op_t* op, uint16_t index) {
  const loom_value_id_t value_id = loom_op_const_operands(op)[index];
  const loom_type_t type = loom_dead_state_value_type(module, value_id);
  if (!loom_type_is_reference(type)) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t use =
      loom_value_uses(value)[loom_op_operand_use_indices(op)[index]];
  if (loom_use_flags(use) & LOOM_USE_FLAG_CONSUMES) {
    return true;
  }
  const loom_call_like_t call = loom_call_like_cast(module, (loom_op_t*)op);
  return (op->traits & LOOM_TRAIT_CALLABLE_BOUNDARY) &&
         (!loom_call_like_isa(call) ||
          index >= loom_call_like_operand_offset(call)) &&
         loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_OWNED;
}

loom_dead_state_op_kind_t loom_dead_state_classify_op(
    void* user_data, const loom_module_t* module, const loom_op_t* op) {
  (void)user_data;
  const loom_op_semantics_t semantics = loom_op_semantics(module, op);
  if (semantics.lifetime_effects_only) {
    if (op->operand_count == 0 && op->result_count == 0) {
      return LOOM_DEAD_STATE_OP_VALUE;
    }
    // Register transfer declarations also cover ordinary scalar carriers.
    const loom_value_id_t anchor = op->result_count
                                       ? loom_op_const_results(op)[0]
                                       : loom_op_const_operands(op)[0];
    if (loom_type_is_reference(loom_dead_state_value_type(module, anchor))) {
      return LOOM_DEAD_STATE_OP_LIFETIME;
    }
  }
  const bool qualified_call =
      loom_callable_effects_application_contract(op, semantics) !=
      LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  if (!qualified_call && !loom_op_is_dead_if_unused(module, op)) {
    return LOOM_DEAD_STATE_OP_ROOT;
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_dead_state_consumes(module, op, i)) {
      return LOOM_DEAD_STATE_OP_CONSUMING;
    }
  }
  return LOOM_DEAD_STATE_OP_VALUE;
}

static bool loom_cfg_dead_state_prunable_branch(const loom_op_t* op) {
  return loom_direct_branch_can_rebuild_payload(op);
}

static bool loom_dead_state_is_local(const loom_cfg_dead_state_t* state,
                                     loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->graph->module, value_id);
  const loom_block_t* block = loom_value_is_block_arg(value)
                                  ? loom_value_def_block(value)
                                  : loom_value_def_op(value)->parent_block;
  return block->parent_region == state->graph->region;
}

static loom_value_ordinal_t loom_dead_state_find(loom_cfg_dead_state_t* state,
                                                 loom_value_ordinal_t value) {
  while (state->groups[value].parent != value) {
    const loom_value_ordinal_t parent = state->groups[value].parent;
    state->groups[value].parent = state->groups[parent].parent;
    value = state->groups[value].parent;
  }
  return value;
}

static void loom_dead_state_join(loom_cfg_dead_state_t* state,
                                 loom_value_ordinal_t left,
                                 loom_value_ordinal_t right) {
  left = loom_dead_state_find(state, left);
  right = loom_dead_state_find(state, right);
  if (left == right) {
    return;
  }
  if (state->groups[left].size < state->groups[right].size) {
    const loom_value_ordinal_t temporary = left;
    left = right;
    right = temporary;
  }
  state->groups[right].parent = left;
  state->groups[left].size += state->groups[right].size;
  const loom_value_ordinal_t next = state->groups[left].next;
  state->groups[left].next = state->groups[right].next;
  state->groups[right].next = next;
}

static void loom_cfg_dead_state_mark(loom_cfg_dead_state_t* state,
                                     loom_value_id_t value_id) {
  // The enclosing region owns captures; nested regions keep their own state.
  if (!loom_dead_state_is_local(state, value_id)) {
    return;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(state->domain, value_id);
  if (state->values[ordinal] & LOOM_CFG_VALUE_LIVE) {
    return;
  }
  loom_value_ordinal_t member = ordinal;
  do {
    state->values[member] |= LOOM_CFG_VALUE_LIVE;
    state->worklist[state->worklist_count++] = state->domain->value_ids[member];
    member = state->groups ? state->groups[member].next : ordinal;
  } while (member != ordinal);
}

static iree_status_t loom_dead_state_inventory(
    loom_cfg_dead_state_t* state, iree_arena_allocator_t* arena,
    loom_dead_state_classify_fn_t classify, void* user_data) {
  iree_host_size_t capacity = 0;
  bool has_lifetimes = false;
  for (uint16_t i = 0; i < state->graph->block_count; ++i) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(state->graph->blocks[i].block, op) {
      if (loom_cfg_dead_state_prunable_branch(op)) {
        continue;
      }
      if (state->operation_count == capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, state->operation_count, state->operation_count + 1,
            sizeof(*state->operations), &capacity, (void**)&state->operations));
      }
      const loom_dead_state_op_kind_t kind =
          op->region_count ? LOOM_DEAD_STATE_OP_ROOT
                           : classify(user_data, state->graph->module, op);
      state->operations[state->operation_count++] =
          (loom_dead_state_operation_t){
              .op = op,
              .kind = kind,
              .anchor = op->result_count
                            ? loom_local_value_domain_ordinal(
                                  state->domain, loom_op_const_results(op)[0])
                            : LOOM_VALUE_ORDINAL_INVALID,
          };
      has_lifetimes |= kind == LOOM_DEAD_STATE_OP_LIFETIME ||
                       kind == LOOM_DEAD_STATE_OP_CONSUMING;
    }
  }
  if (!has_lifetimes) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, state->domain->value_count, sizeof(*state->groups),
      (void**)&state->groups));
  for (loom_value_ordinal_t i = 0; i < state->domain->value_count; ++i) {
    state->groups[i] = (loom_dead_state_group_t){i, i, 1};
  }
  for (iree_host_size_t i = 0; i < state->operation_count; ++i) {
    loom_dead_state_operation_t* row = &state->operations[i];
    const bool lifetime_only = row->kind == LOOM_DEAD_STATE_OP_LIFETIME;
    if (!lifetime_only && row->kind != LOOM_DEAD_STATE_OP_CONSUMING) {
      continue;
    }
    for (uint16_t j = 0; j < row->op->operand_count; ++j) {
      if (!lifetime_only &&
          !loom_dead_state_consumes(state->graph->module, row->op, j)) {
        continue;
      }
      const loom_value_id_t operand = loom_op_const_operands(row->op)[j];
      if (!loom_dead_state_is_local(state, operand)) {
        row->kind = LOOM_DEAD_STATE_OP_ROOT;
        continue;
      }
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_ordinal(state->domain, operand);
      if (row->anchor == LOOM_VALUE_ORDINAL_INVALID) {
        row->anchor = ordinal;
      } else {
        loom_dead_state_join(state, row->anchor, ordinal);
      }
    }
    if (lifetime_only) {
      for (uint16_t j = 1; j < row->op->result_count; ++j) {
        loom_dead_state_join(
            state, row->anchor,
            loom_local_value_domain_ordinal(state->domain,
                                            loom_op_const_results(row->op)[j]));
      }
    }
  }
  // Managed branch payloads transfer their exact lifetime obligation. Scalar
  // arguments retain the ordinary backward-only demand relation.
  for (iree_host_size_t i = 0; i < state->graph->edge_count; ++i) {
    const loom_cfg_edge_info_t* edge = &state->graph->edges[i];
    const loom_op_t* branch = edge->terminator;
    if (!loom_cfg_dead_state_prunable_branch(branch)) {
      continue;
    }
    const loom_block_t* destination =
        state->graph->blocks[edge->target_block_index].block;
    const loom_value_slice_t incoming = loom_direct_branch_args(branch);
    for (uint16_t j = 0; j < destination->arg_count; ++j) {
      if (!loom_type_is_reference(loom_dead_state_value_type(
              state->graph->module, destination->arg_ids[j]))) {
        continue;
      }
      const loom_value_ordinal_t target = loom_local_value_domain_ordinal(
          state->domain, destination->arg_ids[j]);
      if (loom_dead_state_is_local(state, incoming.values[j])) {
        loom_dead_state_join(
            state, target,
            loom_local_value_domain_ordinal(state->domain, incoming.values[j]));
      } else {
        state->values[target] |= LOOM_CFG_VALUE_EXTERNAL_OWNER;
      }
    }
  }
  return iree_ok_status();
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
      preserve_arguments |= !loom_cfg_dead_state_prunable_branch(
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
  }
  for (iree_host_size_t i = 0; i < state->operation_count; ++i) {
    const loom_dead_state_operation_t* row = &state->operations[i];
    if (row->kind == LOOM_DEAD_STATE_OP_ROOT) {
      if (row->op->result_count != 0) {
        loom_cfg_dead_state_mark(state, loom_op_const_results(row->op)[0]);
      } else {
        IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
            graph->module, row->op, loom_cfg_dead_state_mark_reference, state));
      }
    }
  }
  for (loom_value_ordinal_t i = 0; i < state->domain->value_count; ++i) {
    if (state->values[i] & LOOM_CFG_VALUE_EXTERNAL_OWNER) {
      loom_cfg_dead_state_mark(state, state->domain->value_ids[i]);
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
        if (loom_cfg_dead_state_prunable_branch(branch)) {
          const loom_value_slice_t args = loom_direct_branch_args(branch);
          loom_cfg_dead_state_mark(state, args.values[argument]);
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
  loom_block_t* destination = loom_direct_branch_dest(branch);
  const loom_value_slice_t incoming = loom_direct_branch_args(branch);
  uint16_t count = 0;
  for (uint16_t i = 0; i < destination->arg_count; ++i) {
    if (loom_cfg_dead_state_live(state, destination->arg_ids[i])) {
      arguments[count++] = incoming.values[i];
    }
  }
  if (count == branch->operand_count) {
    return iree_ok_status();
  }
  return loom_rewriter_rebuild_direct_branch(rewriter, branch, destination,
                                             arguments, count);
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
  }
  for (iree_host_size_t i = 0; i < state->operation_count; ++i) {
    const loom_dead_state_operation_t* row = &state->operations[i];
    if (row->kind == LOOM_DEAD_STATE_OP_ROOT ||
        (row->anchor != LOOM_VALUE_ORDINAL_INVALID &&
         (state->values[row->anchor] & LOOM_CFG_VALUE_LIVE))) {
      continue;
    }
    if (dead_count == dead_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, dead_count, dead_count + 1, sizeof(*dead_ops), &dead_capacity,
          (void**)&dead_ops));
    }
    dead_ops[dead_count++] = row->op;
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
    if (loom_cfg_dead_state_prunable_branch(branch)) {
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
    loom_dead_state_classify_fn_t classify, void* user_data,
    iree_host_size_t* out_removed_arguments,
    iree_host_size_t* out_removed_operations) {
  *out_removed_arguments = 0;
  *out_removed_operations = 0;
  uint16_t maximum_arguments = 0;
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    const uint16_t count = graph->blocks[i].block->arg_count;
    maximum_arguments = iree_max(maximum_arguments, count);
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
    status = loom_dead_state_inventory(&state, &arena, classify, user_data);
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
