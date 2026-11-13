// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/channel_materialization.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/rewrite/callable.h"

typedef struct loom_channel_materialization_clone_state_t {
  // Immutable occurrence plan in the acquired source value domain.
  const loom_channel_plan_t* source;
  // Compact cloned value correspondence, indexed by source ordinal.
  loom_value_id_t* values;
  // Projected channel actions in the source collection's order.
  loom_channel_plan_action_t* actions;
  // Next source action in clone visitation order.
  iree_host_size_t action_index;
  // Projected callable exits in the source collection's order.
  loom_op_t** returns;
  // Next source exit in clone visitation order.
  iree_host_size_t return_index;
  // Additional occurrence products translated by the same clone traversal.
  loom_ir_clone_observer_t observer;
} loom_channel_materialization_clone_state_t;

static iree_status_t loom_channel_materialization_clone_op(
    void* user_data, const loom_op_t* source_op, loom_op_t* target_op) {
  loom_channel_materialization_clone_state_t* state = user_data;
  const loom_channel_plan_t* plan = state->source;
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
        plan->value_domain, loom_op_const_results(source_op)[i]);
    // Callable signature results are outside the execution region's domain.
    if (ordinal != LOOM_VALUE_ORDINAL_INVALID) {
      state->values[ordinal] = loom_op_results(target_op)[i];
    }
  }
  if (state->action_index < plan->action_count &&
      source_op == plan->actions[state->action_index].op) {
    state->actions[state->action_index] = plan->actions[state->action_index];
    state->actions[state->action_index++].op = target_op;
  }
  if (state->return_index < plan->return_count &&
      source_op == plan->returns[state->return_index]) {
    state->returns[state->return_index++] = target_op;
  }
  return state->observer.fn ? state->observer.fn(state->observer.user_data,
                                                 source_op, target_op)
                            : iree_ok_status();
}

static void loom_channel_materialization_clone_finish(
    void* user_data, const loom_op_t* source_op, loom_op_t* target_op) {
  loom_channel_materialization_clone_state_t* state = user_data;
  if (state->observer.finish_fn) {
    state->observer.finish_fn(state->observer.user_data, source_op, target_op);
  }
}

iree_status_t loom_channel_materialization_clone(
    loom_rewriter_t* rewriter, loom_func_like_t source_function,
    const loom_channel_plan_t* source_plan,
    const loom_cfg_graph_t* source_graph, loom_symbol_ref_t target_symbol,
    loom_ir_clone_observer_t observer,
    loom_channel_materialization_instance_t* out_instance) {
  *out_instance = (loom_channel_materialization_instance_t){0};
  const loom_local_value_domain_t* source_domain = source_plan->value_domain;
  loom_channel_materialization_clone_state_t state = {.source = source_plan,
                                                      .observer = observer};
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, source_domain->value_count,
                                sizeof(*state.values), (void**)&state.values));
  if (source_domain->value_count) {
    memcpy(state.values, source_domain->value_ids,
           source_domain->value_count * sizeof(*state.values));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, source_plan->action_count, sizeof(*state.actions),
      (void**)&state.actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, source_plan->return_count, sizeof(*state.returns),
      (void**)&state.returns));
  loom_ir_remap_t remap;
  const loom_callable_clone_options_t options = {
      .observer = {.fn = loom_channel_materialization_clone_op,
                   .finish_fn = observer.finish_fn
                                    ? loom_channel_materialization_clone_finish
                                    : NULL,
                   .user_data = &state},
      .retained_remap = &remap,
  };
  loom_func_like_t function = {0};
  IREE_RETURN_IF_ERROR(loom_callable_clone_definition(
      &rewriter->builder, source_function, target_symbol, &options,
      rewriter->arena, &function));
  // A change to clone visitation must not silently leave selected source sites
  // pointing at the shared definition that this instance must not rewrite.
  IREE_ASSERT_EQ(state.action_index, source_plan->action_count);
  IREE_ASSERT_EQ(state.return_index, source_plan->return_count);
  const loom_region_t* source_region = source_domain->region;
  const loom_region_t* target_region = loom_func_like_body(function);
  for (uint16_t b = 0; b < source_region->block_count; ++b) {
    const loom_block_t* source_block = source_region->blocks[b];
    const loom_block_t* target_block = target_region->blocks[b];
    for (uint16_t i = 0; i < source_block->arg_count; ++i) {
      const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
          source_domain, source_block->arg_ids[i]);
      state.values[ordinal] = target_block->arg_ids[i];
    }
  }
  *out_instance = (loom_channel_materialization_instance_t){
      .function = function,
      .value_domain =
          {
              .module = rewriter->module,
              .region = target_region,
              .arena = rewriter->arena,
              .value_ids = state.values,
              .value_count = source_domain->value_count,
              .definition_count = source_domain->definition_count,
              .value_capacity = source_domain->value_count,
              .operation_traits = source_domain->operation_traits,
          },
      .plan = *source_plan,
      .graph = *source_graph,
      .remap = remap,
  };
  out_instance->plan.value_domain = &out_instance->value_domain;
  out_instance->plan.actions = state.actions;
  out_instance->plan.returns = state.returns;
  loom_cfg_graph_t* graph = &out_instance->graph;
  graph->region = target_region;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, graph->block_count, sizeof(*graph->blocks),
      (void**)&graph->blocks));
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    graph->blocks[i] = source_graph->blocks[i];
    graph->blocks[i].block = target_region->blocks[i];
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, graph->edge_count,
                                sizeof(*graph->edges), (void**)&graph->edges));
  for (iree_host_size_t i = 0; i < graph->edge_count; ++i) {
    graph->edges[i] = source_graph->edges[i];
    loom_cfg_edge_info_t* edge = &graph->edges[i];
    edge->terminator = target_region->blocks[edge->source_block_index]->last_op;
    if (edge->selector_value_id != LOOM_VALUE_ID_INVALID) {
      edge->selector_value_id = state.values[loom_local_value_domain_ordinal(
          source_domain, edge->selector_value_id)];
    }
  }
  return iree_ok_status();
}

// One source block's retained exit and private protocol state.
typedef struct loom_channel_materialization_block_t {
  // Original source block, stable while forwarding blocks are appended.
  loom_block_t* block;
  // Original control exit; action emission leaves the CFG intact.
  loom_op_t* terminator;
  // Mutable state after the most recently materialized action in this block.
  loom_value_id_t* state;
  // Retained action interval in the source plan's block order.
  struct {
    // First action ordinal in this block.
    iree_host_size_t begin;
    // Exclusive end action ordinal.
    iree_host_size_t end;
  } actions;
  // Incoming edges supply appended state instead of inheriting a predecessor.
  bool has_state_arguments;
} loom_channel_materialization_block_t;

static bool loom_channel_materialization_erases_value(
    const loom_channel_plan_t* plan,
    const loom_channel_materialization_options_t* options,
    loom_value_id_t value) {
  if (!options->erased_values) {
    return false;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(plan->value_domain, value);
  return ordinal != LOOM_VALUE_ORDINAL_INVALID && ordinal < plan->value_count &&
         options->erased_values[ordinal];
}

static iree_status_t loom_channel_materialization_forward_state(
    loom_rewriter_t* rewriter, loom_region_t* region,
    const loom_channel_plan_t* plan, const loom_cfg_graph_t* graph,
    const loom_channel_materialization_options_t* options,
    const loom_channel_materialization_block_t* blocks, uint16_t block_index) {
  const loom_channel_materialization_block_t* block = &blocks[block_index];
  const iree_host_size_t state_count = options->state_count;
  loom_op_t* terminator = block->terminator;
  if (loom_cfg_br_isa(terminator) || loom_func_return_isa(terminator)) {
    const bool is_branch = loom_cfg_br_isa(terminator);
    iree_host_size_t appended_count = 0;
    if (is_branch) {
      const iree_host_size_t destination =
          loom_cfg_graph_block_index(graph, loom_cfg_br_dest(terminator));
      if (blocks[destination].has_state_arguments) {
        appended_count = state_count;
      }
    }
    loom_value_id_t* arguments = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, terminator->operand_count + appended_count,
        sizeof(*arguments), (void**)&arguments));
    iree_host_size_t argument_count = 0;
    for (uint16_t i = 0; i < terminator->operand_count; ++i) {
      const loom_value_id_t value = loom_op_operands(terminator)[i];
      if (!loom_channel_materialization_erases_value(plan, options, value)) {
        arguments[argument_count++] = value;
      }
    }
    if (appended_count) {
      memcpy(arguments + argument_count, block->state,
             appended_count * sizeof(*arguments));
    }
    if (!appended_count && argument_count == terminator->operand_count) {
      return iree_ok_status();
    }
    loom_builder_set_before(&rewriter->builder, terminator);
    loom_op_t* branch = NULL;
    if (is_branch) {
      IREE_RETURN_IF_ERROR(loom_cfg_br_build(
          &rewriter->builder, loom_cfg_br_dest(terminator), arguments,
          argument_count + appended_count, terminator->location, &branch));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_func_return_build(&rewriter->builder, arguments, argument_count,
                                 terminator->location, &branch));
    }
  } else {
    if (!state_count || !terminator->successor_count) {
      return iree_ok_status();
    }
    // Flat source execution uses cfg.br and cfg.cond_br. The latter carries no
    // argument payload, so edges entering joins forward through a plain branch.
    IREE_ASSERT(loom_cfg_cond_br_isa(terminator));
    loom_block_t* edges[2];
    bool changed = false;
    for (uint8_t i = 0; i < 2; ++i) {
      loom_block_t* destination = loom_op_successors(terminator)[i];
      edges[i] = destination;
      if (!blocks[loom_cfg_graph_block_index(graph, destination)]
               .has_state_arguments) {
        continue;
      }
      changed = true;
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(rewriter->module, region, &edges[i]));
      loom_builder_set_block(&rewriter->builder, edges[i]);
      rewriter->builder.ip.parent_op = terminator->parent_op;
      loom_op_t* branch = NULL;
      IREE_RETURN_IF_ERROR(loom_cfg_br_build(&rewriter->builder, destination,
                                             block->state, state_count,
                                             terminator->location, &branch));
    }
    if (!changed) {
      return iree_ok_status();
    }
    loom_builder_set_before(&rewriter->builder, terminator);
    loom_op_t* branch = NULL;
    IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(
        &rewriter->builder, loom_cfg_cond_br_condition(terminator), edges[0],
        edges[1], terminator->location, &branch));
  }
  return loom_rewriter_erase(rewriter, terminator);
}

iree_status_t loom_channel_materialize(
    loom_rewriter_t* rewriter, const loom_channel_plan_t* plan,
    const loom_cfg_graph_t* graph,
    const loom_channel_materialization_options_t* options) {
  loom_module_t* module = rewriter->module;
  loom_region_t* region = (loom_region_t*)plan->value_domain->region;
  const uint16_t block_count = region->block_count;
  const iree_host_size_t state_count = options->state_count;
  loom_channel_materialization_block_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, block_count, sizeof(*blocks), (void**)&blocks));
  uint16_t* order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, block_count, sizeof(*order), (void**)&order));
  memcpy(order, graph->reverse_postorder.values,
         graph->reverse_postorder.count * sizeof(*order));
  iree_host_size_t order_count = graph->reverse_postorder.count;
  loom_value_id_t* states = NULL;
  if (state_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, block_count, state_count * sizeof(*states),
        (void**)&states));
    memcpy(states, options->initial_state, state_count * sizeof(*states));
  }
  iree_host_size_t action_index = 0;
  for (uint16_t b = 0; b < block_count; ++b) {
    loom_block_t* block = region->blocks[b];
    const loom_cfg_block_info_t* info = &graph->blocks[b];
    if (!info->reachable) {
      order[order_count++] = b;
    }
    blocks[b] = (loom_channel_materialization_block_t){
        .block = block,
        .terminator = block->last_op,
        .state = state_count ? states + b * state_count : NULL,
        .actions.begin = action_index,
        .has_state_arguments =
            b != 0 && state_count != 0 &&
            (!info->reachable || info->predecessor_count != 1),
    };
    while (action_index < plan->action_count &&
           plan->actions[action_index].op->parent_block == block) {
      ++action_index;
    }
    blocks[b].actions.end = action_index;
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      const loom_value_id_t value = block->arg_ids[i];
      const loom_type_t carrier =
          options->carrier_types[loom_local_value_domain_ordinal(
              plan->value_domain, value)];
      if (loom_type_kind(carrier) != LOOM_TYPE_NONE) {
        IREE_RETURN_IF_ERROR(
            loom_rewriter_set_value_type(rewriter, value, carrier));
      }
    }
    if (!blocks[b].has_state_arguments) {
      continue;
    }
    for (iree_host_size_t i = 0; i < state_count; ++i) {
      const loom_type_t type =
          loom_module_value_type(module, options->initial_state[i]);
      loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_define_value(module, type, &argument));
      IREE_RETURN_IF_ERROR(loom_block_add_arg(module, block, argument));
      blocks[b].state[i] = argument;
    }
  }
  IREE_ASSERT_EQ(action_index, plan->action_count);
  uint16_t maximum_results = 0;
  for (iree_host_size_t i = 0; i < plan->action_count; ++i) {
    maximum_results =
        iree_max(maximum_results, plan->actions[i].op->result_count);
  }
  loom_value_id_t* results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, maximum_results, sizeof(*results), (void**)&results));
  loom_op_t** consumed_ops = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, plan->action_count,
                                sizeof(*consumed_ops), (void**)&consumed_ops));
  for (iree_host_size_t position = 0; position < order_count; ++position) {
    const uint16_t b = order[position];
    if (state_count && b != 0 && !blocks[b].has_state_arguments) {
      // A reachable single predecessor dominates this block and therefore
      // precedes it in the retained reverse postorder, including inside loops.
      const uint16_t predecessor =
          graph->predecessor_indices[graph->blocks[b].predecessor_start];
      memcpy(blocks[b].state, blocks[predecessor].state,
             state_count * sizeof(*states));
    }
    for (iree_host_size_t i = blocks[b].actions.begin;
         i < blocks[b].actions.end; ++i) {
      const loom_channel_plan_action_t* action = &plan->actions[i];
      loom_builder_set_before(&rewriter->builder, action->op);
      const loom_value_id_t checkpoint =
          loom_rewriter_value_checkpoint(rewriter);
      for (uint16_t result_index = 0; result_index < action->op->result_count;
           ++result_index) {
        results[result_index] = LOOM_VALUE_ID_INVALID;
      }
      IREE_RETURN_IF_ERROR(options->emit.fn(options->emit.user_data, rewriter,
                                            action, blocks[b].state,
                                            state_count, results));
      IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
          rewriter, action->op, results, action->op->result_count, checkpoint));
      for (uint16_t result_index = 0; result_index < action->op->result_count;
           ++result_index) {
        const loom_value_id_t value = loom_op_results(action->op)[result_index];
        if (!loom_channel_materialization_erases_value(plan, options, value)) {
          IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
              rewriter, value, results[result_index]));
        }
      }
      consumed_ops[i] = action->op;
    }
    if (options->emit.exit && loom_func_return_isa(blocks[b].terminator)) {
      loom_builder_set_before(&rewriter->builder, blocks[b].terminator);
      IREE_RETURN_IF_ERROR(options->emit.exit(options->emit.user_data, rewriter,
                                              blocks[b].terminator,
                                              blocks[b].state, state_count));
    }
    IREE_RETURN_IF_ERROR(loom_channel_materialization_forward_state(
        rewriter, region, plan, graph, options, blocks, b));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase_closed_set(rewriter, consumed_ops,
                                                      plan->action_count));
  for (uint16_t b = 1; b < block_count; ++b) {
    loom_block_t* block = blocks[b].block;
    for (uint16_t i = block->arg_count; i > 0; --i) {
      if (loom_channel_materialization_erases_value(plan, options,
                                                    block->arg_ids[i - 1])) {
        IREE_RETURN_IF_ERROR(loom_block_remove_arg(module, block, i - 1));
      }
    }
  }
  return iree_ok_status();
}
