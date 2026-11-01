// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_loop_plan.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/codegen/low/lower/control_carriers.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/control_loop.h"
#include "loom/target/arch/amdgpu/lower/control_mask.h"
#include "loom/target/arch/amdgpu/lower/control_plan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static int loom_amdgpu_loop_plan_table_key;
static int loom_amdgpu_loop_exit_region_table_key;

typedef struct loom_amdgpu_loop_exit_region_match_t {
  uint16_t loop_index;
  uint16_t continuation_index;
  uint16_t exit_entry_index;
  uint8_t clean_successor_index;
  const loom_cfg_region_t* clean_region;
  loom_cfg_edge_index_span_t exit_local_edges;
} loom_amdgpu_loop_exit_region_match_t;

static bool loom_amdgpu_try_find_loop_exit(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           uint16_t* out_loop_index,
                                           uint8_t* out_exit_successor_mask) {
  *out_loop_index = LOOM_CFG_LOOP_NEST_NONE;
  *out_exit_successor_mask = 0;
  const loom_cfg_loop_nest_t* loops =
      &loom_low_lower_context_cfg(context)->loops;
  const loom_cfg_graph_t* graph = loops->graph;
  const uint16_t block_index =
      (uint16_t)loom_cfg_graph_block_index(graph, source_op->parent_block);
  const uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return false;
  }
  const loom_cfg_edge_index_span_t outgoing =
      loom_cfg_graph_successor_edges(graph, block_index);
  const loom_cfg_edge_info_t* true_edge = &graph->edges[outgoing.values[0]];
  const loom_cfg_edge_info_t* false_edge = &graph->edges[outgoing.values[1]];
  const bool true_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, true_edge->target_block_index);
  const bool false_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, false_edge->target_block_index);
  if (true_inside && false_inside) {
    return false;
  }
  *out_loop_index = loop_index;
  *out_exit_successor_mask = (true_inside ? 0 : 1) | (false_inside ? 0 : 2);
  return true;
}

static iree_status_t loom_amdgpu_allocate_loop_plan(
    loom_low_lower_context_t* context, uint32_t exit_capacity,
    loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const iree_host_size_t plan_length =
      sizeof(**out_plan) + exit_capacity * sizeof(loom_amdgpu_loop_exit_plan_t);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(context, plan_length,
                                                         (void**)out_plan));
  **out_plan = (loom_amdgpu_loop_plan_t){
      .saved_exec = LOOM_VALUE_ID_INVALID,
      .continuation_index = LOOM_CFG_LOOP_CONTINUATION_NONE,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_or_allocate_loop_plan(
    loom_low_lower_context_t* context, const loom_cfg_loop_nest_t* loops,
    uint16_t loop_index, loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->exits.count == 1) {
    return loom_amdgpu_allocate_loop_plan(context, loop->direct_exit_count,
                                          out_plan);
  }

  loom_amdgpu_loop_plan_t** plans = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_loop_plan_table_key,
      loops->loop_count * sizeof(*plans), (void**)&plans));
  if (plans[loop_index] == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_allocate_loop_plan(
        context, loop->direct_exit_count, &plans[loop_index]));
  }
  *out_plan = plans[loop_index];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_loop_exit_region_table(
    loom_low_lower_context_t* context,
    loom_amdgpu_loop_exit_region_plan_t*** out_table) {
  const iree_host_size_t block_count =
      loom_low_lower_context_cfg(context)->graph.block_count;
  return loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_loop_exit_region_table_key,
      block_count * sizeof(**out_table), (void**)out_table);
}

static bool loom_amdgpu_try_loop_exit_region_orientation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t loop_index, uint8_t clean_successor_index,
    loom_amdgpu_loop_exit_region_match_t* out_match) {
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_block_t* clean_entry =
      loom_op_const_successors(source_op)[clean_successor_index];
  const loom_block_t* exit_entry =
      loom_op_const_successors(source_op)[clean_successor_index ^ 1u];
  const uint16_t guard_index = source_op->parent_block->region_index;
  const uint16_t clean_entry_index = clean_entry->region_index;
  const uint16_t exit_entry_index = exit_entry->region_index;
  if (facts->dominance.entry_predecessors[clean_entry_index] != guard_index ||
      facts->dominance.entry_predecessors[exit_entry_index] != guard_index) {
    return false;
  }

  const loom_cfg_region_t* clean_region =
      &facts->regions.blocks[clean_entry_index];
  if (clean_region->continuation_index == LOOM_CFG_REGION_CONTINUATION_NONE ||
      !clean_region->exit_count ||
      clean_region->exit_count != clean_region->exit_edges.count ||
      !loom_cfg_loop_nest_contains(loops, loop_index,
                                   clean_region->continuation_index)) {
    return false;
  }
  const uint16_t continuation_index = clean_region->continuation_index;
  if (continuation_index == guard_index) {
    return false;
  }

  const loom_op_t* exit_terminator = loom_block_const_last_op(exit_entry);
  if (!loom_cfg_cond_br_isa(exit_terminator)) {
    return false;
  }
  uint16_t exit_loop_index = LOOM_CFG_LOOP_NEST_NONE;
  uint8_t exit_successor_mask = 0;
  if (!loom_amdgpu_try_find_loop_exit(context, exit_terminator,
                                      &exit_loop_index, &exit_successor_mask) ||
      exit_loop_index != loop_index ||
      (exit_successor_mask != 1 && exit_successor_mask != 2)) {
    return false;
  }

  const loom_cfg_region_t* exit_region =
      &facts->regions.blocks[exit_entry_index];
  const loom_cfg_edge_index_span_t exit_local_edges =
      loom_cfg_regions_edges_to(graph, &facts->dominance, &facts->regions,
                                exit_entry_index, continuation_index);
  if (!exit_local_edges.count ||
      exit_region->exit_count != exit_local_edges.count + 1) {
    return false;
  }

  *out_match = (loom_amdgpu_loop_exit_region_match_t){
      .loop_index = loop_index,
      .continuation_index = continuation_index,
      .exit_entry_index = exit_entry_index,
      .clean_successor_index = clean_successor_index,
      .clean_region = clean_region,
      .exit_local_edges = exit_local_edges,
  };
  return true;
}

static iree_status_t loom_amdgpu_try_prepare_loop_exit_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_prepared) {
  *out_prepared = false;
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_graph_t* graph = &facts->graph;
  const uint16_t guard_index = source_op->parent_block->region_index;
  const uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, guard_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < 2; ++i) {
    if (!loom_cfg_loop_nest_contains(
            loops, loop_index,
            loom_op_const_successors(source_op)[i]->region_index)) {
      return iree_ok_status();
    }
  }

  loom_amdgpu_loop_exit_region_match_t match = {0};
  if (!loom_amdgpu_try_loop_exit_region_orientation(
          context, source_op, loop_index, /*clean_successor_index=*/0,
          &match) &&
      !loom_amdgpu_try_loop_exit_region_orientation(
          context, source_op, loop_index, /*clean_successor_index=*/1,
          &match)) {
    return iree_ok_status();
  }

  const loom_low_lower_block_ref_t continuation =
      loom_low_lower_control_source_block(
          graph->blocks[match.continuation_index].block);
  const uint16_t merge_arg_count =
      loom_low_lower_control_argument_count(context, continuation);
  const uint16_t source_merge_arg_count =
      loom_low_lower_control_carrier_argument_offset(context, continuation);
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  const loom_amdgpu_mask_merge_constants_t* merge_constants = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_masked_merge(
      context, source_op, graph->blocks[match.continuation_index].block,
      continuation, (loom_value_slice_t){0}, (loom_value_slice_t){0},
      &merge_constants));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }

  loom_amdgpu_loop_plan_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_or_allocate_loop_plan(
      context, loops, match.loop_index, &loop));
  loom_amdgpu_loop_exit_region_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_loop_exit_region_plan_t){
      .loop = loop,
      .source_op = source_op,
      .clean_successor_index = match.clean_successor_index,
      .continuation_index = match.continuation_index,
      .continuation = continuation,
      .merge_arg_count = merge_arg_count,
      .source_merge_arg_count = source_merge_arg_count,
      .merge_constants = merge_constants,
      .saved_exec = LOOM_VALUE_ID_INVALID,
      .source_condition = LOOM_VALUE_ID_INVALID,
  };

  loom_type_t exec_mask_type = loom_low_lower_value_binding_type(
      context, loom_cfg_cond_br_condition(source_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block_with_appended_arguments(
      context, /*before=*/0, continuation, &exec_mask_type, 1,
      &plan->arm_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->no_clean_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->clean_only_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, continuation, &plan->merge_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->capture_forward_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->direct_forward_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->no_repeat_block));

  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, source_op, match.clean_successor_index ^ 1u,
      plan->no_clean_block, &plan->arm_body_block));

  const iree_host_size_t clean_exit_count =
      match.clean_region->exit_edges.count;
  IREE_ASSERT_LE(clean_exit_count, UINT16_MAX);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, clean_exit_count * sizeof(*plan->clean_adapters),
      (void**)&plan->clean_adapters));
  for (iree_host_size_t i = 0; i < clean_exit_count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[match.clean_region->exit_edges.values[i]];
    loom_low_lower_block_ref_t adapter = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, continuation, &adapter));
    loom_low_lower_block_ref_t previous_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, edge->terminator, (uint8_t)edge->successor_index, adapter,
        &previous_dest));
    if (previous_dest != continuation) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
    plan->clean_adapters[plan->clean_adapter_count++].block = adapter;
  }

  for (iree_host_size_t i = 0; i < match.exit_local_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[match.exit_local_edges.values[i]];
    loom_low_lower_block_ref_t previous_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, edge->terminator, (uint8_t)edge->successor_index,
        plan->merge_block, &previous_dest));
    if (previous_dest != continuation) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  loom_amdgpu_loop_exit_region_plan_t** table = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_loop_exit_region_table(context, &table));
  IREE_ASSERT(table[match.exit_entry_index] == NULL);
  table[match.exit_entry_index] = plan;
  loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_LOOP_EXIT_REGION, plan));
  *out_prepared = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_validate_divergent_loop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index,
    loom_amdgpu_loop_plan_t* plan) {
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->entries.count != 1) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_entry"));
  }
  uint32_t continuation = loop->continuation_index;
  if (continuation == LOOM_CFG_LOOP_CONTINUATION_NONE &&
      facts->control_structure.postdominance.available) {
    continuation =
        facts->control_structure.postdominance.nodes[loop->header_index]
            .immediate_postdominator;
  }
  if (continuation >= facts->graph.block_count ||
      loom_cfg_loop_nest_contains(loops, loop_index, (uint16_t)continuation)) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_continuation"));
  }
  if (loop->direct_exit_count != loop->exits.count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_direct_exits"));
  }
  plan->continuation_index = (uint16_t)continuation;
  return iree_ok_status();
}

static bool loom_amdgpu_loop_guard_is_active_uniform(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index) {
  const loom_cfg_natural_loop_t* loop = &facts->loops.loops[loop_index];
  if (source_op->parent_block->region_index != loop->header_index) {
    return false;
  }
  const loom_value_fact_induction_t* induction = &facts->inductions[loop_index];
  if (induction->value == LOOM_VALUE_ID_INVALID ||
      induction->step.value == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_fact_table_t* table =
      loom_low_lower_context_fact_table(context);
  return loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(
                 table, induction->initial_value)) &&
         loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(table,
                                                      induction->bound)) &&
         loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(table, induction->step));
}

static iree_status_t loom_amdgpu_validate_loop_exit_regions(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, loom_amdgpu_loop_plan_t* plan) {
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        &facts->graph, exit->source_op->parent_block->region_index);
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const uint16_t target =
          facts->graph.edges[outgoing.values[successor_index]]
              .target_block_index;
      if (target == plan->continuation_index) {
        continue;
      }
      if (facts->regions.blocks[target].continuation_index !=
          plan->continuation_index) {
        plan->validation = LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
        return loom_low_lower_emit_branch_constraint(
            context, source_op, IREE_SV("divergent_loop_single_continuation"));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_plan_loop_exit_adapter(
    loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* facts, loom_cfg_edge_index_t edge_index,
    loom_low_lower_block_ref_t continuation,
    loom_amdgpu_loop_exit_plan_t* exit) {
  const loom_cfg_edge_info_t* edge = &facts->graph.edges[edge_index];
  loom_low_lower_block_ref_t adapter = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, continuation, &adapter));
  loom_low_lower_block_ref_t previous_dest = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, edge->terminator, (uint8_t)edge->successor_index, adapter,
      &previous_dest));
  if (previous_dest != continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, exit->source_op,
        IREE_SV("divergent_loop_single_continuation"));
  }
  exit->adapters[exit->adapter_count++].block = adapter;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_finalize_loop_plan(
    loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index,
    loom_amdgpu_loop_plan_t* plan, iree_arena_allocator_t* analysis_arena) {
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  const loom_cfg_graph_t* graph = &facts->graph;
  IREE_RETURN_IF_ERROR(loom_amdgpu_validate_loop_exit_regions(
      context, plan->exit_plans[plan->exit_plan_count - 1].source_op, facts,
      plan));
  if (plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_REJECTED) {
    return iree_ok_status();
  }
  const loom_cfg_edge_info_t* entry_edge =
      &graph->edges[loop->entries.unique_index];
  const loom_block_t* header_block = graph->blocks[loop->header_index].block;
  const loom_low_lower_block_ref_t low_continuation =
      loom_low_lower_control_source_block(
          graph->blocks[plan->continuation_index].block);

  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    if (plan->exit_plans[i].is_divergent) {
      ++plan->divergent_exit_count;
    }
  }
  IREE_ASSERT_GT(plan->divergent_exit_count, 0);
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    plan->exec_mask_type = loom_low_lower_value_binding_type(
        context, loom_cfg_cond_br_condition(exit->source_op));
    break;
  }
  IREE_ASSERT(
      loom_amdgpu_low_type_is_native_i1_mask(context, plan->exec_mask_type));
  plan->capture_value_count =
      loom_low_lower_control_argument_count(context, low_continuation);
  plan->capture_exits = plan->capture_value_count != 0;
  for (uint32_t i = 0; i < plan->exit_plan_count && !plan->capture_exits; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    const loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        graph, exit->source_op->parent_block->region_index);
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) != 0 &&
          graph->edges[outgoing.values[successor_index]].target_block_index !=
              plan->continuation_index) {
        plan->capture_exits = true;
        break;
      }
    }
  }

  const loom_low_lower_block_ref_t low_header =
      loom_low_lower_control_source_block(header_block);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(context, 0, low_header,
                                                        &plan->entry_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, entry_edge->terminator, (uint8_t)entry_edge->successor_index,
      plan->entry_block, &plan->header_dest));

  if (!plan->capture_exits) {
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &plan->restore_block));
    for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
      const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
      for (uint8_t successor_index = 0; successor_index < 2;
           ++successor_index) {
        if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
          continue;
        }
        loom_low_lower_block_ref_t previous_dest = 0;
        IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
            context, exit->source_op, successor_index, plan->restore_block,
            &previous_dest));
        if (plan->restore_dest == 0) {
          plan->restore_dest = previous_dest;
        } else {
          IREE_ASSERT_EQ(plan->restore_dest, previous_dest);
        }
      }
    }
    IREE_ASSERT_NE(plan->restore_dest, 0);
    return iree_ok_status();
  }

  IREE_ASSERT_LE(plan->exit_plan_count, UINT16_MAX);
  uint16_t* capture_sources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, plan->exit_plan_count, sizeof(*capture_sources),
      (void**)&capture_sources));
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    capture_sources[i] =
        plan->exit_plans[i].source_op->parent_block->region_index;
  }
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    IREE_RETURN_IF_ERROR(loom_low_lower_control_retain_resume_closure(
        context, repeat_block, capture_sources,
        (uint16_t)plan->exit_plan_count));
  }
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_region_plan_t* region = plan->exit_plans[i].region;
    if (region != NULL) {
      region->merge_arg_count =
          loom_low_lower_control_argument_count(context, region->continuation);
    }
  }
  plan->capture_value_count =
      loom_low_lower_control_argument_count(context, low_continuation);

  uint32_t resume_arg_count = 0;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    resume_arg_count +=
        exit->region != NULL
            ? exit->region->merge_arg_count
            : loom_low_lower_control_source_carriers(context, repeat_block)
                  .count;
  }
  if (resume_arg_count > UINT16_MAX) {
    return loom_low_lower_emit_branch_constraint(
        context, plan->exit_plans[0].source_op,
        IREE_SV("divergent_loop_resume_tuple_u16"));
  }
  plan->resume_arg_count = (uint16_t)resume_arg_count;

  uint16_t private_type_count = 1;
  if (plan->divergent_exit_count > 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_type(context, &plan->selector_type));
    ++private_type_count;
  }
  if ((uint32_t)private_type_count + plan->resume_arg_count > UINT16_MAX) {
    return loom_low_lower_emit_branch_constraint(
        context, plan->exit_plans[0].source_op,
        IREE_SV("divergent_loop_resume_tuple_u16"));
  }
  private_type_count += plan->resume_arg_count;
  loom_type_t* private_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, private_type_count, sizeof(*private_types),
      (void**)&private_types));
  uint16_t private_index = 0;
  private_types[private_index++] = plan->exec_mask_type;
  if (plan->divergent_exit_count > 1) {
    private_types[private_index++] = plan->selector_type;
  }
  const uint16_t resume_private_offset = private_index;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    const loom_low_lower_block_ref_t resume_signature =
        loom_low_lower_control_source_block(repeat_block);
    exit->resume_arg_count =
        exit->region != NULL
            ? exit->region->merge_arg_count
            : loom_low_lower_control_source_carriers(context, repeat_block)
                  .count;
    if (exit->region != NULL) {
      for (uint16_t j = 0; j < exit->resume_arg_count; ++j) {
        private_types[private_index++] =
            loom_low_lower_control_argument_type(context, resume_signature, j);
      }
    } else {
      const loom_low_lower_control_carrier_slice_t carriers =
          loom_low_lower_control_source_carriers(context, repeat_block);
      for (uint16_t j = 0; j < carriers.count; ++j) {
        private_types[private_index++] =
            loom_low_lower_control_carrier_type(context, carriers.values[j]);
      }
    }
  }
  IREE_ASSERT_EQ(private_index, private_type_count);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block_with_appended_arguments(
      context, /*before=*/0, low_continuation, private_types,
      private_type_count, &plan->capture_block));
  plan->resume_arg_offset =
      (uint16_t)(plan->capture_value_count + resume_private_offset);
  private_index = resume_private_offset;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    exit->resume_arg_offset =
        (uint16_t)(plan->capture_value_count + private_index);
    private_index += exit->resume_arg_count;
  }
  IREE_ASSERT_EQ(private_index, private_type_count);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->restore_block));
  plan->restore_dest = low_continuation;

  iree_bitmap_t visited_targets = {.bit_count = graph->block_count};
  const iree_host_size_t visited_word_count =
      iree_bitmap_calculate_words(visited_targets.bit_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, visited_word_count, sizeof(*visited_targets.words),
      (void**)&visited_targets.words));
  iree_bitmap_reset_all(visited_targets);
  uint16_t next_selector = 0;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (exit->region != NULL) {
      const loom_type_t local_types[] = {
          plan->exec_mask_type,
          plan->exec_mask_type,
      };
      IREE_RETURN_IF_ERROR(
          loom_low_lower_control_add_block_with_appended_arguments(
              context, /*before=*/0, low_continuation, local_types,
              IREE_ARRAYSIZE(local_types), &exit->local_capture_block));
    }
    loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        graph, exit->source_op->parent_block->region_index);
    iree_host_size_t adapter_capacity = 0;
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const loom_cfg_edge_index_t edge_index = outgoing.values[successor_index];
      const uint16_t target = graph->edges[edge_index].target_block_index;
      adapter_capacity += target == plan->continuation_index
                              ? 1
                              : facts->regions.blocks[target].exit_edges.count;
    }
    IREE_ASSERT_LE(adapter_capacity, UINT16_MAX);
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, adapter_capacity * sizeof(*exit->adapters),
        (void**)&exit->adapters));

    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const loom_cfg_edge_index_t edge_index = outgoing.values[successor_index];
      const uint16_t target = graph->edges[edge_index].target_block_index;
      if (target == plan->continuation_index) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_plan_loop_exit_adapter(
            context, facts, edge_index, low_continuation, exit));
        continue;
      }
      if (iree_bitmap_test(visited_targets, target)) {
        return loom_low_lower_emit_branch_constraint(
            context, exit->source_op,
            IREE_SV("divergent_loop_exit_region_ownership"));
      }
      iree_bitmap_set(visited_targets, target);
      loom_cfg_edge_index_span_t retirement_edges =
          facts->regions.blocks[target].exit_edges;
      for (iree_host_size_t j = 0; j < retirement_edges.count; ++j) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_plan_loop_exit_adapter(
            context, facts, retirement_edges.values[j], low_continuation,
            exit));
      }
    }
    IREE_ASSERT_EQ(exit->adapter_count, adapter_capacity);

    if (!exit->is_divergent) {
      continue;
    }
    if (exit->exit_successor_mask == 3) {
      return loom_low_lower_emit_branch_constraint(
          context, exit->source_op, IREE_SV("divergent_loop_repeat_successor"));
    }
    exit->selector = next_selector++;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &exit->dispatch_block));
    if (exit->resume_arg_count != 0) {
      IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
          context, /*before=*/0, /*signature=*/0, &exit->resume_block));
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &exit->no_exit_block));
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_argless_successor(
        context, exit->source_op, repeat_successor, exit->no_exit_block,
        &exit->source_repeat_dest));
    exit->repeat_dest =
        exit->region != NULL
            ? loom_low_lower_control_source_block(
                  graph->blocks[exit->region->continuation_index].block)
            : exit->source_repeat_dest;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_prepare_loop_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, iree_arena_allocator_t* analysis_arena,
    bool* out_handled, bool* out_active_uniform) {
  *out_handled = false;
  *out_active_uniform = false;
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  uint16_t loop_index = LOOM_CFG_LOOP_NEST_NONE;
  uint8_t exit_successor_mask = 0;
  if (!loom_amdgpu_try_find_loop_exit(context, source_op, &loop_index,
                                      &exit_successor_mask)) {
    return iree_ok_status();
  }
  *out_handled = true;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (is_divergent && loom_amdgpu_loop_guard_is_active_uniform(
                          context, source_op, facts, loop_index)) {
    is_divergent = false;
    *out_active_uniform = true;
  }
  if (loop->exits.count == 1 && !is_divergent) {
    return iree_ok_status();
  }

  loom_amdgpu_loop_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_or_allocate_loop_plan(context, loops, loop_index, &plan));
  if (is_divergent &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED) {
    const uint32_t previous_error_count =
        loom_low_lower_context_error_count(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_validate_divergent_loop(
        context, source_op, facts, loop_index, plan));
    plan->validation =
        loom_low_lower_context_error_count(context) == previous_error_count
            ? LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED
            : LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
  }
  if (plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_REJECTED) {
    return iree_ok_status();
  }

  IREE_ASSERT_LT(plan->exit_plan_count, loop->direct_exit_count);
  loom_amdgpu_loop_exit_plan_t* exit_plan =
      &plan->exit_plans[plan->exit_plan_count++];
  *exit_plan = (loom_amdgpu_loop_exit_plan_t){
      .loop = plan,
      .source_op = source_op,
      .exit_successor_mask = exit_successor_mask,
      .is_divergent = is_divergent,
  };
  const iree_host_size_t block_count = facts->graph.block_count;
  loom_amdgpu_loop_exit_region_plan_t* const* region_table =
      (loom_amdgpu_loop_exit_region_plan_t* const*)
          loom_low_lower_lookup_target_state(
              context, &loom_amdgpu_loop_exit_region_table_key,
              block_count * sizeof(*region_table));
  loom_amdgpu_loop_exit_region_plan_t* region =
      region_table ? region_table[source_op->parent_block->region_index] : NULL;
  if (region != NULL) {
    IREE_ASSERT_EQ(region->loop, plan);
    if (!is_divergent) {
      plan->validation = LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("divergent_loop_nested_exit_condition"));
    }
    IREE_ASSERT(region->exit == NULL);
    region->exit = exit_plan;
    exit_plan->region = region;
  }
  plan->exit_edge_count += exit_successor_mask == 3 ? 2 : 1;
  IREE_ASSERT_LE(plan->exit_edge_count, loop->direct_exit_count);
  if (is_divergent) {
    loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP,
                                 exit_plan));
  }
  if (plan->exit_edge_count == loop->exits.count &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_finalize_loop_plan(
        context, facts, loop_index, plan, analysis_arena));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_loop_try_prepare_exit_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_prepared) {
  return loom_amdgpu_try_prepare_loop_exit_region(context, source_op,
                                                  out_prepared);
}

iree_status_t loom_amdgpu_loop_try_prepare_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, iree_arena_allocator_t* analysis_arena,
    bool* out_handled, bool* out_active_uniform) {
  return loom_amdgpu_try_prepare_loop_exit(context, source_op, is_divergent,
                                           analysis_arena, out_handled,
                                           out_active_uniform);
}
