// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control.h"

#include "loom/analysis/condition_facts.h"
#include "loom/codegen/low/lower/control_carriers.h"
#include "loom/codegen/low/lower/control_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/control_loop.h"
#include "loom/target/arch/amdgpu/lower/control_mask.h"
#include "loom/target/arch/amdgpu/lower/control_plan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_masked_region_t {
  // Shared producer's exact graph edge identities for this region's exits.
  loom_cfg_edge_index_span_t exit_edges;
  // Guard false destination reached when no lane enters the masked region.
  loom_block_t* guard_false_dest;
  // Shared continuation reached by region exits.
  loom_block_t* continuation;
  // Optional transparent false path carrying fallback values to continuation.
  struct {
    // Unconditional forwarding terminator, or NULL for a nontransparent path.
    const loom_op_t* terminator;
    // True when region exits also reach this shared forwarding block.
    bool is_region_exit;
  } false_path;
} loom_amdgpu_masked_region_t;

typedef struct loom_amdgpu_branch_plan_t {
  // Low-only block that restores EXEC before leaving a narrowed region.
  loom_low_lower_block_ref_t restore_block;
  // Effective destination after EXEC has been restored.
  loom_low_lower_block_ref_t restore_dest;
  // False-arm terminator used when restore_block bypasses a pass-through block.
  const loom_op_t* restore_passthrough_terminator;
  // Low-only block that restores EXEC and merges pass-through false values.
  loom_low_lower_block_ref_t merge_restore_block;
  // Shared continuation reached after merge_restore_block.
  loom_low_lower_block_ref_t merge_restore_dest;
  // False-arm terminator providing pass-through merge fallback values.
  const loom_op_t* false_passthrough_terminator;
  // Low-only block that computes the inactive else mask for if/else diamonds.
  loom_low_lower_block_ref_t else_dispatch_block;
  // Low-only block that enters the else body when no true lanes were active.
  loom_low_lower_block_ref_t no_true_else_entry_block;
  // Original low destination for the else body.
  loom_low_lower_block_ref_t else_body_block;
  // Carrier-aware target conditional edge entering the else body.
  loom_low_lower_control_carrier_edge_plan_t else_body_edge;
  // Per-argument logical constants at a masked merge; NULL uses incoming masks.
  const loom_amdgpu_mask_merge_constants_t* merge_constants;
  // Low-only block that restores EXEC when no false lanes were active.
  loom_low_lower_block_ref_t true_only_restore_block;
  // Number of authored and realization values lane-merged by an if/else
  // diamond. Supplemental control carriers follow them unchanged.
  uint16_t if_else_merge_arg_count;
} loom_amdgpu_branch_plan_t;

typedef struct loom_amdgpu_zero_placeholder_t {
  // Low value type represented by value.
  loom_type_t type;
  // Reusable zero value for an inactive merge placeholder.
  loom_value_id_t value;
} loom_amdgpu_zero_placeholder_t;

static iree_status_t loom_amdgpu_emit_plain_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  return loom_low_lower_control_emit_cond_branch(
      context, source_op, low_condition, low_true_dest, low_false_dest);
}

static bool loom_amdgpu_condition_is_reg_class(
    loom_low_lower_context_t* context, loom_type_t low_type,
    uint16_t reg_class_id, uint32_t unit_count) {
  const bool is_class =
      loom_amdgpu_low_type_is_register_class(context, low_type, reg_class_id);
  return is_class && loom_low_register_type_unit_count(low_type) == unit_count;
}

static bool loom_amdgpu_branch_condition_is_uniform(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_value_facts_is_subgroup_uniform(
      loom_value_fact_table_lookup(loom_low_lower_context_fact_table(context),
                                   loom_cfg_cond_br_condition(source_op)));
}

static bool loom_amdgpu_single_predecessor_cfg_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_condition, bool* out_assumed_truth) {
  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  loom_cfg_edge_index_span_t incoming = loom_cfg_graph_predecessor_edges(
      graph, source_op->parent_block->region_index);
  if (incoming.count != 1) {
    return false;
  }
  const loom_cfg_edge_info_t* edge = &graph->edges[incoming.values[0]];
  if (!loom_cfg_cond_br_isa(edge->terminator)) {
    return false;
  }
  *out_condition = edge->selector_value_id;
  *out_assumed_truth = edge->successor_index == 0;
  return true;
}

static iree_status_t loom_amdgpu_cfg_cond_br_edge_implied_bool(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition, bool* out_proven) {
  *out_condition = false;
  *out_proven = false;
  if (!loom_cfg_cond_br_isa(source_op)) {
    return iree_ok_status();
  }

  loom_value_id_t edge_condition = LOOM_VALUE_ID_INVALID;
  bool edge_assumed_truth = false;
  if (!loom_amdgpu_single_predecessor_cfg_condition(
          context, source_op, &edge_condition, &edge_assumed_truth)) {
    return iree_ok_status();
  }

  const loom_value_id_t condition = loom_cfg_cond_br_condition(source_op);
  if (condition == edge_condition) {
    *out_condition = edge_assumed_truth;
    *out_proven = true;
    return iree_ok_status();
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t edge_facts = {0};
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &edge_facts);
  bool complete = false;
  loom_condition_query_t* condition_query = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_condition_query(context, &condition_query));
  IREE_RETURN_IF_ERROR(
      loom_condition_facts_query(condition_query, fact_table, edge_condition,
                                 edge_assumed_truth, &edge_facts, &complete));
  if (!complete) {
    return iree_ok_status();
  }
  if (edge_facts.integer_relation_count == 0) {
    return iree_ok_status();
  }

  return loom_condition_fact_set_proves_condition(condition_query, fact_table,
                                                  &edge_facts, condition,
                                                  out_condition, out_proven);
}

static iree_status_t loom_amdgpu_emit_sgpr_bool_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));

  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero));

  const loom_value_id_t compare_operands[] = {
      low_condition,
      low_zero,
  };
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32,
      compare_operands, IREE_ARRAYSIZE(compare_operands),
      loom_named_attr_slice_empty(), &scc_type, 1, &compare_op));
  return loom_amdgpu_emit_plain_cond_branch(
      context, source_op, loom_op_const_results(compare_op)[0], low_true_dest,
      low_false_dest);
}

static bool loom_amdgpu_try_false_passthrough_continuation(
    const loom_value_fact_cfg_region_t* facts, const loom_op_t* source_op,
    loom_block_t** out_continuation, const loom_op_t** out_false_terminator) {
  *out_continuation = NULL;
  *out_false_terminator = NULL;
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (false_dest->arg_count != 0 || false_dest->op_count != 1) {
    return false;
  }
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* continuation = loom_cfg_br_dest(false_terminator);
  if (continuation == false_dest) {
    return false;
  }
  // An enclosing bypass may also enter this forwarding block. If the true
  // region joins there too, its mask must restore before those paths meet.
  // A true region already exiting beyond the block can still bypass just the
  // guard's false edge, without claiming the shared block's outgoing edge.
  if (!loom_cfg_dominance_block_dominates(&facts->dominance,
                                          source_op->parent_block->region_index,
                                          false_dest->region_index) &&
      facts->regions.blocks[loom_cfg_cond_br_true_dest(source_op)->region_index]
              .continuation_index != continuation->region_index) {
    return false;
  }
  // A reconvergence block immediately before a loop header is the loop's
  // structural entry gateway. Bypassing it would skip any target-owned loop
  // entry interposition installed on its outgoing edge.
  if (facts->graph.blocks[false_dest->region_index].predecessor_count > 1 &&
      facts->graph.blocks[continuation->region_index].is_dfs_backedge_target) {
    return false;
  }
  *out_continuation = continuation;
  *out_false_terminator = false_terminator;
  return true;
}

static iree_status_t loom_amdgpu_analyze_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_masked_region_t* out_region) {
  *out_region = (loom_amdgpu_masked_region_t){0};
  loom_block_t* source_entry = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_destination = loom_cfg_cond_br_false_dest(source_op);
  loom_block_t* continuation = false_destination;
  const loom_op_t* false_terminator = NULL;
  loom_block_t* passthrough = NULL;
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  loom_amdgpu_try_false_passthrough_continuation(facts, source_op, &passthrough,
                                                 &false_terminator);
  if (passthrough) {
    continuation = passthrough;
  }
  if (source_entry->arg_count || false_destination->arg_count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("destination_block_arguments_absent"));
  }
  if (source_entry == continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  uint16_t entry_index = source_entry->region_index;
  if (facts->dominance.entry_predecessors[entry_index] !=
      source_op->parent_block->region_index) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_entry"));
  }
  const loom_cfg_region_t* region = &facts->regions.blocks[entry_index];
  loom_cfg_edge_index_span_t direct_exits = loom_cfg_regions_edges_to(
      &facts->graph, &facts->dominance, &facts->regions, entry_index,
      continuation->region_index);
  loom_cfg_edge_index_span_t forwarded_exits = {0};
  if (false_terminator) {
    forwarded_exits = loom_cfg_regions_edges_to(
        &facts->graph, &facts->dominance, &facts->regions, entry_index,
        false_destination->region_index);
  }
  if (!region->exit_count ||
      region->exit_count != direct_exits.count + forwarded_exits.count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_exits_by_cfg"));
  }
  if (forwarded_exits.count) {
    loom_cfg_edge_index_span_t incoming = loom_cfg_regions_edges_to(
        &facts->graph, &facts->dominance, &facts->regions, 0,
        false_destination->region_index);
    // The guard's false edge is the sole external entry we bypass. All other
    // incoming paths must belong to the true region before sharing its exit.
    if (incoming.count != forwarded_exits.count + 1) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_single_entry"));
    }
  }
  *out_region = (loom_amdgpu_masked_region_t){
      .exit_edges = direct_exits,
      .guard_false_dest = false_destination,
      .continuation = continuation,
      .false_path =
          {
              .terminator = false_terminator,
              .is_region_exit = forwarded_exits.count != 0,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_masked_region_t region = {0};
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_analyze_then_masked_region(context, source_op, &region));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  const loom_low_lower_block_ref_t guard_false_low_dest =
      loom_low_lower_control_source_block(region.guard_false_dest);
  // Keep restoration beside its continuation instead of making each loop
  // iteration detour through the function tail.
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, guard_false_low_dest, guard_false_low_dest,
      &plan->restore_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, source_op, 1, plan->restore_block, &plan->restore_dest));
  if (region.false_path.terminator != NULL) {
    plan->restore_dest = loom_low_lower_control_successor(
        context, region.false_path.terminator, 0);
    plan->restore_passthrough_terminator = region.false_path.terminator;
  }

  loom_low_lower_block_ref_t exit_restore_block = plan->restore_block;
  loom_low_lower_block_ref_t exit_restore_dest = plan->restore_dest;
  if (region.continuation != region.guard_false_dest) {
    const loom_low_lower_block_ref_t continuation_low_dest =
        loom_low_lower_control_source_block(region.continuation);
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, continuation_low_dest, continuation_low_dest,
        &exit_restore_block));
    plan->merge_restore_block = exit_restore_block;
    plan->merge_restore_dest = exit_restore_dest;
    plan->false_passthrough_terminator = region.false_path.terminator;
    IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_masked_merge(
        context, source_op, region.continuation, continuation_low_dest,
        loom_cfg_br_args(region.false_path.terminator), (loom_value_slice_t){0},
        &plan->merge_constants));
    if (loom_low_lower_context_error_count(context) != previous_error_count) {
      return iree_ok_status();
    }
  }

  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  for (iree_host_size_t i = 0; i < region.exit_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[region.exit_edges.values[i]];
    loom_low_lower_block_ref_t previous_low_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, edge->terminator, edge->successor_index, exit_restore_block,
        &previous_low_dest));
    if (previous_low_dest != exit_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }
  if (region.false_path.is_region_exit) {
    loom_low_lower_block_ref_t previous_low_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, region.false_path.terminator, 0, exit_restore_block,
        &previous_low_dest));
    if (previous_low_dest != exit_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION,
                               plan));
  return iree_ok_status();
}

static bool loom_amdgpu_try_if_else_diamond(const loom_op_t* source_op) {
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return false;
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest) {
    return false;
  }
  return true;
}

static loom_value_slice_t loom_amdgpu_single_region_exit_args(
    const loom_cfg_graph_t* graph, const loom_cfg_region_t* region,
    uint16_t expected_count) {
  if (region->exit_edges.count != 1) {
    return (loom_value_slice_t){0};
  }
  const loom_cfg_edge_info_t* edge =
      &graph->edges[region->exit_edges.values[0]];
  if (!loom_cfg_br_isa(edge->terminator)) {
    return (loom_value_slice_t){0};
  }
  loom_value_slice_t args = loom_cfg_br_args(edge->terminator);
  return args.count == expected_count ? args : (loom_value_slice_t){0};
}

static bool loom_amdgpu_try_if_else_regions(
    const loom_value_fact_cfg_region_t* facts, const loom_op_t* source_op,
    const loom_cfg_region_t** out_true_region,
    const loom_cfg_region_t** out_false_region) {
  const loom_block_t* true_entry = loom_cfg_cond_br_true_dest(source_op);
  const loom_block_t* false_entry = loom_cfg_cond_br_false_dest(source_op);
  if (true_entry->arg_count || false_entry->arg_count ||
      true_entry == false_entry) {
    return false;
  }
  uint16_t guard = source_op->parent_block->region_index;
  if (facts->dominance.entry_predecessors[true_entry->region_index] != guard ||
      facts->dominance.entry_predecessors[false_entry->region_index] != guard) {
    return false;
  }
  const loom_cfg_region_t* true_region =
      &facts->regions.blocks[true_entry->region_index];
  const loom_cfg_region_t* false_region =
      &facts->regions.blocks[false_entry->region_index];
  if (true_region->continuation_index == LOOM_CFG_REGION_CONTINUATION_NONE ||
      true_region->continuation_index != false_region->continuation_index ||
      true_region->continuation_index == guard) {
    return false;
  }
  *out_true_region = true_region;
  *out_false_region = false_region;
  return true;
}

static iree_status_t loom_amdgpu_prepare_if_else_regions(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_cfg_region_t* true_region,
    const loom_cfg_region_t* false_region) {
  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  const loom_block_t* continuation =
      graph->blocks[true_region->continuation_index].block;
  const loom_low_lower_block_ref_t merge_low_dest =
      loom_low_lower_control_source_block(continuation);

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  plan->if_else_merge_arg_count =
      loom_low_lower_control_carrier_argument_offset(context, merge_low_dest);

  if (plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, 0, 0, &plan->else_dispatch_block));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_control_add_block(context, 0, 0, &plan->restore_block));

    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, source_op, 1, plan->else_dispatch_block,
        &plan->else_body_block));
    IREE_RETURN_IF_ERROR(loom_low_lower_control_plan_carrier_conditional_edge(
        context, source_op, plan->else_body_block, &plan->else_body_edge));

    loom_low_lower_block_ref_t common_restore_dest = 0;
    for (iree_host_size_t i = 0; i < true_region->exit_edges.count; ++i) {
      loom_low_lower_block_ref_t previous_low_dest = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
          context, graph->edges[true_region->exit_edges.values[i]].terminator,
          graph->edges[true_region->exit_edges.values[i]].successor_index,
          plan->else_dispatch_block, &previous_low_dest));
      if (common_restore_dest == 0) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }

    for (iree_host_size_t i = 0; i < false_region->exit_edges.count; ++i) {
      loom_low_lower_block_ref_t previous_low_dest = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
          context, graph->edges[false_region->exit_edges.values[i]].terminator,
          graph->edges[false_region->exit_edges.values[i]].successor_index,
          plan->restore_block, &previous_low_dest));
      if (common_restore_dest == 0) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }
    if (common_restore_dest == 0) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_single_continuation"));
    }
    plan->restore_dest = common_restore_dest;

    loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND,
                                 plan));
    return iree_ok_status();
  }

  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_masked_merge(
      context, source_op, continuation, merge_low_dest,
      loom_amdgpu_single_region_exit_args(graph, false_region,
                                          continuation->arg_count),
      loom_amdgpu_single_region_exit_args(graph, true_region,
                                          continuation->arg_count),
      &plan->merge_constants));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, 0, merge_low_dest, &plan->else_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, 0, 0, &plan->no_true_else_entry_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, 0, 0, &plan->true_only_restore_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, 0, merge_low_dest, &plan->merge_restore_block));

  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, source_op, 1, plan->no_true_else_entry_block,
      &plan->else_body_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_plan_carrier_conditional_edge(
      context, source_op, plan->else_body_block, &plan->else_body_edge));

  loom_low_lower_block_ref_t common_restore_dest = 0;
  for (iree_host_size_t i = 0; i < true_region->exit_edges.count; ++i) {
    loom_low_lower_block_ref_t previous_low_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, graph->edges[true_region->exit_edges.values[i]].terminator,
        graph->edges[true_region->exit_edges.values[i]].successor_index,
        plan->else_dispatch_block, &previous_low_dest));
    if (common_restore_dest == 0) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }
  for (iree_host_size_t i = 0; i < false_region->exit_edges.count; ++i) {
    loom_low_lower_block_ref_t previous_low_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, graph->edges[false_region->exit_edges.values[i]].terminator,
        graph->edges[false_region->exit_edges.values[i]].successor_index,
        plan->merge_restore_block, &previous_low_dest));
    if (common_restore_dest == 0) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  if (common_restore_dest == 0) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  plan->merge_restore_dest = common_restore_dest;

  loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND, plan));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_exec_mask_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  bool prepared_loop_exit_region = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_loop_try_prepare_exit_region(
      context, source_op, &prepared_loop_exit_region));
  if (prepared_loop_exit_region) {
    return iree_ok_status();
  }
  bool immediate_diamond = loom_amdgpu_try_if_else_diamond(source_op);
  loom_block_t* passthrough_continuation = NULL;
  const loom_op_t* passthrough_terminator = NULL;
  bool has_false_passthrough = loom_amdgpu_try_false_passthrough_continuation(
      facts, source_op, &passthrough_continuation, &passthrough_terminator);
  bool false_path_is_direct_passthrough = false;
  if (has_false_passthrough) {
    const uint16_t false_entry =
        loom_cfg_cond_br_false_dest(source_op)->region_index;
    false_path_is_direct_passthrough =
        facts->regions.blocks[false_entry].continuation_index ==
        passthrough_continuation->region_index;
  }
  const loom_cfg_region_t* true_region = NULL;
  const loom_cfg_region_t* false_region = NULL;
  // A one-branch false block is only a passthrough when it jumps directly to
  // its retained region continuation. A loop preheader has the same local
  // shape but enters the complete false region before converging.
  if ((immediate_diamond || !false_path_is_direct_passthrough) &&
      loom_amdgpu_try_if_else_regions(facts, source_op, &true_region,
                                      &false_region)) {
    return loom_amdgpu_prepare_if_else_regions(context, source_op, true_region,
                                               false_region);
  }
  return loom_amdgpu_prepare_then_masked_region(context, source_op);
}

iree_status_t loom_amdgpu_prepare_branch(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator,
    iree_arena_allocator_t* analysis_arena) {
  (void)user_data;
  if (!loom_cfg_cond_br_isa(source_terminator)) {
    return iree_ok_status();
  }
  bool constant = false;
  if (loom_amdgpu_value_as_i1_constant(
          context, loom_cfg_cond_br_condition(source_terminator), &constant)) {
    // Constant edges still participate in the shared loop's exit accounting,
    // even though structural lowering emits no condition value for them.
    bool handled_loop = false;
    bool active_uniform = false;
    return loom_amdgpu_loop_try_prepare_exit(
        context, source_terminator, /*is_divergent=*/false, analysis_arena,
        &handled_loop, &active_uniform);
  }
  bool edge_implied_condition = false;
  bool edge_condition_proven = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cfg_cond_br_edge_implied_bool(
      context, source_terminator, &edge_implied_condition,
      &edge_condition_proven));
  if (edge_condition_proven) {
    loom_low_lower_set_branch_plan(
        context, source_terminator,
        loom_low_lower_plan_make(edge_implied_condition
                                     ? LOOM_AMDGPU_BRANCH_PLAN_EXACT_TRUE
                                     : LOOM_AMDGPU_BRANCH_PLAN_EXACT_FALSE,
                                 NULL));
    bool handled_loop = false;
    bool active_uniform = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_loop_try_prepare_exit(
        context, source_terminator, /*is_divergent=*/false, analysis_arena,
        &handled_loop, &active_uniform));
    return iree_ok_status();
  }

  const loom_type_t condition_type = loom_low_lower_value_binding_type(
      context, loom_cfg_cond_br_condition(source_terminator));
  if (!loom_amdgpu_condition_is_reg_class(context, condition_type,
                                          LOOM_AMDGPU_REG_CLASS_ID_SCC, 1) &&
      !loom_amdgpu_condition_is_reg_class(context, condition_type,
                                          LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1) &&
      !loom_amdgpu_condition_is_reg_class(context, condition_type,
                                          LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2)) {
    return loom_low_lower_emit_branch_condition_type_unsupported(
        context, source_terminator, condition_type,
        IREE_SV("amdgpu.branch_condition"));
  }

  const bool is_uniform =
      loom_amdgpu_branch_condition_is_uniform(context, source_terminator);
  const bool is_divergent =
      loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2) &&
      !is_uniform;
  bool handled_loop = false;
  bool active_uniform = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_loop_try_prepare_exit(
      context, source_terminator, is_divergent, analysis_arena, &handled_loop,
      &active_uniform));
  if (is_uniform || active_uniform) {
    loom_low_lower_set_branch_plan(
        context, source_terminator,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_UNIFORM, NULL));
  }
  if (handled_loop || !is_divergent || active_uniform) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_exec_mask_branch(context, source_terminator);
}

static iree_status_t loom_amdgpu_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_low_lower_block_ref_t restore_dest_ref) {
  loom_block_t* restore_dest =
      loom_low_lower_control_block(context, restore_dest_ref);
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  loom_value_slice_t args = {0};
  if (iree_status_is_ok(status) &&
      restore_block->arg_count == restore_dest->arg_count) {
    args = (loom_value_slice_t){
        .values = restore_block->arg_ids,
        .count = restore_block->arg_count,
    };
  } else if (iree_status_is_ok(status)) {
    status = loom_low_lower_control_append_carrier_arguments(
        context, restore_dest_ref, restore_block->arg_ids,
        restore_block->arg_count, &args);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, args.values, args.count,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_value_id_t* args,
    uint16_t arg_count) {
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, args, arg_count,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_passthrough_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_op_t* passthrough_terminator) {
  IREE_ASSERT(loom_cfg_br_isa(passthrough_terminator));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);

  // Edge materialization may read EXEC or emit vector copies. The skipped
  // arm's payload belongs to the restored parent lanes, not the narrowed arm.
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  loom_value_slice_t args = loom_cfg_br_args(passthrough_terminator);
  loom_value_slice_t low_args = {0};
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_remap_successor_args(
        context, passthrough_terminator, restore_dest, args.values, args.count,
        &low_args);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, low_args.values,
                               low_args.count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}
static iree_status_t loom_amdgpu_emit_masked_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  loom_block_t* merge_restore_block =
      loom_low_lower_control_block(context, plan->merge_restore_block);
  loom_block_t* merge_restore_dest =
      loom_low_lower_control_block(context, plan->merge_restore_dest);
  IREE_ASSERT(merge_restore_block != NULL);
  IREE_ASSERT(merge_restore_dest != NULL);
  IREE_ASSERT(plan->false_passthrough_terminator != NULL);
  IREE_ASSERT_EQ(merge_restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  loom_value_slice_t false_args =
      loom_cfg_br_args(plan->false_passthrough_terminator);
  const uint16_t arg_count = merge_restore_block->arg_count;
  const uint16_t merge_arg_count =
      loom_low_lower_control_carrier_argument_offset(context,
                                                     plan->merge_restore_dest);
  IREE_ASSERT_EQ(arg_count, merge_restore_dest->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*merged_args), (void**)&merged_args);
  }
  loom_value_slice_t low_false_args = {0};
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_remap_successor_args(
        context, plan->false_passthrough_terminator, merge_restore_dest,
        false_args.values, false_args.count, &low_false_args);
  }
  for (uint16_t i = 0; i < merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t low_false_value = low_false_args.values[i];
    const loom_value_id_t low_true_value =
        loom_block_arg_id(merge_restore_block, i);
    loom_module_t* module = loom_low_lower_context_module(context);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, saved_exec,
        plan->merge_constants ? plan->merge_constants[i]
                              : (loom_amdgpu_mask_merge_constants_t){0},
        low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  for (uint16_t i = merge_arg_count; i < arg_count && iree_status_is_ok(status);
       ++i) {
    merged_args[i] = loom_block_arg_id(merge_restore_block, i);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, merge_restore_dest, merged_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_if_else_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  loom_block_t* merge_restore_block =
      loom_low_lower_control_block(context, plan->merge_restore_block);
  loom_block_t* merge_restore_dest =
      loom_low_lower_control_block(context, plan->merge_restore_dest);
  loom_block_t* else_dispatch_block =
      loom_low_lower_control_block(context, plan->else_dispatch_block);
  IREE_ASSERT(merge_restore_block != NULL);
  IREE_ASSERT(merge_restore_dest != NULL);
  IREE_ASSERT_EQ(merge_restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t merge_arg_count = plan->if_else_merge_arg_count;
  const uint16_t arg_count = merge_restore_block->arg_count;
  IREE_ASSERT_EQ(arg_count, merge_restore_dest->arg_count);
  IREE_ASSERT_EQ(arg_count, else_dispatch_block->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*merged_args), (void**)&merged_args);
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t low_false_value =
        loom_block_arg_id(merge_restore_block, i);
    const loom_value_id_t low_true_value =
        loom_block_arg_id(else_dispatch_block, i);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, saved_exec,
        plan->merge_constants ? plan->merge_constants[i]
                              : (loom_amdgpu_mask_merge_constants_t){0},
        low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  for (uint16_t i = merge_arg_count; i < arg_count && iree_status_is_ok(status);
       ++i) {
    merged_args[i] = loom_block_arg_id(else_dispatch_block, i);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, merge_restore_dest, merged_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_no_true_else_entry_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, const loom_amdgpu_branch_plan_t* plan) {
  loom_block_t* no_true_else_entry_block =
      loom_low_lower_control_block(context, plan->no_true_else_entry_block);
  loom_block_t* else_dispatch_block =
      loom_low_lower_control_block(context, plan->else_dispatch_block);
  IREE_ASSERT_EQ(no_true_else_entry_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, no_true_else_entry_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t arg_count = plan->if_else_merge_arg_count;
  loom_value_id_t* zero_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*zero_args), (void**)&zero_args);
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*placeholders), (void**)&placeholders);
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < arg_count && iree_status_is_ok(status); ++i) {
    zero_args[i] = LOOM_VALUE_ID_INVALID;
    // The no-true path reaches this block with EXEC empty. Restore the original
    // lane mask before defining placeholder VGPRs so the later hardware select
    // never reads an unwritten register operand, even though the placeholders
    // are semantically ignored when every active lane takes the else arm.
    const loom_value_id_t true_arg = loom_block_arg_id(else_dispatch_block, i);
    const loom_type_t value_type = loom_module_value_type(module, true_arg);
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, value_type)) {
        zero_args[i] = placeholders[j].value;
        break;
      }
    }
    if (zero_args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    status = loom_amdgpu_emit_zero_merge_value(context, source_op, value_type,
                                               &zero_args[i]);
    if (iree_status_is_ok(status)) {
      placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
          .type = value_type,
          .value = zero_args[i],
      };
    }
  }
  loom_value_slice_t args = {0};
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_control_append_carrier_arguments(
        context, plan->else_dispatch_block, zero_args, arg_count, &args);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, else_dispatch_block, args.values,
                               args.count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_else_dispatch_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    loom_type_t active_type, const loom_amdgpu_branch_plan_t* plan) {
  loom_block_t* else_dispatch_block =
      loom_low_lower_control_block(context, plan->else_dispatch_block);
  loom_block_t* true_only_restore_block =
      loom_low_lower_control_block(context, plan->true_only_restore_block);
  loom_block_t* restore_block =
      loom_low_lower_control_block(context, plan->restore_block);
  loom_block_t* else_body_block = loom_low_lower_control_block(
      context, plan->else_body_edge.branch_destination);
  IREE_ASSERT_EQ(else_dispatch_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, else_dispatch_block);

  loom_op_t* else_active_op = NULL;
  iree_status_t status = iree_ok_status();
  if (plan->if_else_merge_arg_count == 0) {
    // Store-only branches leave EXEC set to the true lanes. Invert those lanes
    // within saved EXEC without retaining the condition through the true body.
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), &active_type, 1,
        &else_active_op);
  } else {
    // A no-true placeholder predecessor may have restored EXEC to initialize
    // merge values. Derive the false lanes directly from the saved entry mask.
    const loom_value_id_t else_mask_operands[] = {
        saved_exec,
        low_condition,
    };
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ANDN2_B64_EXEC,
        else_mask_operands, IREE_ARRAYSIZE(else_mask_operands),
        loom_named_attr_slice_empty(), &active_type, 1, &else_active_op);
  }
  if (iree_status_is_ok(status)) {
    loom_block_t* const restore_dest = true_only_restore_block != NULL
                                           ? true_only_restore_block
                                           : restore_block;
    loom_op_t* low_cond_br_op = NULL;
    status = loom_low_cond_br_build(loom_low_lower_context_builder(context),
                                    loom_op_const_results(else_active_op)[0],
                                    else_body_block, restore_dest,
                                    source_op->location, &low_cond_br_op);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest, loom_type_t condition_type) {
  loom_low_lower_plan_t branch_plan = loom_low_lower_plan_empty();
  if (!loom_low_lower_lookup_branch_plan(context, source_op, &branch_plan)) {
    IREE_ASSERT_UNREACHABLE("AMDGPU divergent branch has no prepared plan");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (loom_amdgpu_loop_plan_isa(branch_plan)) {
    return loom_amdgpu_loop_emit_cond_branch(context, source_op, branch_plan,
                                             low_condition, low_true_dest,
                                             low_false_dest, condition_type);
  }
  if (branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND) {
    IREE_ASSERT_UNREACHABLE("AMDGPU divergent branch plan id is invalid");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_amdgpu_branch_plan_t* plan =
      (const loom_amdgpu_branch_plan_t*)branch_plan.target_data;
  IREE_ASSERT(plan != NULL);

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
  const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
  if (loom_low_lower_control_block(context, plan->restore_block) != NULL) {
    if (plan->restore_passthrough_terminator != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_passthrough_block(
          context, source_op, saved_exec,
          loom_low_lower_control_block(context, plan->restore_block),
          loom_low_lower_control_block(context, plan->restore_dest),
          plan->restore_passthrough_terminator));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_block(
          context, source_op, saved_exec,
          loom_low_lower_control_block(context, plan->restore_block),
          plan->restore_dest));
    }
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      loom_low_lower_control_block(context, plan->merge_restore_block) !=
          NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
  }

  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, active_type, plan));
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_no_true_else_entry_block(
        context, source_op, saved_exec, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_branch(
        context, source_op, saved_exec,
        loom_low_lower_control_block(context, plan->true_only_restore_block),
        loom_low_lower_control_block(context, plan->merge_restore_dest),
        loom_low_lower_control_block(context, plan->else_dispatch_block)
            ->arg_ids,
        loom_low_lower_control_block(context, plan->else_dispatch_block)
            ->arg_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_if_else_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, active_type, plan));
  }

  return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                            low_true_dest, low_false_dest);
}

// A uniform predicate may still need a native mask for vector selects. Its
// storage does not make the branch divergent. Ignore inactive lanes without
// changing EXEC or introducing masked merges for the branch payloads.
static iree_status_t loom_amdgpu_emit_uniform_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_uniform_i1_condition(
      context, source_op, low_condition, &low_condition));
  return loom_amdgpu_emit_plain_cond_branch(context, source_op, low_condition,
                                            low_true_dest, low_false_dest);
}

iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest) {
  (void)user_data;
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t condition_type = loom_module_value_type(module, low_condition);
  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SCC, 1)) {
    return loom_amdgpu_emit_plain_cond_branch(context, source_op, low_condition,
                                              low_true_dest, low_false_dest);
  }

  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1)) {
    return loom_amdgpu_emit_sgpr_bool_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest);
  }

  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2)) {
    loom_low_lower_plan_t branch_plan = loom_low_lower_plan_empty();
    loom_low_lower_lookup_branch_plan(context, source_op, &branch_plan);
    if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_EXACT_FALSE ||
        branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_EXACT_TRUE) {
      loom_block_t* low_dest =
          branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_EXACT_TRUE ? low_true_dest
                                                               : low_false_dest;
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(loom_low_lower_context_builder(context),
                               low_dest, NULL, 0, source_op->location,
                               &low_br_op);
    }
    if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_UNIFORM) {
      return loom_amdgpu_emit_uniform_mask_cond_branch(
          context, source_op, low_condition, low_true_dest, low_false_dest);
    }
    return loom_amdgpu_emit_exec_mask_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest,
        condition_type);
  }

  IREE_ASSERT_UNREACHABLE("AMDGPU branch condition carrier was not prepared");
  IREE_BUILTIN_UNREACHABLE();
}
