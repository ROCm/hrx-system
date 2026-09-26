// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_uniformity.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_dominance.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_cfg.h"

struct loom_control_uniformity_cfg_region_t {
  // CFG region summarized by this query scratch entry.
  const loom_region_t* region;
  // Graph and control dependencies borrowed from the populated fact scope.
  const loom_value_fact_cfg_region_t* facts;
  // Per-controller generation marks for mandatory-alternative queries.
  uint32_t* query_marks;
  // Current nonzero generation in query_marks.
  uint32_t query_generation;
  // Alternative target blocks dominating every left-hand footprint operation.
  uint16_t* query_lhs_blocks;
  // Mandatory alternative target for each controller in the current query.
  uint16_t* query_alternatives;
};

static iree_host_size_t loom_control_uniformity_region_hash(
    const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  bits ^= bits >> 11;
  return (iree_host_size_t)bits;
}

static void loom_control_uniformity_insert_cfg_region(
    loom_control_uniformity_cfg_region_t** slots, iree_host_size_t capacity,
    loom_control_uniformity_cfg_region_t* summary) {
  iree_host_size_t slot_index =
      loom_control_uniformity_region_hash(summary->region) & (capacity - 1);
  while (slots[slot_index]) {
    slot_index = (slot_index + 1) & (capacity - 1);
  }
  slots[slot_index] = summary;
}

static iree_status_t loom_control_uniformity_reserve_cfg_regions(
    loom_control_uniformity_info_t* info, iree_host_size_t minimum_count) {
  iree_host_size_t capacity = info->cfg_regions.capacity;
  if (capacity == 0) {
    capacity = 8;
  }
  while (minimum_count > capacity - capacity / 4) {
    if (capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "control summary capacity overflow");
    }
    capacity *= 2;
  }
  if (capacity == info->cfg_regions.capacity) {
    return iree_ok_status();
  }

  loom_control_uniformity_cfg_region_t** slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      info->arena, capacity, sizeof(*slots), (void**)&slots));
  memset(slots, 0, capacity * sizeof(*slots));
  for (iree_host_size_t i = 0; i < info->cfg_regions.capacity; ++i) {
    loom_control_uniformity_cfg_region_t* summary = info->cfg_regions.slots[i];
    if (summary) {
      loom_control_uniformity_insert_cfg_region(slots, capacity, summary);
    }
  }
  info->cfg_regions.slots = slots;
  info->cfg_regions.capacity = capacity;
  return iree_ok_status();
}

static loom_control_uniformity_cfg_region_t*
loom_control_uniformity_lookup_cfg_region(
    const loom_control_uniformity_info_t* info, const loom_region_t* region) {
  if (info->cfg_regions.capacity == 0) {
    return NULL;
  }
  iree_host_size_t slot_index = loom_control_uniformity_region_hash(region) &
                                (info->cfg_regions.capacity - 1);
  while (info->cfg_regions.slots[slot_index]) {
    loom_control_uniformity_cfg_region_t* summary =
        info->cfg_regions.slots[slot_index];
    if (summary->region == region) {
      return summary;
    }
    slot_index = (slot_index + 1) & (info->cfg_regions.capacity - 1);
  }
  return NULL;
}

static iree_status_t loom_control_uniformity_cfg_region_initialize(
    const loom_control_uniformity_info_t* info, const loom_region_t* region,
    loom_control_uniformity_cfg_region_t* summary) {
  memset(summary, 0, sizeof(*summary));
  summary->region = region;
  summary->facts =
      loom_value_fact_table_lookup_cfg_region(info->fact_table, region);
  if (!summary->facts || !summary->facts->control_structure.node_count) {
    return iree_ok_status();
  }
  const uint32_t count = summary->facts->graph.block_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      info->arena, count, sizeof(*summary->query_marks),
      (void**)&summary->query_marks));
  memset(summary->query_marks, 0, count * sizeof(*summary->query_marks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      info->arena, count, sizeof(*summary->query_lhs_blocks),
      (void**)&summary->query_lhs_blocks));
  return iree_arena_allocate_array(info->arena, count,
                                   sizeof(*summary->query_alternatives),
                                   (void**)&summary->query_alternatives);
}

// Walk the retained dominator tree, collecting mandatory incoming choices.
// Entry-predecessor classification already accounts for all bypasses and
// backedges. Possible postdominator control paths cannot establish this proof,
// especially when synthetic exits conservatively model nontermination.
static iree_host_size_t loom_control_uniformity_cfg_collect_alternatives(
    const loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary, uint16_t block_index,
    loom_value_fact_uniform_scope_t required_scope, uint16_t* out_blocks) {
  const loom_cfg_graph_t* graph = &summary->facts->graph;
  const loom_cfg_dominance_t* dominance = &summary->facts->dominance;
  if (++summary->query_generation == 0) {
    memset(summary->query_marks, 0,
           graph->block_count * sizeof(*summary->query_marks));
    summary->query_generation = 1;
  }
  if (!graph->blocks[block_index].reachable) {
    return 0;
  }
  iree_host_size_t count = 0;
  for (uint16_t target = block_index; target != 0;
       target = dominance->immediate_dominators[target]) {
    const uint16_t source = dominance->entry_predecessors[target];
    if (source == LOOM_CFG_DOMINATOR_INVALID ||
        graph->blocks[source].component_is_cyclic) {
      continue;
    }
    const loom_cfg_edge_index_span_t successors =
        loom_cfg_graph_successor_edges(graph, source);
    const loom_value_id_t selector =
        graph->edges[successors.values[0]].selector_value_id;
    if (!loom_value_facts_is_uniform_at_scope(
            loom_value_fact_table_lookup(info->fact_table, selector),
            required_scope)) {
      continue;
    }
    summary->query_marks[source] = summary->query_generation;
    summary->query_alternatives[source] = target;
    if (out_blocks) {
      out_blocks[count++] = target;
    }
  }
  return count;
}

static uint16_t loom_control_uniformity_cfg_alternative(
    const loom_control_uniformity_cfg_region_t* summary, uint16_t source) {
  return summary->query_marks[source] == summary->query_generation
             ? summary->query_alternatives[source]
             : LOOM_CFG_DOMINATOR_INVALID;
}

static iree_status_t loom_control_uniformity_cfg_region(
    loom_control_uniformity_info_t* info, const loom_region_t* region,
    loom_control_uniformity_cfg_region_t** out_summary) {
  *out_summary = loom_control_uniformity_lookup_cfg_region(info, region);
  if (*out_summary) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_control_uniformity_reserve_cfg_regions(
      info, info->cfg_regions.count + 1));
  loom_control_uniformity_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(info->arena, sizeof(*summary), (void**)&summary));
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_region_initialize(info, region, summary));
  loom_control_uniformity_insert_cfg_region(
      info->cfg_regions.slots, info->cfg_regions.capacity, summary);
  ++info->cfg_regions.count;
  *out_summary = summary;
  return iree_ok_status();
}

bool loom_control_uniformity_prove_single_entry(
    const loom_control_uniformity_info_t* info, const loom_block_t* block,
    loom_value_fact_uniform_scope_t required_scope,
    loom_condition_assumption_t* out_condition) {
  const loom_value_fact_cfg_region_t* region =
      loom_value_fact_table_lookup_cfg_region(info->fact_table,
                                              block->parent_region);
  if (!region || block->region_index == 0) {
    return false;
  }
  const loom_cfg_graph_t* graph = &region->graph;
  if (!graph->blocks[block->region_index].reachable) {
    return false;
  }
  const loom_cfg_edge_index_span_t incoming =
      loom_cfg_graph_predecessor_edges(graph, block->region_index);
  if (incoming.count != 1) {
    return false;
  }
  const loom_cfg_edge_info_t* edge = &graph->edges[incoming.values[0]];
  if (!loom_cfg_cond_br_isa(edge->terminator) ||
      !loom_control_uniformity_prove_execution(info, edge->terminator,
                                               required_scope, NULL)) {
    return false;
  }
  *out_condition = (loom_condition_assumption_t){
      .condition = edge->selector_value_id,
      .assumed_truth = edge->successor_index == 0,
  };
  return true;
}

static bool loom_control_uniformity_prove_value(
    const loom_control_uniformity_info_t* info, const loom_op_t* control_op,
    loom_value_id_t control_value, loom_control_uniformity_source_t source,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure) {
  const loom_value_facts_t facts =
      control_value == LOOM_VALUE_ID_INVALID
          ? loom_value_facts_unknown()
          : loom_value_fact_table_lookup(info->fact_table, control_value);
  if (loom_value_facts_is_uniform_at_scope(facts, required_scope)) {
    return true;
  }
  if (out_failure) {
    *out_failure = (loom_control_uniformity_failure_t){
        .control_op = control_op,
        .control_value = control_value,
        .control_facts = facts,
        .source = source,
    };
  }
  return false;
}

static bool loom_control_uniformity_prove_cfg_block(
    const loom_control_uniformity_info_t* info, const loom_block_t* block,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure) {
  if (!block || !block->parent_region ||
      (!iree_any_bit_set(block->parent_region->flags,
                         LOOM_REGION_INSTANCE_FLAG_CFG) &&
       block->parent_region->block_count <= 1)) {
    return true;
  }

  const loom_value_fact_cfg_region_t* facts =
      loom_value_fact_table_lookup_cfg_region(info->fact_table,
                                              block->parent_region);
  if (!facts || !facts->control_structure.available ||
      block->region_index >= facts->graph.block_count) {
    if (out_failure) {
      *out_failure = (loom_control_uniformity_failure_t){
          .control_value = LOOM_VALUE_ID_INVALID,
          .control_facts = loom_value_facts_unknown(),
          .source = LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION,
      };
    }
    return false;
  }
  if (loom_value_facts_is_uniform_at_scope(
          loom_value_fact_control_execution(facts->control,
                                            block->region_index),
          required_scope)) {
    return true;
  }
  if (out_failure) {
    loom_value_fact_control_prepare_diagnostics(facts->control);
    const loom_cfg_edge_info_t* edge = loom_cfg_graph_edge(
        &facts->graph, loom_value_fact_control_controller(facts->control,
                                                          block->region_index));
    const loom_value_id_t selector =
        edge ? edge->selector_value_id : LOOM_VALUE_ID_INVALID;
    *out_failure = (loom_control_uniformity_failure_t){
        .control_op = edge ? edge->terminator : NULL,
        .control_value = selector,
        .control_facts =
            selector == LOOM_VALUE_ID_INVALID
                ? loom_value_facts_unknown()
                : loom_value_fact_table_lookup(info->fact_table, selector),
        .source = edge ? LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_SELECTOR
                       : LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION,
    };
  }
  return false;
}

void loom_control_uniformity_info_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_control_uniformity_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = (loom_control_uniformity_info_t){
      .module = module,
      .fact_table = fact_table,
      .arena = arena,
  };
}

bool loom_control_uniformity_prove_execution(
    const loom_control_uniformity_info_t* info, const loom_op_t* op,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure) {
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT(required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
  if (out_failure) {
    *out_failure = (loom_control_uniformity_failure_t){
        .control_value = LOOM_VALUE_ID_INVALID,
        .control_facts = loom_value_facts_unknown(),
    };
  }
  for (const loom_op_t* current_op = op; current_op;
       current_op = current_op->parent_op) {
    if (!loom_control_uniformity_prove_cfg_block(info, current_op->parent_block,
                                                 required_scope, out_failure)) {
      return false;
    }

    const loom_op_t* ancestor_op = current_op->parent_op;
    if (!ancestor_op) {
      continue;
    }
    const loom_region_branch_t branch =
        loom_region_branch_cast(info->module, (loom_op_t*)ancestor_op);
    if (loom_region_branch_isa(branch) &&
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_region_branch_selector(branch),
            LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR, required_scope,
            out_failure)) {
      return false;
    }

    const loom_loop_like_t loop =
        loom_loop_like_cast(info->module, (loom_op_t*)ancestor_op);
    if (!loom_loop_like_isa(loop)) {
      continue;
    }
    if (!loom_loop_like_has_counted_range(loop)) {
      loom_control_uniformity_prove_value(
          info, ancestor_op, LOOM_VALUE_ID_INVALID,
          LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_CONDITION, required_scope,
          out_failure);
      return false;
    }
    if (!loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_lower_bound(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_LOWER_BOUND, required_scope,
            out_failure) ||
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_upper_bound(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_UPPER_BOUND, required_scope,
            out_failure) ||
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_step(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_STEP, required_scope,
            out_failure)) {
      return false;
    }
  }
  return true;
}

static uint16_t loom_control_uniformity_branch_region_containing_operation(
    const loom_control_uniformity_info_t* info, loom_region_branch_t branch,
    const loom_op_t* operation) {
  const loom_op_t* child = operation;
  while (child && child->parent_op != branch.op) {
    child = child->parent_op;
  }
  if (!child || !child->parent_block) {
    return UINT16_MAX;
  }
  const loom_region_t* child_region = child->parent_block->parent_region;
  for (uint16_t region_index = 0; region_index < branch.op->region_count;
       ++region_index) {
    if (loom_region_branch_region(info->module, branch,
                                  (uint8_t)region_index) == child_region) {
      return region_index;
    }
  }
  return UINT16_MAX;
}

// Returns true when |operation| itself or an enclosing control scope may
// execute repeatedly. A RegionBranch decision inside such a scope may choose
// different alternatives on different iterations and cannot establish global
// mutual exclusion.
static bool loom_control_uniformity_ancestor_may_repeat(
    const loom_control_uniformity_info_t* info, const loom_op_t* operation) {
  for (const loom_op_t* current = operation; current;
       current = current->parent_op) {
    if (loom_loop_like_isa(
            loom_loop_like_cast(info->module, (loom_op_t*)current))) {
      return true;
    }
    const loom_block_t* block = current->parent_block;
    if (!block || !block->parent_region ||
        (block->parent_region->block_count <= 1 &&
         !iree_any_bit_set(block->parent_region->flags,
                           LOOM_REGION_INSTANCE_FLAG_CFG))) {
      continue;
    }
    const loom_value_fact_cfg_region_t* facts =
        loom_value_fact_table_lookup_cfg_region(info->fact_table,
                                                block->parent_region);
    if (!facts || block->region_index >= facts->graph.block_count ||
        facts->graph.blocks[block->region_index].component_is_cyclic) {
      return true;
    }
  }
  return false;
}

static bool loom_control_uniformity_prove_region_branch_exclusion(
    const loom_control_uniformity_info_t* info, iree_host_size_t lhs_op_count,
    const loom_op_t* const* lhs_ops, iree_host_size_t rhs_op_count,
    const loom_op_t* const* rhs_ops,
    loom_value_fact_uniform_scope_t required_scope) {
  for (const loom_op_t* child = lhs_ops[0]; child && child->parent_op;
       child = child->parent_op) {
    const loom_region_branch_t branch =
        loom_region_branch_cast(info->module, (loom_op_t*)child->parent_op);
    if (!loom_region_branch_isa(branch) ||
        loom_control_uniformity_ancestor_may_repeat(info, branch.op) ||
        !loom_control_uniformity_prove_value(
            info, branch.op, loom_region_branch_selector(branch),
            LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR, required_scope,
            NULL)) {
      continue;
    }

    bool all_contained = true;
    for (iree_host_size_t i = 0; i < lhs_op_count && all_contained; ++i) {
      const uint16_t lhs_region =
          loom_control_uniformity_branch_region_containing_operation(
              info, branch, lhs_ops[i]);
      if (lhs_region == UINT16_MAX) {
        all_contained = false;
        break;
      }
      for (iree_host_size_t j = 0; j < rhs_op_count; ++j) {
        const uint16_t rhs_region =
            loom_control_uniformity_branch_region_containing_operation(
                info, branch, rhs_ops[j]);
        if (rhs_region == UINT16_MAX || rhs_region == lhs_region) {
          all_contained = false;
          break;
        }
      }
    }
    if (all_contained) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_control_uniformity_prove_mutually_exclusive_execution(
    loom_control_uniformity_info_t* info, iree_host_size_t lhs_op_count,
    const loom_op_t* const* lhs_ops, iree_host_size_t rhs_op_count,
    const loom_op_t* const* rhs_ops,
    loom_value_fact_uniform_scope_t required_scope, bool* out_proven) {
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_GT(lhs_op_count, 0u);
  IREE_ASSERT_ARGUMENT(lhs_ops);
  IREE_ASSERT_GT(rhs_op_count, 0u);
  IREE_ASSERT_ARGUMENT(rhs_ops);
  IREE_ASSERT_ARGUMENT(out_proven);
  IREE_ASSERT(required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
  *out_proven = false;
  for (iree_host_size_t i = 0; i < lhs_op_count; ++i) {
    IREE_ASSERT_ARGUMENT(lhs_ops[i]);
  }
  for (iree_host_size_t i = 0; i < rhs_op_count; ++i) {
    IREE_ASSERT_ARGUMENT(rhs_ops[i]);
  }
  if (loom_control_uniformity_prove_region_branch_exclusion(
          info, lhs_op_count, lhs_ops, rhs_op_count, rhs_ops, required_scope)) {
    *out_proven = true;
    return iree_ok_status();
  }

  const loom_block_t* first_block = lhs_ops[0]->parent_block;
  if (!first_block || !first_block->parent_region) {
    return iree_ok_status();
  }
  const loom_region_t* region = first_block->parent_region;
  for (iree_host_size_t i = 0; i < lhs_op_count; ++i) {
    const loom_block_t* block = lhs_ops[i]->parent_block;
    if (!block || block->parent_region != region) {
      return iree_ok_status();
    }
  }
  for (iree_host_size_t i = 0; i < rhs_op_count; ++i) {
    const loom_block_t* block = rhs_ops[i]->parent_block;
    if (!block || block->parent_region != region) {
      return iree_ok_status();
    }
  }
  if (loom_control_uniformity_ancestor_may_repeat(info,
                                                  lhs_ops[0]->parent_op)) {
    return iree_ok_status();
  }

  loom_control_uniformity_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_region(info, region, &summary));
  if (!summary->query_marks) {
    return iree_ok_status();
  }

  if (first_block->region_index >= summary->facts->graph.block_count) {
    return iree_ok_status();
  }
  iree_host_size_t candidate_count =
      loom_control_uniformity_cfg_collect_alternatives(
          info, summary, first_block->region_index, required_scope,
          summary->query_lhs_blocks);
  for (iree_host_size_t i = 0; i < lhs_op_count; ++i) {
    const loom_block_t* block = lhs_ops[i]->parent_block;
    if (block->region_index >= summary->facts->graph.block_count) {
      return iree_ok_status();
    }
    if (i) {
      loom_control_uniformity_cfg_collect_alternatives(
          info, summary, block->region_index, required_scope, NULL);
    }
    iree_host_size_t retained_count = 0;
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      const uint16_t target = summary->query_lhs_blocks[j];
      const uint16_t source =
          summary->facts->dominance.entry_predecessors[target];
      if (loom_control_uniformity_cfg_alternative(summary, source) == target) {
        summary->query_lhs_blocks[retained_count++] = target;
      }
    }
    candidate_count = retained_count;
    if (candidate_count == 0) {
      return iree_ok_status();
    }
  }

  for (iree_host_size_t i = 0; i < rhs_op_count; ++i) {
    const loom_block_t* block = rhs_ops[i]->parent_block;
    if (block->region_index >= summary->facts->graph.block_count) {
      return iree_ok_status();
    }
    loom_control_uniformity_cfg_collect_alternatives(
        info, summary, block->region_index, required_scope, NULL);
    iree_host_size_t retained_count = 0;
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      const uint16_t target = summary->query_lhs_blocks[j];
      const uint16_t source =
          summary->facts->dominance.entry_predecessors[target];
      const uint16_t alternative =
          loom_control_uniformity_cfg_alternative(summary, source);
      if (alternative != LOOM_CFG_DOMINATOR_INVALID && alternative != target) {
        summary->query_lhs_blocks[retained_count++] = target;
      }
    }
    candidate_count = retained_count;
    if (candidate_count == 0) {
      return iree_ok_status();
    }
  }
  *out_proven = true;
  return iree_ok_status();
}

iree_string_view_t loom_control_uniformity_source_name(
    loom_control_uniformity_source_t source) {
  static const iree_string_view_t names[] = {
      [LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR] =
          IREE_SVL("region_selector"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_LOWER_BOUND] =
          IREE_SVL("loop_lower_bound"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_UPPER_BOUND] =
          IREE_SVL("loop_upper_bound"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_STEP] = IREE_SVL("loop_step"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_CONDITION] =
          IREE_SVL("loop_condition"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_SELECTOR] = IREE_SVL("cfg_selector"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION] =
          IREE_SVL("cfg_execution"),
  };
  IREE_ASSERT_LT((uint32_t)source, IREE_ARRAYSIZE(names));
  return names[source];
}

iree_string_view_t loom_control_uniformity_scope_name(
    loom_value_fact_uniform_scope_t scope) {
  static const iree_string_view_t names[] = {
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE] = IREE_SVL("unknown"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP] = IREE_SVL("subgroup"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP] = IREE_SVL("workgroup"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER] = IREE_SVL("cluster"),
  };
  IREE_ASSERT_LT((uint32_t)scope, IREE_ARRAYSIZE(names));
  return names[scope];
}

iree_string_view_t loom_control_uniformity_fact_distribution_name(
    loom_value_facts_t facts) {
  if (loom_value_facts_is_lane_predicate(facts)) {
    return IREE_SV("lane_predicate");
  }
  if (loom_value_facts_is_lane_varying(facts)) {
    return IREE_SV("lane_varying");
  }
  return loom_control_uniformity_scope_name(
      loom_value_facts_uniform_scope(facts));
}
