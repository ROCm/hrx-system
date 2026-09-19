// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/target/facts.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_table.h"

static const loom_op_t* loom_value_fact_induction_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static bool loom_value_fact_induction_is_invariant(
    const loom_module_t* module, const loom_cfg_loop_nest_t* loops,
    uint16_t loop_index, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_block_t* block = NULL;
  if (loom_value_is_block_arg(value)) {
    block = loom_value_def_block(value);
  } else {
    const loom_op_t* op = loom_value_def_op(value);
    if (loom_index_constant_isa(op)) {
      return true;
    }
    block = op->parent_block;
  }
  return block->parent_region != loops->graph->region ||
         !loom_cfg_loop_nest_contains(loops, loop_index, block->region_index);
}

static loom_value_fact_cfg_induction_t loom_value_fact_induction_recognize(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_cfg_loop_nest_t* loops, uint16_t loop_index) {
  const loom_value_fact_cfg_induction_t unknown = {
      .value = LOOM_VALUE_ID_INVALID,
  };
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->entries.count != 1 || loop->backedges.count != 1 ||
      loop->exits.count != 1) {
    return unknown;
  }
  const loom_cfg_graph_t* graph = loops->graph;
  const loom_cfg_edge_info_t* exit = &graph->edges[loop->exits.unique_index];
  const loom_block_t* header = graph->blocks[loop->header_index].block;
  const loom_op_t* terminator = header->last_op;
  if (exit->source_block_index != loop->header_index ||
      exit->successor_index != 1 || !loom_cfg_cond_br_isa(terminator)) {
    return unknown;
  }
  const loom_value_id_t condition = loom_value_fact_table_query_identity(
      table, loom_cfg_cond_br_condition(terminator));
  const loom_op_t* compare =
      loom_value_fact_induction_defining_op(module, condition);
  if (compare && iree_any_bit_set(compare->traits, LOOM_TRAIT_CONSTANT_LIKE) &&
      loom_value_facts_is_zero(
          loom_value_fact_table_lookup(table, condition))) {
    return (loom_value_fact_cfg_induction_t){
        .value = LOOM_VALUE_ID_INVALID,
        .exits_at_header = true,
    };
  }
  if (!compare || !loom_index_cmp_isa(compare)) {
    return unknown;
  }
  loom_loop_bound_flags_t bound_flags = LOOM_LOOP_BOUND_NONE;
  switch (loom_index_cmp_predicate(compare)) {
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      bound_flags = LOOM_LOOP_BOUND_SIGNED;
      break;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      bound_flags = LOOM_LOOP_BOUND_SIGNED | LOOM_LOOP_BOUND_INCLUSIVE;
      break;
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      break;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      bound_flags = LOOM_LOOP_BOUND_INCLUSIVE;
      break;
    default:
      return unknown;
  }
  const loom_value_id_t counter =
      loom_value_fact_table_query_identity(table, loom_index_cmp_lhs(compare));
  const loom_value_t* value = loom_module_value(module, counter);
  if (!loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != header ||
      !loom_type_is_scalar(loom_module_value_type(module, counter))) {
    return unknown;
  }
  const uint16_t argument_index = loom_value_def_index(value);
  const loom_cfg_edge_info_t* entry = &graph->edges[loop->entries.unique_index];
  const loom_cfg_edge_info_t* backedge =
      &graph->edges[loop->backedges.unique_index];
  const loom_op_t* entry_branch =
      graph->blocks[entry->source_block_index].block->last_op;
  const loom_op_t* backedge_branch =
      graph->blocks[backedge->source_block_index].block->last_op;
  if (!loom_cfg_br_isa(entry_branch) || !loom_cfg_br_isa(backedge_branch)) {
    return unknown;
  }
  const loom_value_id_t upper_bound =
      loom_value_fact_table_query_identity(table, loom_index_cmp_rhs(compare));
  const loom_value_id_t initial_value = loom_value_fact_table_query_identity(
      table, loom_cfg_br_args(entry_branch).values[argument_index]);
  if (!loom_value_fact_induction_is_invariant(module, loops, loop_index,
                                              initial_value) ||
      !loom_value_fact_induction_is_invariant(module, loops, loop_index,
                                              upper_bound)) {
    return unknown;
  }
  loom_value_fact_cfg_induction_t induction = {
      .value = counter,
      .initial_value = initial_value,
      .upper_bound = upper_bound,
      .step = LOOM_VALUE_ID_INVALID,
      .bound_flags = bound_flags,
  };
  // A false entry guard proves an empty loop even when its dead increment has
  // folded away. Retain the guard equation independently of the backedge form.
  const loom_value_id_t next = loom_value_fact_table_query_identity(
      table, loom_cfg_br_args(backedge_branch).values[argument_index]);
  const loom_op_t* add = loom_value_fact_induction_defining_op(module, next);
  if (!add || !loom_index_add_isa(add)) {
    return induction;
  }
  const loom_value_id_t lhs =
      loom_value_fact_table_query_identity(table, loom_index_add_lhs(add));
  const loom_value_id_t rhs =
      loom_value_fact_table_query_identity(table, loom_index_add_rhs(add));
  const loom_value_id_t step = lhs == counter ? rhs : lhs;
  if ((lhs == counter || rhs == counter) &&
      loom_value_fact_induction_is_invariant(module, loops, loop_index, step)) {
    induction.step = step;
  }
  return induction;
}

void loom_value_fact_cfg_update_induction(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index) {
  const uint16_t loop_index =
      loom_cfg_loop_nest_innermost(&region->loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      region->loops.loops[loop_index].header_index != block_index) {
    return;
  }
  region->inductions[loop_index] = loom_value_fact_induction_recognize(
      table, module, &region->loops, loop_index);
}

loom_loop_recurrence_facts_t loom_value_fact_cfg_induction_facts(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_induction_t* induction) {
  const loom_loop_recurrence_facts_t unknown = {
      .values = loom_value_facts_unknown(),
  };
  if (induction->exits_at_header) {
    return (loom_loop_recurrence_facts_t){
        .values = loom_value_facts_unknown(),
        .trip_count_known = true,
    };
  }
  if (induction->value == LOOM_VALUE_ID_INVALID) {
    return unknown;
  }
  const loom_value_facts_t initial =
      loom_value_fact_table_lookup(table, induction->initial_value);
  const loom_value_facts_t upper =
      loom_value_fact_table_lookup(table, induction->upper_bound);
  const loom_value_facts_t step =
      loom_value_fact_table_lookup(table, induction->step);
  if (!loom_value_facts_is_exact(initial) ||
      !loom_value_facts_is_exact(upper)) {
    return unknown;
  }
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(module, induction->value));
  uint8_t bitwidth = 64;
  if (table->context.target_facts) {
    const loom_target_snapshot_t* target =
        &table->context.target_facts->storage.snapshot;
    bitwidth = scalar_type == LOOM_SCALAR_TYPE_INDEX ? target->index_bitwidth
                                                     : target->offset_bitwidth;
  }
  return loom_loop_domain_recurrence_facts(
      induction->bound_flags, bitwidth, initial.range_lo, upper.range_lo,
      loom_value_facts_is_exact(step) ? step.range_lo : 0);
}
