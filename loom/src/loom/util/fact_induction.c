// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_induction.h"

#include "loom/analysis/condition_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
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
    if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
      return true;
    }
    block = op->parent_block;
  }
  return block->parent_region != loops->graph->region ||
         !loom_cfg_loop_nest_contains(loops, loop_index, block->region_index);
}

typedef struct loom_value_fact_induction_guard_t {
  loom_value_id_t counter;
  // Invariant update operand when the guard observes the next state.
  loom_value_id_t step;
  // Whether the observed next state subtracts |step|.
  bool subtract_step;
} loom_value_fact_induction_guard_t;

static bool loom_value_fact_induction_update(const loom_op_t* op,
                                             bool* out_subtract) {
  if (loom_index_add_isa(op) || loom_scalar_addi_isa(op)) {
    *out_subtract = false;
    return true;
  }
  if (loom_index_sub_isa(op) || loom_scalar_subi_isa(op)) {
    *out_subtract = true;
    return true;
  }
  return false;
}

static bool loom_value_fact_induction_guard_match(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* header, loom_value_id_t candidate,
    loom_value_fact_induction_guard_t* out_guard) {
  const loom_value_id_t identity =
      loom_value_fact_table_query_identity(table, candidate);
  const loom_value_t* value = loom_module_value(module, identity);
  if (loom_value_is_block_arg(value) && loom_value_def_block(value) == header &&
      loom_type_is_scalar(loom_module_value_type(module, identity))) {
    *out_guard = (loom_value_fact_induction_guard_t){
        .counter = identity,
        .step = LOOM_VALUE_ID_INVALID,
    };
    return true;
  }

  const loom_op_t* update =
      loom_value_fact_induction_defining_op(module, identity);
  bool subtract_step = false;
  if (!update || !loom_value_fact_induction_update(update, &subtract_step)) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(update);
  const loom_value_id_t lhs =
      loom_value_fact_table_query_identity(table, operands[0]);
  const loom_value_id_t rhs =
      loom_value_fact_table_query_identity(table, operands[1]);
  loom_value_id_t counter = lhs;
  loom_value_id_t step = operands[1];
  value = loom_module_value(module, counter);
  if ((!loom_value_is_block_arg(value) ||
       loom_value_def_block(value) != header) &&
      !subtract_step) {
    counter = rhs;
    step = operands[0];
    value = loom_module_value(module, counter);
  }
  if (!loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != header ||
      !loom_type_is_scalar(loom_module_value_type(module, counter))) {
    return false;
  }
  *out_guard = (loom_value_fact_induction_guard_t){
      .counter = counter,
      .step = step,
      .subtract_step = subtract_step,
  };
  return true;
}

static bool loom_value_fact_induction_comparison(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* compare, const loom_block_t* header,
    loom_value_fact_induction_guard_t* out_guard, loom_value_id_t* out_bound,
    loom_loop_bound_flags_t* out_flags) {
  loom_condition_integer_comparison_t comparison;
  if (!loom_condition_integer_comparison_describe(module, compare,
                                                  &comparison)) {
    return false;
  }
  loom_symbolic_integer_relation_t relation;
  bool unsigned_order = false;
  if (!loom_condition_integer_comparison_relation(&comparison, &relation,
                                                  &unsigned_order)) {
    return false;
  }
  loom_loop_bound_flags_t bound_flags =
      unsigned_order ? LOOM_LOOP_BOUND_NONE : LOOM_LOOP_BOUND_SIGNED;
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      break;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      bound_flags |= LOOM_LOOP_BOUND_INCLUSIVE;
      break;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      bound_flags |= LOOM_LOOP_BOUND_DESCENDING;
      break;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      bound_flags |= LOOM_LOOP_BOUND_DESCENDING | LOOM_LOOP_BOUND_INCLUSIVE;
      break;
    default:
      return false;
  }
  loom_value_fact_induction_guard_t guard;
  loom_value_id_t bound = comparison.rhs;
  if (!loom_value_fact_induction_guard_match(table, module, header,
                                             comparison.lhs, &guard)) {
    if (!loom_value_fact_induction_guard_match(table, module, header,
                                               comparison.rhs, &guard)) {
      return false;
    }
    bound = comparison.lhs;
    bound_flags ^= LOOM_LOOP_BOUND_DESCENDING;
  }
  *out_guard = guard;
  *out_bound = bound;
  *out_flags = bound_flags;
  return true;
}

static bool loom_value_fact_induction_nonzero_comparison(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* compare, const loom_block_t* header, bool compare_truth,
    loom_value_fact_induction_guard_t* out_guard, loom_value_id_t* out_zero) {
  loom_condition_integer_comparison_t comparison;
  if (!loom_condition_integer_comparison_describe(module, compare,
                                                  &comparison)) {
    return false;
  }
  loom_symbolic_integer_relation_t relation;
  bool unsigned_order = false;
  if (!loom_condition_integer_comparison_relation(&comparison, &relation,
                                                  &unsigned_order)) {
    return false;
  }
  if (!compare_truth) {
    relation = loom_symbolic_integer_relation_invert(relation);
  }

  loom_value_id_t candidate = comparison.lhs;
  loom_value_id_t zero = comparison.rhs;
  loom_value_fact_recurrence_operand_t zero_operand =
      loom_value_fact_recurrence_operand_make(table, module, zero);
  if (!zero_operand.is_literal || zero_operand.literal != 0) {
    candidate = comparison.rhs;
    zero = comparison.lhs;
    relation = loom_symbolic_integer_relation_swap(relation);
    zero_operand = loom_value_fact_recurrence_operand_make(table, module, zero);
    if (!zero_operand.is_literal || zero_operand.literal != 0) {
      return false;
    }
  }
  if (relation != LOOM_SYMBOLIC_INTEGER_RELATION_NE &&
      !(unsigned_order && relation == LOOM_SYMBOLIC_INTEGER_RELATION_GT)) {
    return false;
  }

  candidate = loom_value_fact_table_query_identity(table, candidate);
  const loom_value_t* candidate_value = loom_module_value(module, candidate);
  if (!loom_value_is_block_arg(candidate_value) ||
      loom_value_def_block(candidate_value) != header ||
      !loom_type_is_scalar(loom_module_value_type(module, candidate))) {
    return false;
  }
  *out_guard = (loom_value_fact_induction_guard_t){
      .counter = candidate,
      .step = LOOM_VALUE_ID_INVALID,
  };
  *out_zero = zero;
  return true;
}

static bool loom_value_fact_induction_shrui_update(const loom_op_t* op,
                                                   loom_value_id_t* out_source,
                                                   loom_value_id_t* out_shift) {
  if (!loom_index_shrui_isa(op) && !loom_scalar_shrui_isa(op)) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  *out_source = operands[0];
  *out_shift = operands[1];
  return true;
}

static bool loom_value_fact_induction_steps_equal(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t lhs, bool lhs_subtract, loom_value_id_t rhs,
    bool rhs_subtract) {
  if (lhs_subtract != rhs_subtract) {
    return false;
  }
  if (loom_value_fact_table_query_identity(table, lhs) ==
      loom_value_fact_table_query_identity(table, rhs)) {
    return true;
  }
  // Retained equations may outlive current numeric facts. Only immutable
  // literals can prove equality between distinct SSA inputs.
  const loom_value_fact_recurrence_operand_t lhs_operand =
      loom_value_fact_recurrence_operand_make(table, module, lhs);
  const loom_value_fact_recurrence_operand_t rhs_operand =
      loom_value_fact_recurrence_operand_make(table, module, rhs);
  return lhs_operand.is_literal && rhs_operand.is_literal &&
         lhs_operand.literal == rhs_operand.literal;
}

static loom_value_fact_induction_t loom_value_fact_induction_recognize(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_cfg_loop_nest_t* loops, uint16_t loop_index) {
  const loom_value_fact_induction_t unknown = {
      .value = LOOM_VALUE_ID_INVALID,
  };
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->entries.count != 1 || loop->backedges.count != 1) {
    return unknown;
  }
  const loom_cfg_graph_t* graph = loops->graph;
  const loom_block_t* header = graph->blocks[loop->header_index].block;
  const loom_op_t* terminator = header->last_op;
  if (!loom_cfg_cond_br_isa(terminator)) {
    return unknown;
  }
  const bool continue_on_true = loom_cfg_loop_nest_contains(
      loops, loop_index, loom_cfg_cond_br_true_dest(terminator)->region_index);
  const bool continue_on_false = loom_cfg_loop_nest_contains(
      loops, loop_index, loom_cfg_cond_br_false_dest(terminator)->region_index);
  if (continue_on_true == continue_on_false) {
    return unknown;
  }
  const loom_value_id_t condition = loom_value_fact_table_query_identity(
      table, loom_cfg_cond_br_condition(terminator));
  const loom_op_t* compare =
      loom_value_fact_induction_defining_op(module, condition);
  int64_t condition_value = 0;
  if (compare && iree_any_bit_set(compare->traits, LOOM_TRAIT_CONSTANT_LIKE) &&
      loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(table, condition), &condition_value) &&
      ((condition_value != 0) != continue_on_true)) {
    return (loom_value_fact_induction_t){
        .value = LOOM_VALUE_ID_INVALID,
        .flags = LOOM_VALUE_FACT_INDUCTION_EXITS_AT_HEADER,
    };
  }
  if (!compare) {
    return unknown;
  }
  loom_loop_bound_flags_t bound_flags = LOOM_LOOP_BOUND_NONE;
  loom_value_fact_induction_guard_t guard_match = {
      .counter = LOOM_VALUE_ID_INVALID,
      .step = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t bound = LOOM_VALUE_ID_INVALID;
  const bool has_additive_guard = loom_value_fact_induction_comparison(
      table, module, compare, header, &guard_match, &bound, &bound_flags);
  loom_value_fact_induction_guard_t shrui_guard = {
      .counter = LOOM_VALUE_ID_INVALID,
      .step = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t shrui_zero = LOOM_VALUE_ID_INVALID;
  bool has_shrui_guard = false;
  if (!has_additive_guard) {
    has_shrui_guard = loom_value_fact_induction_nonzero_comparison(
        table, module, compare, header, continue_on_true, &shrui_guard,
        &shrui_zero);
    if (!has_shrui_guard) {
      return unknown;
    }
    guard_match = shrui_guard;
    bound = shrui_zero;
  }
  const loom_value_id_t counter = guard_match.counter;
  if (has_additive_guard && !continue_on_true) {
    bound_flags ^= LOOM_LOOP_BOUND_DESCENDING | LOOM_LOOP_BOUND_INCLUSIVE;
  }
  const loom_value_t* value = loom_module_value(module, counter);
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
  const loom_value_id_t initial_value =
      loom_cfg_br_args(entry_branch).values[argument_index];
  if (!loom_value_fact_induction_is_invariant(
          module, loops, loop_index,
          loom_value_fact_table_query_identity(table, initial_value)) ||
      !loom_value_fact_induction_is_invariant(
          module, loops, loop_index,
          loom_value_fact_table_query_identity(table, bound)) ||
      (guard_match.step != LOOM_VALUE_ID_INVALID &&
       !loom_value_fact_induction_is_invariant(
           module, loops, loop_index,
           loom_value_fact_table_query_identity(table, guard_match.step)))) {
    return unknown;
  }
  loom_value_fact_induction_t induction = {
      .value = counter,
      .initial_value =
          loom_value_fact_recurrence_operand_make(table, module, initial_value),
      .bound = loom_value_fact_recurrence_operand_make(table, module, bound),
      .step = {.value = LOOM_VALUE_ID_INVALID},
      .bound_flags = bound_flags,
      .flags = loop->exits.count > 1 ? LOOM_VALUE_FACT_INDUCTION_MAY_EXIT_EARLY
                                     : LOOM_VALUE_FACT_INDUCTION_NONE,
      .update = LOOM_VALUE_FACT_INDUCTION_UPDATE_ADDITIVE,
  };
  // A false entry guard proves an empty loop even when its dead increment has
  // folded away. Retain the guard equation independently of the backedge form.
  const loom_value_id_t next = loom_value_fact_table_query_identity(
      table, loom_cfg_br_args(backedge_branch).values[argument_index]);
  const loom_op_t* update = loom_value_fact_induction_defining_op(module, next);
  bool subtract_step = false;
  if (!update || !loom_value_fact_induction_update(update, &subtract_step)) {
    loom_value_id_t source = LOOM_VALUE_ID_INVALID;
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    const bool has_shrui_update =
        update &&
        loom_value_fact_induction_shrui_update(update, &source, &shift) &&
        loom_value_fact_table_query_identity(table, source) == counter;
    if (has_shrui_update && !has_shrui_guard) {
      has_shrui_guard = loom_value_fact_induction_nonzero_comparison(
          table, module, compare, header, continue_on_true, &shrui_guard,
          &shrui_zero);
    }
    if (has_shrui_update && has_shrui_guard && shrui_guard.counter == counter &&
        loom_value_fact_induction_is_invariant(
            module, loops, loop_index,
            loom_value_fact_table_query_identity(table, shift))) {
      induction.bound =
          loom_value_fact_recurrence_operand_make(table, module, shrui_zero);
      induction.step =
          loom_value_fact_recurrence_operand_make(table, module, shift);
      induction.bound_flags = LOOM_LOOP_BOUND_NONE;
      induction.update = LOOM_VALUE_FACT_INDUCTION_UPDATE_SHRUI_TO_ZERO;
      return induction;
    }
    return has_additive_guard && guard_match.step == LOOM_VALUE_ID_INVALID
               ? induction
               : unknown;
  }
  if (!has_additive_guard) {
    return unknown;
  }
  const loom_value_id_t* operands = loom_op_const_operands(update);
  const loom_value_id_t lhs =
      loom_value_fact_table_query_identity(table, operands[0]);
  const loom_value_id_t rhs =
      loom_value_fact_table_query_identity(table, operands[1]);
  const loom_value_id_t step = lhs == counter ? operands[1] : operands[0];
  if ((lhs != counter && (subtract_step || rhs != counter)) ||
      !loom_value_fact_induction_is_invariant(
          module, loops, loop_index,
          loom_value_fact_table_query_identity(table, step))) {
    return guard_match.step == LOOM_VALUE_ID_INVALID ? induction : unknown;
  }
  if (guard_match.step != LOOM_VALUE_ID_INVALID &&
      !loom_value_fact_induction_steps_equal(table, module, guard_match.step,
                                             guard_match.subtract_step, step,
                                             subtract_step)) {
    return unknown;
  }
  induction.step = loom_value_fact_recurrence_operand_make(table, module, step);
  if (subtract_step) {
    induction.flags |= LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP;
  }
  if (guard_match.step != LOOM_VALUE_ID_INVALID) {
    induction.flags |= LOOM_VALUE_FACT_INDUCTION_GUARD_USES_NEXT;
  }
  return induction;
}

// A value visible at this single-block boundary is either a local definition
// or a dominating capture. Literal constants remain invariant inside a region.
static bool loom_value_fact_loop_is_invariant(const loom_module_t* module,
                                              loom_loop_like_t loop,
                                              loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_block_t* block = NULL;
  if (loom_value_is_block_arg(value)) {
    block = loom_value_def_block(value);
  } else {
    const loom_op_t* op = loom_value_def_op(value);
    if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
      return true;
    }
    block = op->parent_block;
  }
  return block->parent_region != loom_loop_like_condition_region(loop) &&
         block->parent_region != loom_loop_like_body(loop);
}

static bool loom_value_fact_condition_forwards_counter(
    const loom_module_t* module, const loom_value_fact_table_t* table,
    const loom_op_t* condition, const loom_block_t* body,
    loom_value_id_t value_id, loom_value_id_t counter) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) &&
         loom_value_def_block(value) == body &&
         loom_value_fact_table_query_identity(
             table, loom_op_const_operands(
                        condition)[1 + loom_value_def_index(value)]) == counter;
}

loom_value_fact_induction_t loom_value_fact_condition_loop_induction(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, loom_value_id_t guard, bool guard_truth) {
  const loom_value_fact_induction_t unknown = {
      .value = LOOM_VALUE_ID_INVALID,
  };
  const loom_block_t* before =
      loom_region_const_entry_block(loom_loop_like_condition_region(loop));
  const loom_block_t* body =
      loom_region_const_entry_block(loom_loop_like_body(loop));
  const loom_op_t* condition = before->last_op;
  const loom_op_t* yield = body->last_op;
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop);
  const uint16_t result_count = loop.op->result_count;
  // Rewriter builders publish the shell before its terminators are complete.
  if (!condition || condition->operand_count != result_count + 1 || !yield ||
      yield->operand_count != initial.count) {
    return unknown;
  }
  const loom_value_id_t selector = loom_value_fact_table_query_identity(
      table, loom_op_const_operands(condition)[0]);
  const loom_op_t* compare =
      loom_value_fact_induction_defining_op(module, guard);
  int64_t condition_value = 0;
  if (compare && iree_any_bit_set(compare->traits, LOOM_TRAIT_CONSTANT_LIKE) &&
      loom_value_facts_as_exact_i64(loom_value_fact_table_lookup(table, guard),
                                    &condition_value) &&
      ((condition_value != 0) != guard_truth)) {
    return (loom_value_fact_induction_t){
        .value = LOOM_VALUE_ID_INVALID,
        .flags = LOOM_VALUE_FACT_INDUCTION_EXITS_AT_HEADER,
    };
  }
  loom_loop_bound_flags_t bound_flags = LOOM_LOOP_BOUND_NONE;
  loom_value_fact_induction_guard_t guard_match = {
      .counter = LOOM_VALUE_ID_INVALID,
      .step = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t bound = LOOM_VALUE_ID_INVALID;
  if (!compare || compare->parent_block != before) {
    return unknown;
  }
  const bool has_additive_guard = loom_value_fact_induction_comparison(
      table, module, compare, before, &guard_match, &bound, &bound_flags);
  loom_value_fact_induction_guard_t shrui_guard = {
      .counter = LOOM_VALUE_ID_INVALID,
      .step = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t shrui_zero = LOOM_VALUE_ID_INVALID;
  bool has_shrui_guard = false;
  if (!has_additive_guard) {
    has_shrui_guard = loom_value_fact_induction_nonzero_comparison(
        table, module, compare, before, guard_truth, &shrui_guard, &shrui_zero);
    if (!has_shrui_guard) {
      return unknown;
    }
    guard_match = shrui_guard;
    bound = shrui_zero;
  }
  const loom_value_id_t counter = guard_match.counter;
  if (has_additive_guard && !guard_truth) {
    bound_flags ^= LOOM_LOOP_BOUND_DESCENDING | LOOM_LOOP_BOUND_INCLUSIVE;
  }
  const loom_value_t* value = loom_module_value(module, counter);
  const uint16_t index = loom_value_def_index(value);
  if (!loom_value_fact_loop_is_invariant(
          module, loop, loom_value_fact_table_query_identity(table, bound)) ||
      (guard_match.step != LOOM_VALUE_ID_INVALID &&
       !loom_value_fact_loop_is_invariant(
           module, loop,
           loom_value_fact_table_query_identity(table, guard_match.step)))) {
    return unknown;
  }
  loom_value_fact_induction_t induction = {
      .value = counter,
      .initial_value = loom_value_fact_recurrence_operand_make(
          table, module, initial.values[index]),
      .bound = loom_value_fact_recurrence_operand_make(table, module, bound),
      .step = {.value = LOOM_VALUE_ID_INVALID},
      .bound_flags = bound_flags,
      .flags = guard != selector || !guard_truth
                   ? LOOM_VALUE_FACT_INDUCTION_MAY_EXIT_EARLY
                   : LOOM_VALUE_FACT_INDUCTION_NONE,
      .update = LOOM_VALUE_FACT_INDUCTION_UPDATE_ADDITIVE,
  };
  const loom_value_id_t next = loom_value_fact_table_query_identity(
      table, loom_op_const_operands(yield)[index]);
  const loom_op_t* update = loom_value_fact_induction_defining_op(module, next);
  bool subtract_step = false;
  if (!update || !loom_value_fact_induction_update(update, &subtract_step)) {
    loom_value_id_t source = LOOM_VALUE_ID_INVALID;
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    const bool has_shrui_update =
        update &&
        loom_value_fact_induction_shrui_update(update, &source, &shift) &&
        loom_value_fact_condition_forwards_counter(module, table, condition,
                                                   body, source, counter);
    if (has_shrui_update && !has_shrui_guard) {
      has_shrui_guard = loom_value_fact_induction_nonzero_comparison(
          table, module, compare, before, guard_truth, &shrui_guard,
          &shrui_zero);
    }
    if (has_shrui_update && has_shrui_guard && shrui_guard.counter == counter &&
        loom_value_fact_loop_is_invariant(
            module, loop, loom_value_fact_table_query_identity(table, shift))) {
      induction.bound =
          loom_value_fact_recurrence_operand_make(table, module, shrui_zero);
      induction.step =
          loom_value_fact_recurrence_operand_make(table, module, shift);
      induction.bound_flags = LOOM_LOOP_BOUND_NONE;
      induction.update = LOOM_VALUE_FACT_INDUCTION_UPDATE_SHRUI_TO_ZERO;
      return induction;
    }
    return has_additive_guard && guard_match.step == LOOM_VALUE_ID_INVALID
               ? induction
               : unknown;
  }
  if (!has_additive_guard) {
    return unknown;
  }
  const loom_value_id_t* operands = loom_op_const_operands(update);
  const loom_value_id_t lhs =
      loom_value_fact_table_query_identity(table, operands[0]);
  const loom_value_id_t rhs =
      loom_value_fact_table_query_identity(table, operands[1]);
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_condition_forwards_counter(module, table, condition, body,
                                                 lhs, counter)) {
    step = operands[1];
  } else if (!subtract_step &&
             loom_value_fact_condition_forwards_counter(
                 module, table, condition, body, rhs, counter)) {
    step = operands[0];
  }
  if (step != LOOM_VALUE_ID_INVALID &&
      loom_value_fact_loop_is_invariant(
          module, loop, loom_value_fact_table_query_identity(table, step))) {
    if (guard_match.step != LOOM_VALUE_ID_INVALID &&
        !loom_value_fact_induction_steps_equal(table, module, guard_match.step,
                                               guard_match.subtract_step, step,
                                               subtract_step)) {
      return unknown;
    }
    induction.step =
        loom_value_fact_recurrence_operand_make(table, module, step);
    if (subtract_step) {
      induction.flags |= LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP;
    }
    if (guard_match.step != LOOM_VALUE_ID_INVALID) {
      induction.flags |= LOOM_VALUE_FACT_INDUCTION_GUARD_USES_NEXT;
    }
  } else if (guard_match.step != LOOM_VALUE_ID_INVALID) {
    return unknown;
  }
  return induction;
}

typedef struct loom_value_fact_loop_scope_t {
  // Module defining the queried values.
  const loom_module_t* module;
  // Structured loop whose captures are invariant.
  loom_loop_like_t loop;
} loom_value_fact_loop_scope_t;

static bool loom_value_fact_loop_scope_is_invariant(void* user_data,
                                                    loom_value_id_t value) {
  const loom_value_fact_loop_scope_t* scope = user_data;
  return loom_value_fact_loop_is_invariant(scope->module, scope->loop, value);
}

iree_status_t loom_value_fact_loop_build_recurrences(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, iree_arena_allocator_t* arena,
    loom_value_fact_recurrence_set_t* out_set) {
  *out_set = (loom_value_fact_recurrence_set_t){0};
  const loom_block_t* body =
      loom_region_const_entry_block(loom_loop_like_body(loop));
  const loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const loom_block_t* header =
      condition_region ? loom_region_const_entry_block(condition_region) : body;
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop);
  loom_op_t* yield = body->last_op;
  loom_op_t* condition = condition_region ? header->last_op : NULL;
  // A rewriter can publish a loop shell before completing its terminators.
  if (!yield || yield->operand_count != initial.count ||
      (condition_region &&
       (!condition || condition->operand_count != body->arg_count + 1))) {
    return iree_ok_status();
  }
  loom_value_fact_loop_scope_t owner = {.module = module, .loop = loop};
  const loom_value_fact_recurrence_scope_t scope = {
      .user_data = &owner,
      .is_invariant = loom_value_fact_loop_scope_is_invariant,
      .forwarding_block = condition ? body : NULL,
      .forwarding_values =
          condition ? (loom_value_slice_t){loom_op_operands(condition) + 1,
                                           body->arg_count}
                    : (loom_value_slice_t){0},
  };
  const uint16_t argument_offset =
      loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
          ? 0
          : (uint16_t)loop.vtable->iv_block_arg_index + 1;
  return loom_value_fact_recurrence_set_build(
      table, module, header, argument_offset, initial,
      (loom_value_slice_t){loom_op_operands(yield), initial.count}, &scope,
      arena, out_set);
}

typedef struct loom_value_fact_cfg_loop_scope_t {
  // Module defining the queried values.
  const loom_module_t* module;
  // Natural-loop membership established by the CFG owner.
  const loom_cfg_loop_nest_t* loops;
  // Loop whose captures are invariant.
  uint16_t loop_index;
} loom_value_fact_cfg_loop_scope_t;

static bool loom_value_fact_cfg_loop_scope_is_invariant(void* user_data,
                                                        loom_value_id_t value) {
  const loom_value_fact_cfg_loop_scope_t* scope = user_data;
  return loom_value_fact_induction_is_invariant(scope->module, scope->loops,
                                                scope->loop_index, value);
}

iree_status_t loom_value_fact_cfg_build_recurrences(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index,
    iree_arena_allocator_t* arena) {
  const uint16_t loop_index =
      loom_cfg_loop_nest_innermost(&region->loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      region->loops.loops[loop_index].header_index != block_index) {
    return iree_ok_status();
  }
  loom_value_fact_recurrence_set_t* set = &region->recurrences[loop_index];
  *set = (loom_value_fact_recurrence_set_t){0};
  region->inductions[loop_index] = loom_value_fact_induction_recognize(
      table, module, &region->loops, loop_index);
  if (region->inductions[loop_index].value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  const loom_cfg_natural_loop_t* loop = &region->loops.loops[loop_index];
  const loom_cfg_edge_info_t* entry =
      &region->graph.edges[loop->entries.unique_index];
  const loom_cfg_edge_info_t* backedge =
      &region->graph.edges[loop->backedges.unique_index];
  loom_value_fact_cfg_loop_scope_t owner = {
      .module = module,
      .loops = &region->loops,
      .loop_index = loop_index,
  };
  const loom_value_fact_recurrence_scope_t scope = {
      .user_data = &owner,
      .is_invariant = loom_value_fact_cfg_loop_scope_is_invariant,
  };
  return loom_value_fact_recurrence_set_build(
      table, module, region->graph.blocks[block_index].block, 0,
      loom_cfg_br_args(
          region->graph.blocks[entry->source_block_index].block->last_op),
      loom_cfg_br_args(
          region->graph.blocks[backedge->source_block_index].block->last_op),
      &scope, arena, set);
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

loom_value_facts_t loom_value_fact_induction_step_facts(
    const loom_value_fact_table_t* table,
    const loom_value_fact_induction_t* induction) {
  loom_value_facts_t step =
      loom_value_fact_recurrence_operand_facts(table, induction->step);
  if (iree_any_bit_set(induction->flags,
                       LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP)) {
    // Exact deltas describe raw carrier addition. The minimum i64 value is
    // its own modular negation; negating a range instead requires a complete
    // mathematical interval and uses the ordinary overflow semantics.
    if (loom_value_facts_is_exact(step)) {
      step = loom_value_facts_exact_i64(
          step.range_lo == INT64_MIN ? INT64_MIN : -step.range_lo);
    } else {
      loom_value_facts_negi(&step, &step);
    }
  }
  return step;
}

loom_loop_recurrence_facts_t loom_value_fact_induction_facts(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_induction_t* induction) {
  const loom_loop_recurrence_facts_t unknown = {
      .values = loom_value_facts_unknown(),
      .body_values = loom_value_facts_unknown(),
      .exit_value = loom_value_facts_unknown(),
  };
  if (iree_any_bit_set(induction->flags,
                       LOOM_VALUE_FACT_INDUCTION_EXITS_AT_HEADER)) {
    return (loom_loop_recurrence_facts_t){
        .values = loom_value_facts_unknown(),
        .body_values = loom_value_facts_unknown(),
        .exit_value = loom_value_facts_unknown(),
        .trip_count_kind = LOOM_LOOP_TRIP_COUNT_EXACT,
    };
  }
  if (induction->value == LOOM_VALUE_ID_INVALID) {
    return unknown;
  }
  const loom_value_facts_t initial =
      loom_value_fact_recurrence_operand_facts(table, induction->initial_value);
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(module, induction->value));
  uint8_t bitwidth = (uint8_t)loom_scalar_type_bitwidth(scalar_type);
  const loom_value_t* induction_value =
      loom_module_value(module, induction->value);
  const loom_target_facts_t* target_facts =
      loom_value_fact_table_block_target_facts(
          table, loom_value_def_block(induction_value));
  if (target_facts) {
    const loom_target_snapshot_t* target = &target_facts->storage.snapshot;
    if (scalar_type == LOOM_SCALAR_TYPE_INDEX) {
      bitwidth = target->index_bitwidth;
    } else if (scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
      bitwidth = target->offset_bitwidth;
    }
  }
  loom_loop_recurrence_facts_t result;
  if (induction->update == LOOM_VALUE_FACT_INDUCTION_UPDATE_SHRUI_TO_ZERO) {
    const loom_value_facts_t shift =
        loom_value_fact_recurrence_operand_facts(table, induction->step);
    result = loom_loop_domain_shrui_to_zero_recurrence_facts(bitwidth, initial,
                                                             shift);
  } else {
    const loom_value_facts_t bound =
        loom_value_fact_recurrence_operand_facts(table, induction->bound);
    const loom_value_facts_t step =
        loom_value_fact_induction_step_facts(table, induction);
    result = iree_any_bit_set(induction->flags,
                              LOOM_VALUE_FACT_INDUCTION_GUARD_USES_NEXT)
                 ? loom_loop_domain_next_guard_recurrence_facts(
                       induction->bound_flags, bitwidth, initial, bound, step)
                 : loom_loop_domain_recurrence_facts(
                       induction->bound_flags, bitwidth, initial, bound, step);
  }
  if (iree_any_bit_set(induction->flags,
                       LOOM_VALUE_FACT_INDUCTION_MAY_EXIT_EARLY) &&
      result.trip_count_kind == LOOM_LOOP_TRIP_COUNT_EXACT &&
      result.trip_count != 0) {
    result.trip_count_kind = LOOM_LOOP_TRIP_COUNT_BOUNDED;
    result.exit_value = result.values;
  }
  return result;
}
