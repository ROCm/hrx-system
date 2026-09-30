// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_loop.h"

#include <string.h>

#include "loom/analysis/condition_edge_projection.h"
#include "loom/analysis/loop_domain.h"
#include "loom/analysis/scc.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_induction.h"

#define LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS 8

struct loom_value_fact_condition_scratch_t {
  // Module bound to the reusable query state.
  const loom_module_t* module;
  // Reusable complete condition query state.
  loom_condition_query_t query;
};

static loom_op_t* loom_value_fact_region_terminator(loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return NULL;
  }
  loom_block_t* block = loom_region_entry_block(region);
  return block && block->op_count > 0 ? block->last_op : NULL;
}

static int64_t loom_value_fact_loop_iv_base_divisor(loom_value_facts_t facts) {
  if (!loom_value_facts_is_exact(facts) || facts.range_lo == INT64_MIN) {
    return facts.known_divisor;
  }
  // Preserve gcd(0, step) so a zero lower bound keeps step divisibility.
  return facts.range_lo >= 0 ? facts.range_lo : -facts.range_lo;
}

static bool loom_value_fact_counted_loop_trip_count(int64_t lower,
                                                    int64_t upper, int64_t step,
                                                    uint64_t* out_count) {
  if (step <= 0) {
    return false;
  }
  if (lower >= upper) {
    *out_count = 0;
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper, lower, &span)) {
    return false;
  }
  *out_count = ((uint64_t)span + (uint64_t)step - 1) / (uint64_t)step;
  return true;
}

static bool loom_value_fact_counted_loop_last_reachable_iv(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step, int64_t* out_hi) {
  if (!loom_value_facts_is_exact(step) || step.range_lo <= 0 ||
      lower_bound.range_lo >= upper_bound.range_hi) {
    return false;
  }

  if (loom_value_facts_is_exact(upper_bound)) {
    uint64_t lower_min_trip_count = 0;
    uint64_t lower_max_trip_count = 0;
    if (loom_value_fact_counted_loop_trip_count(
            lower_bound.range_lo, upper_bound.range_lo, step.range_lo,
            &lower_min_trip_count) &&
        loom_value_fact_counted_loop_trip_count(
            lower_bound.range_hi, upper_bound.range_hi, step.range_hi,
            &lower_max_trip_count) &&
        lower_min_trip_count == lower_max_trip_count &&
        lower_max_trip_count > 0 && lower_max_trip_count <= INT64_MAX) {
      int64_t stepped_offset = 0;
      if (iree_checked_mul_i64((int64_t)(lower_max_trip_count - 1),
                               step.range_lo, &stepped_offset) &&
          iree_checked_add_i64(lower_bound.range_hi, stepped_offset, out_hi)) {
        return true;
      }
    }
  }

  if (!loom_value_facts_is_exact(lower_bound)) {
    return false;
  }

  int64_t last_possible_offset = 0;
  if (!iree_checked_sub_i64(upper_bound.range_hi, lower_bound.range_lo,
                            &last_possible_offset) ||
      !iree_checked_sub_i64(last_possible_offset, 1, &last_possible_offset)) {
    return false;
  }

  int64_t stepped_offset = 0;
  const int64_t trip_index = last_possible_offset / step.range_lo;
  if (!iree_checked_mul_i64(trip_index, step.range_lo, &stepped_offset) ||
      !iree_checked_add_i64(lower_bound.range_lo, stepped_offset, out_hi)) {
    return false;
  }
  return true;
}

static loom_value_facts_t loom_value_fact_counted_loop_iv_facts(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return loom_value_facts_unknown();
  }

  int64_t lower_divisor = loom_value_fact_loop_iv_base_divisor(lower_bound);
  int64_t divisor = iree_math_gcd_i64(lower_divisor, step.known_divisor);

  int64_t hi = INT64_MAX;
  if (loom_value_fact_counted_loop_last_reachable_iv(lower_bound, upper_bound,
                                                     step, &hi)) {
    return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
  }

  if (iree_checked_sub_i64(upper_bound.range_hi, 1, &hi)) {
    if (lower_bound.range_lo <= hi) {
      return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
    }
  }
  return loom_value_facts_unknown();
}

iree_status_t loom_value_fact_table_seed_loop_iv_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_loop_like_t loop = loom_loop_like_cast(module, parent_op);
  if (!loom_loop_like_isa(loop) || !loom_loop_like_has_counted_range(loop)) {
    return iree_ok_status();
  }
  if (loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE ||
      loop.vtable->iv_block_arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0 ||
      block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  loom_value_facts_t iv_facts =
      loom_value_fact_counted_loop_iv_facts(lower_bound, upper_bound, step);
  loom_value_facts_propagate_ternary_distribution(lower_bound, upper_bound,
                                                  step, &iv_facts);
  loom_value_id_t iv_id =
      loom_block_arg_id(block, loop.vtable->iv_block_arg_index);
  return loom_value_fact_table_define(table, iv_id, iv_facts);
}

static uint16_t loom_value_fact_loop_carried_arg_offset(loom_loop_like_t loop) {
  return loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
             ? 0
             : (uint16_t)loop.vtable->iv_block_arg_index + 1;
}

static uint16_t loom_value_fact_loop_state_count(loom_loop_like_t loop) {
  return loom_loop_like_iter_args(loop).count;
}

static iree_status_t loom_value_fact_table_allocate_fact_array(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(table->transient_arena, count,
                                   sizeof(loom_value_facts_t),
                                   (void**)out_facts);
}

static iree_status_t loom_value_fact_table_initialize_loop_state(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, loom_value_facts_t** out_init_facts,
    loom_value_facts_t** out_current_facts, loom_type_t** out_types) {
  uint16_t count = loom_value_fact_loop_state_count(loop);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, out_init_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, count, out_current_facts));
  *out_types = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, count, sizeof(loom_type_t), (void**)out_types));
  }

  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t value_id = iter_args.values[i];
    (*out_init_facts)[i] = loom_value_fact_table_lookup(table, value_id);
    (*out_current_facts)[i] = (*out_init_facts)[i];
    (*out_types)[i] = value_id < module->values.count
                          ? loom_module_value_type(module, value_id)
                          : loom_type_none();
  }
  return iree_ok_status();
}

// Lack of change in one slot is not convergence: a long carried queue can
// delay another slot's change past the solve budget. Unknown state bounds all
// iterations and lets the final body evaluation publish sound derived facts.
static void loom_value_fact_loop_forget_state(const loom_type_t* types,
                                              uint16_t count,
                                              loom_value_facts_t* facts) {
  for (uint16_t i = 0; i < count; ++i) {
    facts[i] = loom_value_facts_unknown();
    int64_t range_lo = 0;
    int64_t range_hi = 0;
    if (loom_type_is_scalar(types[i]) &&
        loom_value_facts_scalar_type_domain(loom_type_element_type(types[i]),
                                            &range_lo, &range_hi)) {
      facts[i] = loom_value_facts_make(range_lo, range_hi, 1);
    }
  }
}

// Direct state forwarding is solved in dependency order. Closed rotations
// contain only their initial values; computed producers remain outer feedback.
typedef struct loom_value_fact_loop_equations_t {
  // State-slot order, or NULL for a trivial or not-yet-built loop body.
  uint16_t* order;
  // Earlier solved source slot, or UINT16_MAX to consume the evaluated yield.
  uint16_t* sources;
  // Counter constraint applied before testing convergence, independently of
  // widening the other state slots.
  struct {
    // Controlling before argument's state slot, or UINT16_MAX when
    // unrecognized.
    uint16_t index;
    // Proven header range, including the false-guard observation.
    loom_value_facts_t range;
  } counter;
} loom_value_fact_loop_equations_t;

static uint16_t loom_value_fact_loop_forwarded_argument(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_block_t* block, uint16_t argument_offset, uint16_t count) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value) || loom_value_def_block(value) != block) {
    return UINT16_MAX;
  }
  const uint16_t index = loom_value_def_index(value);
  return index >= argument_offset && index - argument_offset < count
             ? index - argument_offset
             : UINT16_MAX;
}

static iree_status_t loom_value_fact_loop_visit_forwarded_argument(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const uint16_t* sources = user_data;
  return sources[node] != UINT16_MAX
             ? successor.fn(successor.user_data, sources[node])
             : iree_ok_status();
}

static iree_status_t loom_value_fact_table_initialize_loop_equations(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, const loom_type_t* types,
    loom_value_facts_t* current_facts,
    loom_value_fact_loop_equations_t* out_forwarding) {
  *out_forwarding = (loom_value_fact_loop_equations_t){
      .counter = {.index = UINT16_MAX},
  };
  const uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count < 2) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  if (!yield || yield->operand_count < count) {
    return iree_ok_status();
  }
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const loom_op_t* condition =
      loom_value_fact_region_terminator(condition_region);
  if (condition_region &&
      (!condition || condition->operand_count < loop.op->result_count + 1)) {
    return iree_ok_status();
  }
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const uint16_t body_state_count = loop.op->result_count;
  const uint16_t argument_offset =
      loom_value_fact_loop_carried_arg_offset(loop);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->transient_arena, count, 2 * sizeof(uint16_t),
      (void**)&out_forwarding->order));
  out_forwarding->sources = out_forwarding->order + count;
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t source = loom_value_fact_loop_forwarded_argument(
        module, loom_op_const_operands(yield)[i], body_block, argument_offset,
        body_state_count);
    if (source != UINT16_MAX && condition) {
      source = loom_value_fact_loop_forwarded_argument(
          module, loom_op_const_operands(condition)[1 + source],
          loom_region_const_entry_block(condition_region), 0, count);
    }
    out_forwarding->sources[i] = source;
  }
  const loom_scc_graph_t graph = {
      .node_count = count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_value_fact_loop_visit_forwarded_argument,
          out_forwarding->sources),
  };
  loom_scc_list_t components = {0};
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, NULL, table->transient_arena, &components));
  uint16_t ordinal = 0;
  for (iree_host_size_t i = 0; i < components.count; ++i) {
    const loom_scc_t* component = &components.values[i];
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      out_forwarding->order[ordinal++] = component->nodes[j];
    }
    if (!component->is_cycle) {
      continue;
    }
    const loom_type_t type = types[component->nodes[0]];
    loom_value_facts_t facts = current_facts[component->nodes[0]];
    for (iree_host_size_t j = 1; j < component->node_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, type, table, facts, table,
          current_facts[component->nodes[j]], &facts));
    }
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      const iree_host_size_t slot = component->nodes[j];
      current_facts[slot] = facts;
      // Every cycle member has the same complete invariant. Its yielded
      // facts consume that invariant rather than a partially visited peer.
      out_forwarding->sources[slot] = UINT16_MAX;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_define_loop_entry_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, uint16_t arg_offset, const loom_value_facts_t* facts,
    uint16_t count) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* block = loom_region_entry_block(region);
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t arg_index = arg_offset + i;
    if (arg_index >= block->arg_count) {
      break;
    }
    loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
    if (arg_id >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, arg_id, facts[i]));
  }
  return iree_ok_status();
}

static void loom_value_fact_table_collect_terminator_operands(
    loom_value_fact_table_t* table, const loom_op_t* terminator,
    uint16_t operand_offset, loom_value_facts_t* facts, uint16_t count) {
  const loom_value_id_t* operands =
      terminator ? loom_op_const_operands(terminator) : NULL;
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t operand_index = operand_offset + i;
    facts[i] =
        (terminator && operand_index < terminator->operand_count)
            ? loom_value_fact_table_lookup(table, operands[operand_index])
            : loom_value_facts_unknown();
  }
}

static iree_status_t loom_value_fact_table_join_loop_backedge(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_type_t* types, const loom_value_facts_t* init_facts,
    const loom_value_facts_t* yielded_facts, const loom_value_facts_t* current,
    uint16_t count, const loom_value_fact_loop_equations_t* forwarding,
    uint32_t iteration, loom_value_facts_t* next, bool* out_changed) {
  *out_changed = false;
  for (uint16_t ordinal = 0; ordinal < count; ++ordinal) {
    const uint16_t i = forwarding->order ? forwarding->order[ordinal] : ordinal;
    const uint16_t source =
        forwarding->sources ? forwarding->sources[i] : UINT16_MAX;
    const loom_value_facts_t yielded =
        source != UINT16_MAX ? next[source] : yielded_facts[i];
    loom_value_facts_t joined = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
        table, module, types[i], table, init_facts[i], table, yielded,
        &joined));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
        table, module, types[i], table, current[i], table, joined, iteration,
        &next[i]));
    if (i == forwarding->counter.index) {
      next[i] = loom_value_facts_clamp_domain(
          next[i], forwarding->counter.range.range_lo,
          forwarding->counter.range.range_hi);
    }
    if (!loom_value_fact_table_facts_equal_for_type(
            module, types[i], table, current[i], table, next[i])) {
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_counted_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0) {
    return iree_ok_status();
  }
  uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count == 0) {
    return loom_value_fact_table_compute_region_tree(table, module, body,
                                                     loop.op);
  }

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));
  loom_value_fact_loop_equations_t forwarding;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_equations(
      table, module, loop, types, current_facts, &forwarding));

  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &yielded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &next_facts));

  uint16_t carried_arg_offset = loom_value_fact_loop_carried_arg_offset(loop);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);

    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts, count,
        &forwarding, iteration, next_facts, &changed));
    memcpy(current_facts, next_facts, count * sizeof(loom_value_facts_t));
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    loom_value_fact_loop_forget_state(types, count, current_facts);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  bool zero_trip =
      loom_loop_domain_proven_empty(lower_bound, upper_bound, step);
  bool at_least_one_trip =
      loom_loop_domain_proven_nonempty(lower_bound, upper_bound, step);

  loom_value_facts_t* result_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &result_facts));
  loom_value_facts_t trip_facts = loom_value_facts_unknown();
  loom_value_facts_propagate_ternary_distribution(lower_bound, upper_bound,
                                                  step, &trip_facts);
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  // Rewriter builders finalize the new loop before moving its body. Missing
  // terminators contribute unknown facts until the region edit is published.
  const loom_value_id_t* yielded_values =
      yield ? loom_op_const_operands(yield) : NULL;
  const loom_value_slice_t initial_values = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    if (zero_trip) {
      result_facts[i] = init_facts[i];
    } else if (at_least_one_trip) {
      result_facts[i] = yielded_facts[i];
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, types[i], table, init_facts[i], table,
          yielded_facts[i], &result_facts[i]));
    }
    // Uniform state at one iteration need not be uniform after invocations
    // exit on different iterations. Directly unchanged state is independent
    // of the trip count, as is a result proven exact by the scalar domain.
    const bool unchanged =
        yielded_values &&
        (yielded_values[i] ==
             loom_block_arg_id(body_block, carried_arg_offset + i) ||
         yielded_values[i] == initial_values.values[i]);
    if (!zero_trip && !unchanged) {
      loom_value_facts_propagate_binary_distribution(
          result_facts[i], trip_facts, &result_facts[i]);
      if (loom_value_facts_is_lane_varying(result_facts[i])) {
        loom_value_facts_mark_lane_distribution_for_type(types[i],
                                                         &result_facts[i]);
      }
    }
  }
  return loom_value_fact_table_define_results(table, module, loop.op,
                                              result_facts, count, out_changed);
}

static iree_status_t loom_value_fact_condition_scratch_get(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_fact_condition_scratch_t** out_scratch) {
  loom_value_fact_condition_scratch_t* scratch = table->scratch.condition;
  if (!scratch) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->transient_arena, sizeof(*scratch), (void**)&scratch));
    memset(scratch, 0, sizeof(*scratch));
    scratch->module = module;
    loom_condition_query_initialize(module, /*value_domain=*/NULL,
                                    table->transient_arena, &scratch->query);
    table->scratch.condition = scratch;
  }
  IREE_ASSERT_EQ(scratch->module, module,
                 "fact scope condition query must remain module-local");
  *out_scratch = scratch;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_retain_condition_body_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* condition_region, loom_region_t* body,
    const loom_op_t* condition) {
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_mutable_region_condition_projection(table,
                                                                       body);
  const bool had_integer_relations =
      projection && projection->visible_integer_relation_count != 0;
  if (!condition || !body_block ||
      condition->operand_count != body_block->arg_count + 1) {
    if (projection) {
      loom_condition_edge_projection_reset(projection);
    }
    if (had_integer_relations) {
      --table->condition_integer_projection_count;
    }
    return iree_ok_status();
  }

  if (!projection) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->transient_arena, sizeof(*projection), (void**)&projection));
    loom_condition_edge_projection_initialize(table->transient_arena,
                                              projection);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_condition_projection(
        table, body, projection));
  }
  loom_value_fact_condition_scratch_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_condition_scratch_get(table, module, &scratch));
  IREE_RETURN_IF_ERROR(loom_condition_facts_query_complete(
      &scratch->query, table, loom_op_const_operands(condition)[0],
      /*assumed_truth=*/true, &projection->source_derivation));
  IREE_RETURN_IF_ERROR(loom_condition_edge_projection_update_mapping(
      projection, module, condition_region, body_block,
      loom_op_const_operands(condition) + 1, body_block->arg_count));
  const bool has_integer_relations =
      projection->visible_integer_relation_count != 0;
  if (had_integer_relations != has_integer_relations) {
    if (has_integer_relations) {
      ++table->condition_integer_projection_count;
    } else {
      --table->condition_integer_projection_count;
    }
  }
  return iree_ok_status();
}

// The condition's tuple has separate true-edge and false-edge observations.
// Only directly forwarded counter identities receive its recurrence bounds;
// facts for computations in the before region still cover every header visit.
static void loom_value_fact_table_collect_condition_operands(
    loom_value_fact_table_t* table, const loom_op_t* condition,
    loom_value_id_t counter, loom_value_facts_t range,
    loom_value_facts_t* facts, uint16_t count) {
  loom_value_fact_table_collect_terminator_operands(
      table, condition, /*operand_offset=*/1, facts, count);
  if (counter == LOOM_VALUE_ID_INVALID) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(condition);
  for (uint16_t i = 0; i < count; ++i) {
    if (loom_value_fact_table_query_identity(table, operands[1 + i]) ==
        counter) {
      facts[i] = loom_value_facts_clamp_domain(facts[i], range.range_lo,
                                               range.range_hi);
    }
  }
}

// Returns whether one condition result is independent of the iteration that
// reaches the false edge. Header and result tuples are different domains: a
// result can directly expose any stable header slot, and the body can return
// that slot through any condition-forwarded body argument.
static bool loom_value_fact_condition_result_is_unchanged(
    const loom_module_t* module, loom_loop_like_t loop,
    const loom_op_t* condition, const loom_op_t* yield, uint16_t result_index) {
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop);
  if (!condition || condition->operand_count != loop.op->result_count + 1 ||
      !yield || yield->operand_count != initial.count ||
      result_index >= loop.op->result_count) {
    return false;
  }
  const loom_block_t* header =
      loom_region_const_entry_block(loom_loop_like_condition_region(loop));
  const loom_block_t* body =
      loom_region_const_entry_block(loom_loop_like_body(loop));
  const loom_value_id_t forwarded =
      loom_op_const_operands(condition)[1 + result_index];
  for (uint16_t header_index = 0; header_index < initial.count;
       ++header_index) {
    if (forwarded == initial.values[header_index]) {
      return true;
    }
    if (forwarded != loom_block_arg_id(header, header_index)) {
      continue;
    }
    const loom_value_id_t backedge =
        loom_op_const_operands(yield)[header_index];
    if (backedge == initial.values[header_index]) {
      return true;
    }
    const loom_value_t* backedge_value = loom_module_value(module, backedge);
    if (!backedge_value || !loom_value_is_block_arg(backedge_value) ||
        loom_value_def_block(backedge_value) != body) {
      continue;
    }
    const uint16_t body_index = loom_value_def_index(backedge_value);
    if (body_index < loop.op->result_count &&
        loom_op_const_operands(condition)[1 + body_index] == forwarded) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_value_fact_table_compute_condition_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  loom_region_t* body = loom_loop_like_body(loop);
  if (!condition_region || !body || condition_region->block_count == 0 ||
      body->block_count == 0) {
    return iree_ok_status();
  }
  const uint16_t header_count = loom_value_fact_loop_state_count(loop);
  const uint16_t result_count = loop.op->result_count;

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));

  loom_value_fact_loop_equations_t forwarding;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_equations(
      table, module, loop, types, current_facts, &forwarding));

  loom_value_facts_t* forwarded_facts = NULL;
  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, result_count, &forwarded_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, header_count, &yielded_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, header_count, &next_facts));

  loom_value_fact_induction_t induction = {.value = LOOM_VALUE_ID_INVALID};
  loom_loop_recurrence_facts_t recurrence =
      loom_value_fact_induction_facts(table, module, &induction);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        header_count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_condition_operands(
        table, condition, induction.value, recurrence.body_values,
        forwarded_facts, result_count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, result_count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, header_count);

    if (iteration == 0) {
      induction = loom_value_fact_condition_loop_induction(table, module, loop);
      recurrence = loom_value_fact_induction_facts(table, module, &induction);
      IREE_RETURN_IF_ERROR(loom_value_fact_table_set_condition_induction(
          table, condition_region, induction));
      if (induction.value != LOOM_VALUE_ID_INVALID) {
        forwarding.counter.index =
            loom_value_def_index(loom_module_value(module, induction.value));
        forwarding.counter.range = recurrence.values;
      }
    }
    if (recurrence.trip_count_known && recurrence.trip_count == 0) {
      // The before region has already run with initial state. A dead body
      // cannot contribute a backedge or change its forwarded exit values.
      converged = true;
      break;
    }
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts,
        header_count, &forwarding, iteration, next_facts, &changed));
    if (header_count != 0) {
      memcpy(current_facts, next_facts,
             header_count * sizeof(loom_value_facts_t));
    }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    loom_value_fact_loop_forget_state(types, header_count, current_facts);
    if (forwarding.counter.index != UINT16_MAX) {
      current_facts[forwarding.counter.index] = recurrence.values;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        header_count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_condition_operands(
        table, condition, induction.value, recurrence.body_values,
        forwarded_facts, result_count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, result_count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, header_count);
  }

  const loom_op_t* condition =
      loom_value_fact_region_terminator(condition_region);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_retain_condition_body_facts(
      table, module, condition_region, body, condition));
  loom_value_fact_table_collect_condition_operands(
      table, condition, induction.value, recurrence.exit_value, forwarded_facts,
      result_count);
  const loom_value_facts_t condition_facts =
      loom_value_fact_table_lookup(table, loom_loop_like_condition(loop));
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  for (uint16_t i = 0; i < result_count; ++i) {
    const bool unchanged = loom_value_fact_condition_result_is_unchanged(
        module, loop, condition, yield, i);
    if (!unchanged) {
      loom_value_facts_propagate_binary_distribution(
          forwarded_facts[i], condition_facts, &forwarded_facts[i]);
      if (loom_value_facts_is_lane_varying(forwarded_facts[i])) {
        const loom_type_t result_type =
            loom_module_value_type(module, loom_op_const_results(loop.op)[i]);
        loom_value_facts_mark_lane_distribution_for_type(result_type,
                                                         &forwarded_facts[i]);
      }
    }
  }
  return loom_value_fact_table_define_results(
      table, module, loop.op, forwarded_facts, result_count, out_changed);
}

iree_status_t loom_value_fact_table_compute_loop_like_summary(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed) {
  loom_loop_like_t loop = loom_loop_like_cast(module, op);
  if (!loom_loop_like_isa(loop)) {
    return iree_ok_status();
  }
  if (loom_loop_like_condition_region(loop)) {
    return loom_value_fact_table_compute_condition_loop_summary(
        table, module, loop, out_changed);
  }
  if (loom_loop_like_has_counted_range(loop)) {
    return loom_value_fact_table_compute_counted_loop_summary(
        table, module, loop, out_changed);
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, regions[i], op));
  }
  return iree_ok_status();
}
