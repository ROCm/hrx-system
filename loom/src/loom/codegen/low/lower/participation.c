// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/participation.h"

#include "loom/analysis/control_uniformity.h"
#include "loom/codegen/low/lower/context.h"

struct loom_low_lower_participation_state_t {
  // Shared control facts borrowed from the immutable source snapshot.
  loom_control_uniformity_info_t control_uniformity;
  // Participation indexed by root-region block ordinal.
  loom_low_lower_participation_t* blocks;
};

static iree_status_t loom_low_lower_participation_initialize(
    loom_low_lower_context_t* context,
    loom_low_lower_participation_state_t* state) {
  loom_control_uniformity_info_initialize(
      context->module, context->lowering.fact_table, &context->function_arena,
      &state->control_uniformity);
  loom_region_t* body = loom_func_like_body(context->source_function);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, body->block_count, sizeof(*state->blocks),
      (void**)&state->blocks));
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_low_lower_participation_t* participation =
        &state->blocks[block->region_index];
    *participation = (loom_low_lower_participation_t){0};
    if (loom_control_uniformity_prove_execution(
            &state->control_uniformity, block->first_op,
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP, NULL)) {
      participation->kind = LOOM_LOW_LOWER_PARTICIPATION_FULL;
      continue;
    }
    loom_condition_assumption_t condition;
    if (!loom_control_uniformity_prove_single_entry(
            &state->control_uniformity, block,
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP, &condition)) {
      continue;
    }
    loom_condition_integer_comparison_t comparison;
    const loom_value_t* selector =
        loom_module_value(context->module, condition.condition);
    if (loom_value_is_block_arg(selector) ||
        !loom_condition_integer_comparison_describe(
            context->module, loom_value_def_op(selector), &comparison)) {
      continue;
    }
    loom_symbolic_expr_context_t* expressions =
        loom_low_lower_context_symbolic_expr_context(context);
    loom_low_lower_participation_condition_t* retained = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &context->function_arena, sizeof(*retained), (void**)&retained));
    *retained = (loom_low_lower_participation_condition_t){
        .comparison = comparison,
        .assumed_truth = condition.assumed_truth,
    };
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expressions, comparison.lhs, &retained->lhs.expression));
    retained->lhs.projection = loom_symbolic_expr_lookup_projection(
        expressions, &retained->lhs.expression);
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expressions, comparison.rhs, &retained->rhs.expression));
    retained->rhs.projection = loom_symbolic_expr_lookup_projection(
        expressions, &retained->rhs.expression);
    participation->kind = LOOM_LOW_LOWER_PARTICIPATION_COMPARISON;
    participation->condition = retained;
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_source_subgroup_participation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_participation_t* out_participation) {
  loom_low_lower_participation_state_t* state =
      context->lowering.report.participation;
  if (!state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&context->function_arena,
                                             sizeof(*state), (void**)&state));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_participation_initialize(context, state));
    context->lowering.report.participation = state;
  }
  if (source_op->parent_block->parent_region ==
      loom_func_like_body(context->source_function)) {
    *out_participation = state->blocks[source_op->parent_block->region_index];
  } else {
    *out_participation = (loom_low_lower_participation_t){
        .kind = loom_control_uniformity_prove_execution(
                    &state->control_uniformity, source_op,
                    LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP, NULL)
                    ? LOOM_LOW_LOWER_PARTICIPATION_FULL
                    : LOOM_LOW_LOWER_PARTICIPATION_UNKNOWN,
    };
  }
  return iree_ok_status();
}
