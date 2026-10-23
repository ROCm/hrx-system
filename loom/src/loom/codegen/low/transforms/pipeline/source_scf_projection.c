// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_scf_projection.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/canonicalize.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

typedef struct loom_low_source_scf_projection_state_t {
  // Scratch storage retained for this source-to-Low invocation.
  iree_arena_allocator_t* arena;
  // Module containing the function being transformed.
  loom_module_t* module;
  // Rewriter maintaining value uses during projection.
  loom_rewriter_t* rewriter;
  // Selected source-to-Low control representation.
  loom_low_control_flow_lowering_t control_flow_lowering;
  // Projection counts returned to the owning source-to-Low pass.
  loom_low_source_scf_projection_result_t* result;
} loom_low_source_scf_projection_state_t;

static loom_op_t* loom_low_source_scf_switch_yield(loom_region_t* region) {
  loom_op_t* yield = loom_region_entry_block(region)->last_op;
  IREE_ASSERT(loom_scf_yield_isa(yield));
  return yield;
}

static iree_status_t loom_low_source_scf_switch_inline_default(
    loom_low_source_scf_projection_state_t* state, loom_op_t* op) {
  loom_region_t* default_region = loom_scf_switch_default_region(op);
  loom_op_t* default_yield = loom_low_source_scf_switch_yield(default_region);
  const loom_value_slice_t replacements = loom_scf_yield_values(default_yield);
  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      state->rewriter, default_region, default_yield, op));
  return loom_scf_replace_results_and_erase(
      op, state->rewriter, replacements.values, replacements.count);
}

static iree_status_t loom_low_source_scf_project_switch(
    loom_low_source_scf_projection_state_t* state, loom_op_t* op) {
  const loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count == 0) {
    IREE_RETURN_IF_ERROR(loom_low_source_scf_switch_inline_default(state, op));
    ++state->result->switches_projected;
    return iree_ok_status();
  }

  loom_type_t* result_types = NULL;
  if (op->result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, op->result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_types[i] = loom_module_value_type(state->module, results[i]);
    }
  }
  loom_op_t** if_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, case_regions.count, sizeof(*if_ops), (void**)&if_ops));

  loom_builder_t* builder = &state->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_builder_ip_t root_ip = loom_builder_save(builder);
  const loom_value_id_t result_checkpoint =
      loom_rewriter_value_checkpoint(state->rewriter);
  const loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  const loom_type_t selector_type =
      loom_module_value_type(state->module, loom_scf_switch_selector(op));

  for (uint16_t i = 0; i < case_regions.count; ++i) {
    loom_op_t* key_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(case_keys.i64_array[i]), selector_type,
        op->location, &key_op));
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, loom_scf_switch_selector(op),
        loom_index_constant_result(key_op), op->location, &comparison_op));
    IREE_RETURN_IF_ERROR(loom_scf_if_build(
        builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
        loom_index_cmp_result(comparison_op), result_types, op->result_count,
        /*tied_results=*/NULL, /*tied_result_count=*/0, op->location,
        &if_ops[i]));

    loom_region_t* then_region = loom_scf_if_then_region(if_ops[i]);
    const loom_builder_ip_t parent_ip =
        loom_builder_enter_region(builder, if_ops[i], then_region);
    loom_op_t* source_yield =
        loom_low_source_scf_switch_yield(case_regions.regions[i]);
    const loom_value_slice_t yielded = loom_scf_yield_values(source_yield);
    loom_op_t* target_yield = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, yielded.values, yielded.count,
                             source_yield->location, &target_yield));
    loom_builder_restore(builder, parent_ip);
    IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
        state->rewriter, case_regions.regions[i], source_yield, target_yield));

    loom_builder_enter_region(builder, if_ops[i],
                              loom_scf_if_else_region(if_ops[i]));
  }

  loom_region_t* default_region = loom_scf_switch_default_region(op);
  loom_op_t* default_yield = loom_low_source_scf_switch_yield(default_region);
  const loom_value_slice_t default_values =
      loom_scf_yield_values(default_yield);
  loom_op_t* projected_default_yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, default_values.values, default_values.count,
                           default_yield->location, &projected_default_yield));
  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      state->rewriter, default_region, default_yield, projected_default_yield));

  for (uint16_t i = case_regions.count - 1; i > 0; --i) {
    loom_region_t* outer_else = loom_scf_if_else_region(if_ops[i - 1]);
    loom_builder_enter_region(builder, if_ops[i - 1], outer_else);
    loom_op_t* yield = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, loom_op_const_results(if_ops[i]),
                             if_ops[i]->result_count, op->location, &yield));
  }
  loom_builder_restore(builder, root_ip);

  const loom_value_slice_t replacements = loom_scf_if_results(if_ops[0]);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, op, replacements.values, replacements.count,
      result_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, op, replacements.values, replacements.count));
  ++state->result->switches_projected;
  return iree_ok_status();
}

static iree_status_t loom_low_source_scf_project_lookup(
    loom_low_source_scf_projection_state_t* state, loom_op_t* op) {
  const loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  const loom_value_slice_t table = loom_scf_lookup_values(op);
  const uint16_t column_count = op->result_count;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, column_count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  memcpy(replacements,
         table.values + (iree_host_size_t)case_keys.count * column_count,
         (iree_host_size_t)column_count * sizeof(*replacements));

  loom_builder_t* builder = &state->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t result_checkpoint =
      loom_rewriter_value_checkpoint(state->rewriter);
  const loom_type_t selector_type =
      loom_module_value_type(state->module, loom_scf_lookup_selector(op));
  for (uint16_t row = case_keys.count; row > 0; --row) {
    const uint16_t case_index = row - 1;
    loom_op_t* key_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(case_keys.i64_array[case_index]), selector_type,
        op->location, &key_op));
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, loom_scf_lookup_selector(op),
        loom_index_constant_result(key_op), op->location, &comparison_op));
    for (uint16_t column = 0; column < column_count; ++column) {
      const loom_value_id_t case_value =
          table.values[(iree_host_size_t)case_index * column_count + column];
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          builder, loom_index_cmp_result(comparison_op), case_value,
          replacements[column],
          loom_module_value_type(state->module,
                                 loom_op_const_results(op)[column]),
          op->location, &select_op));
      replacements[column] = loom_scf_select_result(select_op);
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, op, replacements, column_count, result_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, op, replacements, column_count));
  ++state->result->lookups_projected;
  return iree_ok_status();
}

static iree_status_t loom_low_source_scf_projection_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_low_source_scf_projection_state_t* state =
      (loom_low_source_scf_projection_state_t*)user_data;
  if (loom_scf_lookup_isa(op)) {
    return loom_low_source_scf_project_lookup(state, op);
  }
  if (state->control_flow_lowering ==
          LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW &&
      loom_scf_switch_isa(op)) {
    return loom_low_source_scf_project_switch(state, op);
  }
  return iree_ok_status();
}

iree_status_t loom_low_source_scf_projection_run(
    loom_module_t* module, loom_func_like_t function,
    loom_low_control_flow_lowering_t control_flow_lowering,
    iree_arena_allocator_t* arena,
    loom_low_source_scf_projection_result_t* out_result) {
  *out_result = (loom_low_source_scf_projection_result_t){0};
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, arena);
  loom_low_source_scf_projection_state_t state = {
      .arena = arena,
      .module = module,
      .rewriter = &rewriter,
      .control_flow_lowering = control_flow_lowering,
      .result = out_result,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_status_t status = loom_walk_region(
      module, body, LOOM_WALK_POST_ORDER,
      (loom_walk_callback_t){.fn = loom_low_source_scf_projection_visit,
                             .user_data = &state},
      &walk_result);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
