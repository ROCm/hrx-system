// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/loop/entry.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"

bool loom_loop_entry_can_guard(const loom_module_t* module,
                               loom_loop_like_t loop) {
  if (!loom_op_regions_have_read_effects(loop.op)) {
    return false;
  }
  if (loom_scf_for_isa(loop.op)) {
    return true;
  }
  if (!loom_scf_while_isa(loop.op)) {
    return false;
  }
  const loom_block_t* header =
      loom_region_entry_block(loom_scf_while_before(loop.op));
  for (const loom_op_t* op = header->first_op; op != header->last_op;
       op = op->next_op) {
    if (op->region_count || op->tied_result_count ||
        !loom_motion_op_can_relocate_effect_free(module, op)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_loop_entry_build_condition(
    loom_rewriter_t* rewriter, loom_loop_like_t loop, loom_ir_remap_t* remap,
    loom_value_id_t* out_condition, loom_value_id_t* out_empty_results) {
  loom_builder_t* builder = &rewriter->builder;
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop);
  if (loom_scf_for_isa(loop.op)) {
    const loom_value_id_t lower = loom_loop_like_lower_bound(loop);
    const loom_scalar_type_t type =
        loom_type_element_type(loom_module_value_type(rewriter->module, lower));
    loom_op_t* comparison = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder,
        type == LOOM_SCALAR_TYPE_OFFSET ? LOOM_INDEX_CMP_PREDICATE_ULT
                                        : LOOM_INDEX_CMP_PREDICATE_SLT,
        lower, loom_loop_like_upper_bound(loop), loop.op->location,
        &comparison));
    *out_condition = loom_index_cmp_result(comparison);
    for (uint16_t i = 0; i < loop.op->result_count; ++i) {
      out_empty_results[i] = initial.values[i];
    }
    return iree_ok_status();
  }

  const loom_block_t* header =
      loom_region_entry_block(loom_scf_while_before(loop.op));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(remap, header->arg_ids,
                                                initial.values, initial.count));
  const loom_ir_clone_block_options_t options = {.omit_terminators = true};
  IREE_RETURN_IF_ERROR(
      loom_ir_clone_block_ops(builder, header, remap, &options));
  IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
      remap, loom_scf_condition_condition(header->last_op), out_condition));
  const loom_value_slice_t forwarded =
      loom_scf_condition_forwarded(header->last_op);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < forwarded.count && iree_status_is_ok(status); ++i) {
    status = loom_ir_remap_resolve_value(remap, forwarded.values[i],
                                         &out_empty_results[i]);
  }
  return status;
}

static iree_status_t loom_loop_entry_hoist_impl(
    loom_rewriter_t* rewriter, loom_loop_like_t loop,
    const loom_motion_loop_entry_plan_t* plan) {
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, loop.op);
  loom_ir_remap_t remap;
  const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      rewriter->module, rewriter->module, rewriter->arena, &options, &remap));
  loom_value_id_t* empty_results = NULL;
  loom_value_id_t* guard_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, loop.op->result_count, sizeof(*empty_results),
      (void**)&empty_results));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, loop.op->result_count, sizeof(*guard_results),
      (void**)&guard_results));
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_loop_entry_build_condition(
      rewriter, loop, &remap, &condition, empty_results));

  // Reserve the complete tuple before remapping dependent result types. The
  // guard's types must name its own extent/layout results, not the inner loop.
  IREE_RETURN_IF_ERROR(loom_builder_reserve_values(
      builder, loop.op->result_count, guard_results));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&remap, loom_op_const_results(loop.op),
                               guard_results, loop.op->result_count));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_value_types(&remap, loom_op_const_results(loop.op),
                                loop.op->result_count, &result_types));
  loom_op_t* guard = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition, result_types,
      loop.op->result_count, NULL, 0, loop.op->location, &guard));
  // Redirect external users before adding the guard's yields. The inner
  // loop keeps its own recurring type scheme and result identities.
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < loop.op->result_count && iree_status_is_ok(status);
       ++i) {
    status = loom_rewriter_replace_all_uses_except(
        rewriter, loom_op_const_results(loop.op)[i], guard_results[i], loop.op);
    if (iree_status_is_ok(status)) {
      status = loom_module_copy_value_name(rewriter->module,
                                           loom_op_const_results(loop.op)[i],
                                           guard_results[i]);
    }
  }
  loom_region_t* then_region = loom_scf_if_then_region(guard);
  loom_block_t* then_block = loom_region_entry_block(then_region);
  for (uint32_t i = 0; i < plan->count && iree_status_is_ok(status); ++i) {
    status = loom_rewriter_move_to_block_end(rewriter, plan->operations[i],
                                             then_block, guard);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_move_to_block_end(rewriter, loop.op, then_block, guard));
  loom_builder_enter_region(builder, guard, then_region);
  loom_op_t* yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, loom_op_const_results(loop.op),
                           loop.op->result_count, loop.op->location, &yield));
  loom_builder_enter_region(builder, guard, loom_scf_if_else_region(guard));
  return loom_scf_yield_build(builder, empty_results, loop.op->result_count,
                              loop.op->location, &yield);
}

iree_status_t loom_loop_entry_hoist(loom_rewriter_t* rewriter,
                                    loom_loop_like_t loop,
                                    const loom_motion_loop_entry_plan_t* plan) {
  const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  iree_status_t status = loom_loop_entry_hoist_impl(rewriter, loop, plan);
  loom_builder_restore(&rewriter->builder, saved_ip);
  return status;
}
