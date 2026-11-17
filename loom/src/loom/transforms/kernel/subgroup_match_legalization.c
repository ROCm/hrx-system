// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/subgroup_match_legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

bool loom_kernel_subgroup_match_any_is_supported(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0) {
    return false;
  }
  const loom_type_t mask_type = loom_module_value_type(
      context->module, loom_kernel_subgroup_match_any_mask(op));
  return bundle->snapshot->subgroup_size <=
         (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(mask_type));
}

static iree_status_t loom_kernel_match_build_i32_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_match_build_mask_constant(
    loom_builder_t* builder, uint64_t value, loom_type_t mask_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64((int64_t)value), mask_type, location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_match_build_lane_step(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t value_type,
    loom_value_id_t lane_id_i32, loom_value_id_t active_mask,
    loom_value_id_t zero_mask, uint32_t source_lane_index,
    loom_value_id_t current_mask, loom_type_t mask_type,
    loom_location_id_t location, loom_value_id_t* out_next_mask) {
  *out_next_mask = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_match_build_i32_constant(
      builder, source_lane_index, location, &source_lane));

  loom_op_t* is_source_lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, lane_id_i32, source_lane,
      location, &is_source_lane_op));
  const loom_value_id_t is_source_lane =
      loom_scalar_cmpi_result(is_source_lane_op);

  // Broadcasting from an inactive source lane is undefined. Test the source
  // bit in the active mask without introducing another subgroup vote.
  loom_value_id_t source_lane_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_match_build_mask_constant(
      builder, UINT64_C(1) << source_lane_index, mask_type, location,
      &source_lane_bit));
  loom_op_t* active_lane_bit_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(builder, active_mask,
                                              source_lane_bit, mask_type,
                                              location, &active_lane_bit_op));
  const loom_value_id_t active_lane_bit =
      loom_scalar_andi_result(active_lane_bit_op);
  loom_op_t* source_active_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_NE, active_lane_bit, zero_mask,
      location, &source_active_op));
  const loom_value_id_t source_active =
      loom_scalar_cmpi_result(source_active_op);

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        source_active, &mask_type, 1, /*tied_results=*/NULL,
                        /*tied_result_count=*/0, location, &if_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, if_op, loom_scf_if_then_region(if_op));
  loom_op_t* broadcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_build(
      builder, value, source_lane, value_type, location, &broadcast_op));
  const loom_value_id_t source_value =
      loom_kernel_subgroup_broadcast_result(broadcast_op);

  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, value,
                             source_value, location, &equal_op));
  const loom_value_id_t equal = loom_scalar_cmpi_result(equal_op);

  loom_op_t* equivalence_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_ballot_build(
      builder, equal, mask_type, location, &equivalence_mask_op));
  const loom_value_id_t equivalence_mask =
      loom_kernel_subgroup_vote_ballot_mask(equivalence_mask_op);

  loom_op_t* selected_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, is_source_lane, equivalence_mask, current_mask, mask_type,
      location, &selected_mask_op));
  const loom_value_id_t selected_mask =
      loom_scf_select_result(selected_mask_op);
  loom_op_t* then_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &selected_mask, 1,
                                            location, &then_yield_op));
  loom_builder_restore(builder, saved_ip);

  saved_ip =
      loom_builder_enter_region(builder, if_op, loom_scf_if_else_region(if_op));
  loom_op_t* else_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &current_mask, 1, location,
                                            &else_yield_op));
  loom_builder_restore(builder, saved_ip);

  *out_next_mask = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_subgroup_match_any(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_id_t value = loom_kernel_subgroup_match_any_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_value_id_t mask = loom_kernel_subgroup_match_any_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);
  const uint32_t subgroup_size =
      loom_target_legalization_context_bundle(context)->snapshot->subgroup_size;

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* lane_id_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      op->location, &lane_id_op));
  const loom_value_id_t lane_id =
      loom_kernel_subgroup_lane_id_result(lane_id_op);

  loom_op_t* lane_id_i32_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &rewriter->builder, lane_id, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), op->location, &lane_id_i32_op));
  const loom_value_id_t lane_id_i32 = loom_index_cast_result(lane_id_i32_op);

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      &rewriter->builder, mask_type, op->location, &active_mask_op));
  const loom_value_id_t active_mask =
      loom_kernel_subgroup_active_mask_mask(active_mask_op);

  loom_value_id_t current_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_match_build_mask_constant(
      &rewriter->builder, 0, mask_type, op->location, &current_mask));
  const loom_value_id_t zero_mask = current_mask;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < subgroup_size && iree_status_is_ok(status); ++i) {
    status = loom_kernel_match_build_lane_step(
        &rewriter->builder, value, value_type, lane_id_i32, active_mask,
        zero_mask, i, current_mask, mask_type, op->location, &current_mask);
  }
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &current_mask, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &current_mask, 1));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_subgroup_match_all(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_id_t value = loom_kernel_subgroup_match_all_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_value_id_t mask = loom_kernel_subgroup_match_all_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* first_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_first_build(
      &rewriter->builder, value, value_type, op->location, &first_op));
  const loom_value_id_t first_value =
      loom_kernel_subgroup_broadcast_first_result(first_op);

  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(&rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_EQ,
                             value, first_value, op->location, &equal_op));
  const loom_value_id_t equal = loom_scalar_cmpi_result(equal_op);

  loom_op_t* all_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_all_build(
      &rewriter->builder, equal, op->location, &all_op));
  const loom_value_id_t all_equal =
      loom_kernel_subgroup_vote_all_result(all_op);

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      &rewriter->builder, mask_type, op->location, &active_mask_op));
  const loom_value_id_t active_mask =
      loom_kernel_subgroup_active_mask_mask(active_mask_op);

  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, loom_attr_i64(0), mask_type, op->location, &zero_op));
  const loom_value_id_t zero_mask = loom_scalar_constant_result(zero_op);

  loom_op_t* selected_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(&rewriter->builder, all_equal,
                                             active_mask, zero_mask, mask_type,
                                             op->location, &selected_mask_op));
  const loom_value_id_t replacements[] = {
      loom_scf_select_result(selected_mask_op),
      all_equal,
  };
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements),
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}
