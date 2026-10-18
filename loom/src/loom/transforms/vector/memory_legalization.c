// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/memory_legalization.h"

#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/to_scalar.h"
#include "loom/util/fact_table.h"

static iree_status_t loom_vector_guard_uniform_mask(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  loom_rewriter_t* rewriter = context->rewriter;
  const loom_memory_access_t access =
      loom_memory_access_cast(rewriter->module, op);
  const loom_value_id_t mask = loom_memory_access_mask(access);
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (mask == LOOM_VALUE_ID_INVALID ||
      !loom_value_fact_table_query_uniform_element_origin(
          rewriter->fact_table, rewriter->module, mask, &condition)) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  const loom_value_id_t passthrough = loom_memory_access_passthrough(access);
  const uint16_t result_count = op->result_count;
  const loom_type_t result_type =
      result_count ? loom_module_value_type(rewriter->module,
                                            loom_op_const_results(op)[0])
                   : (loom_type_t){0};
  // A whole-result guard carries the vector across control flow. Targets with
  // only scalar carriers retain the existing per-element guarded lowering.
  if (result_count && !loom_target_legalization_has_source_vector_carrier(
                          context, result_type, loom_op_const_results(op)[0])) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* guard_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      builder, result_count ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION : 0,
      condition, result_count ? &result_type : NULL, result_count, NULL, 0,
      op->location, &guard_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      builder, guard_op, loom_scf_if_then_region(guard_op));
  loom_op_t* unmasked_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_memory_build_unmasked(
      builder, op, &cache_policy, &unmasked_op));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, loom_op_const_results(unmasked_op),
                           result_count, op->location, &yield_op));
  loom_builder_restore(builder, saved);

  if (result_count) {
    saved = loom_builder_enter_region(builder, guard_op,
                                      loom_scf_if_else_region(guard_op));
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(
        builder, &passthrough, result_count, op->location, &yield_op));
    loom_builder_restore(builder, saved);
  }
  const loom_value_slice_t results = loom_scf_if_results(guard_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, results.values, results.count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, results.values, results.count));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_legalize_non_dense_memory(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_guard_uniform_mask(context, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_guard_uniform_mask(context, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_atomic_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}
