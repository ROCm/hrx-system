// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/legalization.h"

#include <string.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/atomic.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/encoding/vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/value/vector_transform.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/arch/amdgpu/vector_packet_legalization.h"
#include "loom/transforms/view/target_legalization.h"

static bool loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_AMDGPU_TARGET_STABLE_ID;
}

static bool loom_amdgpu_subgroup_mask_type_covers_wavefront(
    loom_type_t mask_type, uint32_t wavefront_size) {
  if (!loom_type_is_scalar(mask_type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(mask_type);
  if (!loom_scalar_type_is_integer(scalar_type)) {
    return false;
  }
  return (uint32_t)loom_scalar_type_bitwidth(scalar_type) >= wavefront_size;
}

static uint32_t loom_amdgpu_legalizer_wavefront_size(
    const loom_target_legalization_context_t* context) {
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL) {
    return 0;
  }
  return bundle->snapshot->subgroup_size;
}

static loom_target_legalizer_action_t loom_amdgpu_defer_or_reject_final(
    const loom_target_legalization_context_t* context) {
  return context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL
             ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
             : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
}

static iree_status_t loom_amdgpu_retain_native_vector_op(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_decode(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_legalizer_descriptor_set_is_amdgpu(context->descriptor_set) &&
      loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
          context->module, context->fact_table, context->descriptor_set, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_transform(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_legalizer_descriptor_set_is_amdgpu(context->descriptor_set) &&
      loom_amdgpu_vector_transform_can_lower(context->module,
                                             context->descriptor_set, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_atomic_addf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  if (loom_attr_as_enum(loom_memory_access_atomic_kind(access)) !=
      LOOM_ATOMIC_KIND_ADDF) {
    return iree_ok_status();
  }
  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_memory_access_value(access));
  if (!loom_type_equal(value_type, loom_type_scalar(LOOM_SCALAR_TYPE_F32))) {
    return iree_ok_status();
  }

  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table,
                                       loom_memory_access_view(access)),
          &view_reference)) {
    return iree_ok_status();
  }
  const loom_amdgpu_atomic_operation_kind_t operation_kind =
      loom_view_atomic_reduce_isa(op) ? LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
                                      : LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
  if (loom_amdgpu_atomic_has_descriptor_candidate(
          context->descriptor_set, view_reference.memory_space, operation_kind,
          LOOM_ATOMIC_KIND_ADDF, value_type)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }
  if (!loom_amdgpu_atomic_has_descriptor_candidate(
          context->descriptor_set, view_reference.memory_space,
          LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG, LOOM_ATOMIC_KIND_ADDF,
          value_type)) {
    return iree_ok_status();
  }
  return loom_view_target_legalize_atomic_addf_reference(context, op,
                                                         out_result);
}

static bool loom_amdgpu_match_value_type_is_supported(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_amdgpu_build_i32_constant(
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

static iree_status_t loom_amdgpu_build_zero_mask(loom_builder_t* builder,
                                                 loom_type_t mask_type,
                                                 loom_location_id_t location,
                                                 loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(0),
                                                  mask_type, location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_build_match_any_lane_step(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t value_type,
    loom_value_id_t lane_id_i32, uint32_t source_lane_index,
    loom_value_id_t current_mask, loom_type_t mask_type,
    loom_location_id_t location, loom_value_id_t* out_next_mask) {
  *out_next_mask = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_i32_constant(
      builder, source_lane_index, location, &source_lane));

  loom_op_t* is_source_lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, lane_id_i32, source_lane,
      location, &is_source_lane_op));
  const loom_value_id_t is_source_lane =
      loom_scalar_cmpi_result(is_source_lane_op);

  loom_op_t* source_active_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_any_build(
      builder, is_source_lane, location, &source_active_op));
  const loom_value_id_t source_active =
      loom_kernel_subgroup_vote_any_result(source_active_op);

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

static iree_status_t loom_amdgpu_legalize_kernel_subgroup_match_any(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t value = loom_kernel_subgroup_match_any_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_value_id_t mask = loom_kernel_subgroup_match_any_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);
  const uint32_t wavefront_size = loom_amdgpu_legalizer_wavefront_size(context);
  if (!loom_amdgpu_match_value_type_is_supported(value_type) ||
      !loom_amdgpu_wavefront_size_is_valid(wavefront_size) ||
      !loom_amdgpu_subgroup_mask_type_covers_wavefront(mask_type,
                                                       wavefront_size)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = loom_amdgpu_defer_or_reject_final(context),
    };
    return iree_ok_status();
  }

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

  loom_value_id_t current_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_zero_mask(
      &rewriter->builder, mask_type, op->location, &current_mask));
  for (uint32_t i = 0; i < wavefront_size; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_match_any_lane_step(
        &rewriter->builder, value, value_type, lane_id_i32, i, current_mask,
        mask_type, op->location, &current_mask));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &current_mask, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &current_mask, 1));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_kernel_subgroup_match_all(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t value = loom_kernel_subgroup_match_all_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  if (!loom_amdgpu_match_value_type_is_supported(value_type)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = loom_amdgpu_defer_or_reject_final(context),
    };
    return iree_ok_status();
  }

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

static const loom_target_legalizer_rule_t kAmdgpuLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
        .legalize = loom_amdgpu_legalize_atomic_addf,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
        .legalize = loom_amdgpu_legalize_atomic_addf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_amdgpu_legalize_oversized_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_amdgpu_legalize_oversized_vector_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSFORM,
        .legalize = loom_amdgpu_legalize_vector_transform,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4F8,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DECODE,
        .legalize = loom_amdgpu_legalize_vector_decode,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY,
        .legalize = loom_amdgpu_legalize_kernel_subgroup_match_any,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL,
        .legalize = loom_amdgpu_legalize_kernel_subgroup_match_all,
    },
};

const loom_target_legalizer_provider_t
    loom_amdgpu_target_legalizer_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kAmdgpuLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kAmdgpuLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void) {
  return &loom_amdgpu_target_legalizer_provider_storage;
}
