// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/legalization.h"

#include <string.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/symbolic_expr_proof.h"
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
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/value/vector_transform.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/arch/amdgpu/vector_packet_legalization.h"
#include "loom/transforms/vector/to_scalar.h"
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

static bool loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE ||
      plan->role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      plan->register_count <= 1 || plan->packet_count == 0) {
    return false;
  }
  switch (plan->payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      if (plan->epilogue_strategy !=
              LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE &&
          plan->epilogue_strategy !=
              LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE &&
          plan->epilogue_strategy !=
              LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE &&
          plan->epilogue_strategy !=
              LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE) {
        return false;
      }
      break;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32:
      break;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT:
    default:
      return false;
  }
  for (uint16_t i = 0; i < plan->packet_count; ++i) {
    if (plan->packets[i].result_register_count != 1) {
      return false;
    }
  }
  return true;
}

enum {
  LOOM_AMDGPU_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS = 8,
  LOOM_AMDGPU_PACKED_FRAGMENT_EPILOGUE_LOOP_MIN_GROUP_COUNT = 4,
  LOOM_AMDGPU_PACKED_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS = 32,
};

static bool loom_amdgpu_fragment_epilogue_group_wants_physical_loop(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    iree_host_size_t group_count) {
  const iree_host_size_t register_iterations =
      plan->register_count * group_count;
  if (plan->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 ||
      plan->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16) {
    switch (plan->epilogue_strategy) {
      case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE:
        return register_iterations >=
               LOOM_AMDGPU_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS;
      case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE:
      case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE:
      case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE:
        return group_count >=
                   LOOM_AMDGPU_PACKED_FRAGMENT_EPILOGUE_LOOP_MIN_GROUP_COUNT &&
               register_iterations >=
                   LOOM_AMDGPU_PACKED_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS;
      case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE:
      default:
        return false;
    }
  }
  return register_iterations >=
         LOOM_AMDGPU_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS;
}

static bool loom_amdgpu_fragment_epilogue_strategy_is_packed_b16(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy) {
  switch (strategy) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE:
      return true;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE:
    default:
      return false;
  }
}

typedef struct loom_amdgpu_fragment_store_rectangle_t {
  // Destination view value for the fragment store.
  loom_value_id_t view;
  // Symbolic inclusive begin coordinates for the rank-2 logical footprint.
  loom_symbolic_expr_t begin[2];
  // Symbolic exclusive end coordinates for the rank-2 logical footprint.
  loom_symbolic_expr_t end[2];
} loom_amdgpu_fragment_store_rectangle_t;

static iree_status_t loom_amdgpu_fragment_store_origin_expression(
    loom_symbolic_expr_context_t* expression_context,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, uint16_t* dynamic_index_ordinal,
    loom_symbolic_expr_t* out_expression, bool* out_selected) {
  *out_selected = false;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return iree_ok_status();
  }
  const int64_t static_index = static_indices.i64_array[axis];
  if (static_index == INT64_MIN) {
    if (*dynamic_index_ordinal >= dynamic_indices.count) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expression_context, dynamic_indices.values[*dynamic_index_ordinal],
        out_expression));
    ++*dynamic_index_ordinal;
  } else {
    loom_symbolic_expr_constant(static_index, out_expression);
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_rectangle_from_op(
    loom_symbolic_expr_context_t* expression_context, const loom_op_t* op,
    loom_amdgpu_fragment_store_rectangle_t* out_rectangle, bool* out_selected) {
  *out_selected = false;
  if (!loom_vector_fragment_store_isa(op)) {
    return iree_ok_status();
  }
  loom_attribute_t static_indices =
      loom_vector_fragment_store_static_indices(op);
  loom_value_slice_t dynamic_indices = loom_vector_fragment_store_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != IREE_ARRAYSIZE(out_rectangle->begin)) {
    return iree_ok_status();
  }

  loom_value_id_t extents[2] = {
      loom_vector_fragment_store_rows(op),
      loom_vector_fragment_store_columns(op),
  };
  *out_rectangle = (loom_amdgpu_fragment_store_rectangle_t){
      .view = loom_vector_fragment_store_view(op),
  };
  uint16_t dynamic_index_ordinal = 0;
  for (uint8_t axis = 0; axis < IREE_ARRAYSIZE(out_rectangle->begin); ++axis) {
    bool selected = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_origin_expression(
        expression_context, static_indices, dynamic_indices, axis,
        &dynamic_index_ordinal, &out_rectangle->begin[axis], &selected));
    if (!selected) {
      return iree_ok_status();
    }
    loom_symbolic_expr_t extent = {0};
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(expression_context,
                                                       extents[axis], &extent));
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_add(expression_context, &out_rectangle->begin[axis],
                               &extent, &out_rectangle->end[axis]));
  }
  if (dynamic_index_ordinal != dynamic_indices.count) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_rectangles_are_disjoint(
    loom_symbolic_expr_context_t* expression_context,
    const loom_amdgpu_fragment_store_rectangle_t* left,
    const loom_amdgpu_fragment_store_rectangle_t* right, bool* out_disjoint) {
  *out_disjoint = false;
  if (left->view != right->view) {
    return iree_ok_status();
  }
  for (uint8_t axis = 0; axis < IREE_ARRAYSIZE(left->begin); ++axis) {
    loom_symbolic_proof_result_t result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
        expression_context, &left->end[axis], &right->begin[axis], &result));
    if (result == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_disjoint = true;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
        expression_context, &right->end[axis], &left->begin[axis], &result));
    if (result == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_disjoint = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_group_reserve(
    loom_target_legalization_context_t* context, loom_op_t*** group_ops,
    loom_amdgpu_fragment_store_rectangle_t** rectangles,
    iree_host_size_t current_count, iree_host_size_t* capacity,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= *capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = *capacity * 2;
  if (new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  loom_op_t** new_group_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, new_capacity,
                                                 sizeof(*new_group_ops),
                                                 (void**)&new_group_ops));
  loom_amdgpu_fragment_store_rectangle_t* new_rectangles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, new_capacity,
                                                 sizeof(*new_rectangles),
                                                 (void**)&new_rectangles));
  memcpy(new_group_ops, *group_ops, current_count * sizeof(*new_group_ops));
  memcpy(new_rectangles, *rectangles, current_count * sizeof(*new_rectangles));
  *group_ops = new_group_ops;
  *rectangles = new_rectangles;
  *capacity = new_capacity;
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_store_plans_can_share_epilogue_loop(
    const loom_amdgpu_fragment_memory_plan_t* first_plan,
    const loom_amdgpu_fragment_memory_plan_t* candidate_plan) {
  return candidate_plan->operation_kind == first_plan->operation_kind &&
         candidate_plan->role == first_plan->role &&
         candidate_plan->layout_kind == first_plan->layout_kind &&
         candidate_plan->view_rank == first_plan->view_rank &&
         candidate_plan->register_count == first_plan->register_count &&
         candidate_plan->payload_register_count ==
             first_plan->payload_register_count &&
         candidate_plan->element_byte_count == first_plan->element_byte_count &&
         candidate_plan->view_element_type == first_plan->view_element_type &&
         candidate_plan->payload_form == first_plan->payload_form &&
         candidate_plan->epilogue_strategy == first_plan->epilogue_strategy &&
         candidate_plan->narrowed_result_scale_source ==
             first_plan->narrowed_result_scale_source;
}

static iree_status_t loom_amdgpu_fragment_store_plan_can_join_group(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_amdgpu_fragment_memory_plan_t* first_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_vector_fragment_store_isa(op)) {
    return iree_ok_status();
  }
  loom_amdgpu_fragment_memory_plan_t candidate_plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_analyze_vector_fragment_memory_plan(
      context->module, context->fact_table, context->view_regions,
      loom_target_legalization_context_bundle(context), context->descriptor_set,
      loom_amdgpu_target_facts_cast(context->target_facts), context->function,
      op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, &candidate_plan, &selected));
  if (!selected ||
      !loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(
          &candidate_plan) ||
      !loom_amdgpu_fragment_store_plans_can_share_epilogue_loop(
          first_plan, &candidate_plan)) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_collect_fragment_store_epilogue_group(
    loom_target_legalization_context_t* context, loom_op_t* current_op,
    const loom_amdgpu_fragment_memory_plan_t* first_plan,
    loom_op_t*** out_group_ops, iree_host_size_t* out_group_count) {
  *out_group_ops = NULL;
  *out_group_count = 0;

  loom_op_t* local_ops[8] = {0};
  loom_amdgpu_fragment_store_rectangle_t local_rectangles[8] = {0};
  loom_op_t** group_ops = local_ops;
  loom_amdgpu_fragment_store_rectangle_t* rectangles = local_rectangles;
  iree_host_size_t group_count = 1;
  iree_host_size_t capacity = IREE_ARRAYSIZE(local_ops);
  group_ops[0] = current_op;

  bool selected = false;
  if (current_op->prev_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, current_op->prev_op, first_plan, &selected));
  }
  if (!selected && current_op->next_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, current_op->next_op, first_plan, &selected));
  }
  if (!selected) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    (*out_group_ops)[0] = current_op;
    *out_group_count = group_count;
    return iree_ok_status();
  }

  loom_symbolic_expr_context_t expression_context = {0};
  loom_symbolic_expr_context_initialize(context->module, context->fact_table,
                                        context->arena, &expression_context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
      &expression_context, current_op, &rectangles[0], &selected));
  if (!selected) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    (*out_group_ops)[0] = current_op;
    *out_group_count = group_count;
    return iree_ok_status();
  }

  for (loom_op_t* candidate = current_op->prev_op; candidate != NULL;
       candidate = candidate->prev_op) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, candidate, first_plan, &selected));
    if (!selected) {
      break;
    }

    loom_amdgpu_fragment_store_rectangle_t candidate_rectangle = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
        &expression_context, candidate, &candidate_rectangle, &selected));
    if (!selected) {
      break;
    }

    for (iree_host_size_t i = 0; i < group_count; ++i) {
      bool disjoint = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangles_are_disjoint(
          &expression_context, &rectangles[i], &candidate_rectangle,
          &disjoint));
      if (!disjoint) {
        selected = false;
        break;
      }
    }
    if (!selected) {
      break;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_group_reserve(
        context, &group_ops, &rectangles, group_count, &capacity,
        group_count + 1));
    memmove(&group_ops[1], group_ops, group_count * sizeof(*group_ops));
    memmove(&rectangles[1], rectangles, group_count * sizeof(*rectangles));
    group_ops[0] = candidate;
    rectangles[0] = candidate_rectangle;
    ++group_count;
  }

  for (loom_op_t* candidate = current_op->next_op; candidate != NULL;
       candidate = candidate->next_op) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, candidate, first_plan, &selected));
    if (!selected) {
      break;
    }

    loom_amdgpu_fragment_store_rectangle_t candidate_rectangle = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
        &expression_context, candidate, &candidate_rectangle, &selected));
    if (!selected) {
      break;
    }

    for (iree_host_size_t i = 0; i < group_count; ++i) {
      bool disjoint = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangles_are_disjoint(
          &expression_context, &rectangles[i], &candidate_rectangle,
          &disjoint));
      if (!disjoint) {
        selected = false;
        break;
      }
    }
    if (!selected) {
      break;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_group_reserve(
        context, &group_ops, &rectangles, group_count, &capacity,
        group_count + 1));
    group_ops[group_count] = candidate;
    rectangles[group_count] = candidate_rectangle;
    ++group_count;
  }

  if (group_ops == local_ops) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    memcpy(*out_group_ops, group_ops, group_count * sizeof(*group_ops));
  } else {
    *out_group_ops = group_ops;
  }
  *out_group_count = group_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_result_fragment_store_epilogue_loop(
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

  loom_amdgpu_fragment_memory_plan_t plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_analyze_vector_fragment_memory_plan(
      context->module, context->fact_table, context->view_regions,
      loom_target_legalization_context_bundle(context), context->descriptor_set,
      loom_amdgpu_target_facts_cast(context->target_facts), context->function,
      op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, &plan, &selected));
  if (!selected ||
      !loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(&plan) ||
      plan.view_rank != 2) {
    return iree_ok_status();
  }

  const loom_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan.layout_kind);
  loom_op_t** group_ops = NULL;
  iree_host_size_t group_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collect_fragment_store_epilogue_group(
      context, op, &plan, &group_ops, &group_count));
  if (!loom_amdgpu_fragment_epilogue_group_wants_physical_loop(&plan,
                                                               group_count)) {
    return iree_ok_status();
  }
  if (plan.narrowed_result_scale_source != LOOM_VALUE_ID_INVALID &&
      loom_amdgpu_fragment_epilogue_strategy_is_packed_b16(
          plan.epilogue_strategy)) {
    // The fragment memory lowerer can apply the scale while packing the BF16
    // store. Rewriting to a physical-result loop here would lose that packet
    // plan and scalarize a path the target already knows how to emit.
    return iree_ok_status();
  }
  const bool preserve_packed_path =
      loom_amdgpu_fragment_epilogue_strategy_is_packed_b16(
          plan.epilogue_strategy);
  const loom_vector_to_scalar_flags_t rewrite_flags =
      preserve_packed_path
          ? LOOM_VECTOR_TO_SCALAR_FLAG_REQUIRE_PRODUCER_LANE_PROGRAM
          : LOOM_VECTOR_TO_SCALAR_FLAG_NONE;
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_fragment_store_to_scalar_physical_result_loop_rewrite_ops(
          context->pass, context->rewriter, group_ops, group_count, layout,
          plan.register_count, rewrite_flags, &rewritten));
  if (!rewritten) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kAmdgpuLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
        .legalize = loom_amdgpu_legalize_atomic_addf,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
        .legalize = loom_amdgpu_legalize_atomic_addf,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_amdgpu_legalize_result_fragment_store_epilogue_loop,
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
        .entries = kAmdgpuLegalizerEntries,
        .entry_count = IREE_ARRAYSIZE(kAmdgpuLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void) {
  return &loom_amdgpu_target_legalizer_provider_storage;
}
