// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/mask.h"

#include <stddef.h>
#include <stdint.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

static const loom_amdgpu_compare_descriptor_candidate_t*
loom_amdgpu_find_compare_descriptor_candidate(loom_op_kind_t op_kind,
                                              uint8_t predicate) {
  for (iree_host_size_t i = 0; i < kLoomAmdgpuCompareDescriptorCandidateCount;
       ++i) {
    const loom_amdgpu_compare_descriptor_candidate_t* row =
        &kLoomAmdgpuCompareDescriptorCandidates[i];
    if (row->op_kind == op_kind && row->predicate == predicate) {
      return row;
    }
  }
  return NULL;
}

typedef struct loom_amdgpu_select_descriptors_t {
  // Descriptor row selected for register-register lane selects.
  loom_low_lower_resolved_descriptor_t register_descriptor;
  // Optional descriptor row selected when the false lane is an inline source.
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor;
  // Optional descriptor row selected when the true lane is an inline source.
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor;
  // Optional descriptor row selected when the false lane is a literal source.
  loom_low_lower_resolved_descriptor_t src0_literal_descriptor;
  // Optional descriptor row selected when the true lane is a literal source.
  loom_low_lower_resolved_descriptor_t src1_literal_descriptor;
  // Optional descriptor row selected when false is literal and true is inline.
  loom_low_lower_resolved_descriptor_t src0_literal_src1_inline_descriptor;
  // Optional descriptor row selected when true is literal and false is inline.
  loom_low_lower_resolved_descriptor_t src1_literal_src0_inline_descriptor;
} loom_amdgpu_select_descriptors_t;

static iree_status_t loom_amdgpu_resolve_optional_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  bool present = false;
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, &present);
}

static iree_status_t loom_amdgpu_select_vector_compare_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t result,
    loom_scalar_type_t payload_element_type, uint8_t predicate,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_compare_plan_t){0};
  *out_selected = false;
  const loom_amdgpu_compare_descriptor_candidate_t* candidate =
      loom_amdgpu_find_compare_descriptor_candidate(source_op->kind, predicate);
  if (candidate == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, candidate->descriptor_ref, &descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, candidate->src0_inline_descriptor_ref, &src0_inline_descriptor));
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, candidate->src1_inline_descriptor_ref, &src1_inline_descriptor));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lhs_type = loom_module_value_type(module, lhs);
  const uint32_t lhs_lane_count = loom_amdgpu_static_vector_lane_count(
      lhs_type, payload_element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  if (lhs_lane_count == 0 ||
      !loom_type_equal(loom_module_value_type(module, rhs), lhs_type) ||
      loom_amdgpu_vector_i1_lane_count(
          loom_module_value_type(module, result)) != lhs_lane_count) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_vector_compare_plan_t){
      .descriptor = descriptor,
      .src0_inline_descriptor = src0_inline_descriptor,
      .src1_inline_descriptor = src1_inline_descriptor,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .lane_count = lhs_lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_cmpi_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_vector_compare_plan(
      context, source_op, loom_vector_cmpi_lhs(source_op),
      loom_vector_cmpi_rhs(source_op), loom_vector_cmpi_result(source_op),
      LOOM_SCALAR_TYPE_I32, loom_vector_cmpi_predicate(source_op), out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_vector_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_vector_compare_plan(
      context, source_op, loom_vector_cmpf_lhs(source_op),
      loom_vector_cmpf_rhs(source_op), loom_vector_cmpf_result(source_op),
      LOOM_SCALAR_TYPE_F32, loom_vector_cmpf_predicate(source_op), out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_scalar_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_compare_plan_t){0};
  *out_selected = false;
  const loom_amdgpu_compare_descriptor_candidate_t* candidate =
      loom_amdgpu_find_compare_descriptor_candidate(
          source_op->kind, loom_scalar_cmpf_predicate(source_op));
  if (candidate == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, candidate->descriptor_ref, &descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, candidate->src0_inline_descriptor_ref, &src0_inline_descriptor));
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, candidate->src1_inline_descriptor_ref, &src1_inline_descriptor));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t lhs = loom_scalar_cmpf_lhs(source_op);
  const loom_value_id_t rhs = loom_scalar_cmpf_rhs(source_op);
  const loom_value_id_t result = loom_scalar_cmpf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, lhs)) ||
      !loom_type_equal(loom_module_value_type(module, rhs),
                       loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_i1(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_vector_compare_plan_t){
      .descriptor = descriptor,
      .src0_inline_descriptor = src0_inline_descriptor,
      .src1_inline_descriptor = src1_inline_descriptor,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .lane_count = 1,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_resolve_select_vgpr_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_select_descriptors_t* out_descriptors, bool* out_present) {
  *out_descriptors = (loom_amdgpu_select_descriptors_t){0};
  *out_present = false;

  bool register_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      &out_descriptors->register_descriptor, &register_descriptor_present));
  if (!register_descriptor_present) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_INLINE,
      &out_descriptors->src0_inline_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_INLINE,
      &out_descriptors->src1_inline_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT,
      &out_descriptors->src0_literal_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_LIT,
      &out_descriptors->src1_literal_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT_SRC1_INLINE,
      &out_descriptors->src0_literal_src1_inline_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_LIT_SRC0_INLINE,
      &out_descriptors->src1_literal_src0_inline_descriptor));

  *out_present = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_vector_storage(
    loom_type_t result_type, loom_amdgpu_vector_storage_t* out_storage,
    bool* out_allows_vector_mask, bool* out_allows_lane_immediates) {
  *out_allows_vector_mask = false;
  *out_allows_lane_immediates = false;
  if (!loom_amdgpu_type_vector_storage(result_type, out_storage)) {
    return false;
  }
  switch (out_storage->kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      *out_allows_vector_mask = true;
      *out_allows_lane_immediates = true;
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      *out_allows_vector_mask = true;
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
    default:
      return false;
  }
}

static bool loom_amdgpu_select_scalar_splat_condition(
    const loom_module_t* module, loom_value_id_t condition,
    uint32_t expected_lane_count, loom_value_id_t* out_scalar_condition) {
  *out_scalar_condition = LOOM_VALUE_ID_INVALID;
  if (condition >= module->values.count) {
    return false;
  }
  const loom_value_t* condition_value = loom_module_value(module, condition);
  if (loom_value_is_block_arg(condition_value) ||
      loom_value_def_index(condition_value) != 0) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(condition_value);
  if (defining_op == NULL || !loom_vector_splat_isa(defining_op) ||
      loom_vector_splat_result(defining_op) != condition) {
    return false;
  }
  if (loom_amdgpu_vector_i1_lane_count(
          loom_module_value_type(module, condition)) != expected_lane_count) {
    return false;
  }
  const loom_value_id_t scalar = loom_vector_splat_scalar(defining_op);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, scalar))) {
    return false;
  }
  *out_scalar_condition = scalar;
  return true;
}

iree_status_t loom_amdgpu_select_vector_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  bool allows_vector_mask = false;
  bool allows_lane_immediates = false;
  if (!loom_amdgpu_select_vector_storage(result_type, &storage,
                                         &allows_vector_mask,
                                         &allows_lane_immediates)) {
    return iree_ok_status();
  }
  const loom_value_id_t condition = loom_vector_select_condition(source_op);
  const loom_value_id_t true_value = loom_vector_select_true_value(source_op);
  const loom_value_id_t false_value = loom_vector_select_false_value(source_op);
  if (!loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_amdgpu_select_condition_kind_t condition_kind =
      LOOM_AMDGPU_SELECT_CONDITION_KIND_NONE;
  loom_value_id_t selected_condition = condition;
  uint32_t registers_per_condition_lane = 1;
  const uint32_t condition_lane_count = loom_amdgpu_vector_i1_lane_count(
      loom_module_value_type(module, condition));
  if (allows_vector_mask && condition_lane_count == storage.element_count) {
    condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK;
    registers_per_condition_lane = storage.element_register_count;
  } else if (loom_amdgpu_select_scalar_splat_condition(module, condition,
                                                       storage.element_count,
                                                       &selected_condition)) {
    condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK;
  } else {
    return iree_ok_status();
  }

  loom_amdgpu_select_descriptors_t select_descriptors = {0};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_select_vgpr_descriptors(
      context, &select_descriptors, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
      .condition_kind = condition_kind,
      .register_descriptor = select_descriptors.register_descriptor,
      .src0_inline_descriptor = select_descriptors.src0_inline_descriptor,
      .src1_inline_descriptor = select_descriptors.src1_inline_descriptor,
      .src0_literal_descriptor = select_descriptors.src0_literal_descriptor,
      .src1_literal_descriptor = select_descriptors.src1_literal_descriptor,
      .src0_literal_src1_inline_descriptor =
          select_descriptors.src0_literal_src1_inline_descriptor,
      .src1_literal_src0_inline_descriptor =
          select_descriptors.src1_literal_src0_inline_descriptor,
      .condition = selected_condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = storage.register_count,
      .registers_per_condition_lane = registers_per_condition_lane,
      .allow_lane_immediates = allows_lane_immediates,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_select(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vector_select_isa(op) ||
      !loom_amdgpu_low_legality_bundle_is_amdgpu(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t result = loom_vector_select_result(op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  bool allows_vector_mask = false;
  bool allows_lane_immediates = false;
  if (!loom_amdgpu_select_vector_storage(result_type, &storage,
                                         &allows_vector_mask,
                                         &allows_lane_immediates)) {
    return iree_ok_status();
  }
  (void)allows_lane_immediates;

  *out_handled = true;
  if (!loom_type_equal(
          loom_module_value_type(module, loom_vector_select_true_value(op)),
          result_type) ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_select_false_value(op)),
          result_type)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("select.payload_type"));
  }

  const loom_value_id_t condition = loom_vector_select_condition(op);
  const uint32_t condition_lane_count = loom_amdgpu_vector_i1_lane_count(
      loom_module_value_type(module, condition));
  loom_value_id_t unused_scalar_condition = LOOM_VALUE_ID_INVALID;
  const bool scalar_splat_condition = loom_amdgpu_select_scalar_splat_condition(
      module, condition, storage.element_count, &unused_scalar_condition);
  if (scalar_splat_condition) {
    return iree_ok_status();
  }
  if (condition_lane_count != storage.element_count) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("select.mask_shape"));
  }
  if (allows_vector_mask) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("select.packed_mask_uniform"));
}

static bool loom_amdgpu_select_scalar_storage(
    loom_type_t type, uint32_t* out_register_count,
    bool* out_allows_lane_immediates) {
  *out_register_count = 0;
  *out_allows_lane_immediates = false;
  if (loom_amdgpu_type_is_i32(type) ||
      loom_amdgpu_type_is_address_scalar(type) ||
      loom_amdgpu_type_is_f32(type)) {
    *out_register_count = 1;
    *out_allows_lane_immediates = true;
    return true;
  }
  if (loom_amdgpu_type_is_i64(type) || loom_amdgpu_type_is_f64(type)) {
    *out_register_count = 2;
    return true;
  }
  if (loom_amdgpu_type_is_i8(type) || loom_amdgpu_type_is_i16(type) ||
      loom_amdgpu_type_is_16bit_float(type)) {
    *out_register_count = 1;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_payload_storage(
    loom_type_t type, uint32_t* out_register_count,
    bool* out_allows_lane_immediates) {
  if (loom_amdgpu_select_scalar_storage(type, out_register_count,
                                        out_allows_lane_immediates)) {
    return true;
  }
  loom_amdgpu_vector_storage_t storage = {0};
  bool unused_allows_vector_mask = false;
  if (!loom_amdgpu_select_vector_storage(type, &storage,
                                         &unused_allows_vector_mask,
                                         out_allows_lane_immediates)) {
    return false;
  }
  *out_register_count = storage.register_count;
  return true;
}

static iree_status_t loom_amdgpu_resolve_compare_descriptor_if_present(
    loom_low_lower_context_t* context, loom_op_kind_t op_kind,
    uint8_t predicate, loom_low_lower_resolved_descriptor_t* out_descriptor,
    bool* out_present) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_present = false;
  const loom_amdgpu_compare_descriptor_candidate_t* candidate =
      loom_amdgpu_find_compare_descriptor_candidate(op_kind, predicate);
  if (candidate == NULL) {
    return iree_ok_status();
  }
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, candidate->descriptor_ref, out_descriptor, out_present);
}

static iree_status_t loom_amdgpu_resolve_i1_mask_select_descriptors(
    loom_low_lower_context_t* context, loom_amdgpu_vector_select_plan_t* plan,
    bool* out_present) {
  *out_present = false;
  bool all_present = true;
  bool scc_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, &plan->scc_descriptor,
      &scc_present));
  all_present = all_present && scc_present;
  bool exec_read_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      &plan->mask_exec_read_descriptor, &exec_read_present));
  all_present = all_present && exec_read_present;
  if (!all_present ||
      plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC ||
      plan->true_value == plan->false_value) {
    *out_present = all_present;
    return iree_ok_status();
  }

  bool true_constant = false;
  const bool true_is_constant = loom_amdgpu_value_as_i1_constant(
      context, plan->true_value, &true_constant);
  bool false_constant = false;
  const bool false_is_constant = loom_amdgpu_value_as_i1_constant(
      context, plan->false_value, &false_constant);

  if (true_is_constant && false_is_constant) {
    if (true_constant == false_constant) {
      *out_present = all_present;
      return iree_ok_status();
    }
    if (true_constant && !false_constant) {
      *out_present = all_present;
      return iree_ok_status();
    }
    if (!true_constant && false_constant) {
      bool xor_present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          &plan->mask_xor_descriptor, &xor_present));
      *out_present = xor_present;
      return iree_ok_status();
    }
  }

  if (plan->true_value == plan->condition ||
      (true_is_constant && true_constant)) {
    bool or_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, &plan->mask_or_descriptor,
        &or_present));
    *out_present = or_present;
    return iree_ok_status();
  }

  bool and_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, &plan->mask_and_descriptor,
      &and_present));
  all_present = all_present && and_present;
  if (!all_present || plan->false_value == plan->condition ||
      (false_is_constant && !false_constant)) {
    *out_present = all_present;
    return iree_ok_status();
  }

  if (true_is_constant && !true_constant) {
    bool xor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        &plan->mask_xor_descriptor, &xor_present));
    *out_present = xor_present;
    return iree_ok_status();
  }

  if (false_is_constant && false_constant) {
    bool xor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        &plan->mask_xor_descriptor, &xor_present));
    all_present = all_present && xor_present;
    bool or_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, &plan->mask_or_descriptor,
        &or_present));
    *out_present = all_present && or_present;
    return iree_ok_status();
  }

  bool xor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, &plan->mask_xor_descriptor,
      &xor_present));
  all_present = all_present && xor_present;
  bool or_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, &plan->mask_or_descriptor,
      &or_present));
  all_present = all_present && or_present;
  *out_present = all_present;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_scf_select_i1_mask_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_scf_select_result(source_op);
  const loom_value_id_t condition = loom_scf_select_condition(source_op);
  const loom_value_id_t true_value = loom_scf_select_true_value(source_op);
  const loom_value_id_t false_value = loom_scf_select_false_value(source_op);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, condition)) ||
      !loom_amdgpu_type_is_i1(loom_module_value_type(module, true_value)) ||
      !loom_amdgpu_type_is_i1(loom_module_value_type(module, false_value))) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  bool result_is_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &result_is_mask));
  if (!result_is_mask) {
    return iree_ok_status();
  }

  loom_type_t condition_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, condition,
                                                &condition_low_type));
  bool condition_is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1,
      &condition_is_scc));
  bool condition_is_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &condition_is_mask));
  if (!condition_is_scc && !condition_is_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_select_plan_t plan = {
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK,
      .condition_kind = condition_is_scc
                            ? LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC
                            : LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK,
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = 2,
      .registers_per_condition_lane = 1,
  };
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_i1_mask_select_descriptors(
      context, &plan, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }
  *out_plan = plan;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scf_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_scf_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (loom_amdgpu_type_is_i1(result_type)) {
    return loom_amdgpu_select_scf_select_i1_mask_plan(context, source_op,
                                                      out_plan, out_selected);
  }
  uint32_t register_count = 0;
  bool allows_lane_immediates = false;
  if (!loom_amdgpu_select_payload_storage(result_type, &register_count,
                                          &allows_lane_immediates)) {
    return iree_ok_status();
  }
  const loom_value_id_t condition = loom_scf_select_condition(source_op);
  const loom_value_id_t true_value = loom_scf_select_true_value(source_op);
  const loom_value_id_t false_value = loom_scf_select_false_value(source_op);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, condition)) ||
      !loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_type_t condition_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, condition,
                                                &condition_low_type));
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));

  bool condition_is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1,
      &condition_is_scc));
  bool result_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, register_count,
      &result_is_sgpr));
  if (condition_is_scc && result_is_sgpr) {
    loom_low_lower_resolved_descriptor_t scc_descriptor = {0};
    bool scc_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, &scc_descriptor,
        &scc_descriptor_present));
    if (!scc_descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_vector_select_plan_t){
        .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
        .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC,
        .scc_descriptor = scc_descriptor,
        .condition = condition,
        .true_value = true_value,
        .false_value = false_value,
        .result = result,
        .lane_count = register_count,
        .registers_per_condition_lane = 1,
        .allow_lane_immediates = allows_lane_immediates,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  bool condition_is_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &condition_is_mask));
  if (!condition_is_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_select_descriptors_t select_descriptors = {0};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_select_vgpr_descriptors(
      context, &select_descriptors, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
      .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK,
      .register_descriptor = select_descriptors.register_descriptor,
      .src0_inline_descriptor = select_descriptors.src0_inline_descriptor,
      .src1_inline_descriptor = select_descriptors.src1_inline_descriptor,
      .src0_literal_descriptor = select_descriptors.src0_literal_descriptor,
      .src1_literal_descriptor = select_descriptors.src1_literal_descriptor,
      .src0_literal_src1_inline_descriptor =
          select_descriptors.src0_literal_src1_inline_descriptor,
      .src1_literal_src0_inline_descriptor =
          select_descriptors.src1_literal_src0_inline_descriptor,
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = register_count,
      .registers_per_condition_lane = 1,
      .allow_lane_immediates = allows_lane_immediates,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_clampf_ordered_descriptors(
    loom_low_lower_context_t* context, loom_op_kind_t compare_op_kind,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_present) {
  *out_present = false;
  bool lower_compare_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_compare_descriptor_if_present(
      context, compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OLT,
      &out_plan->lower_compare_descriptor, &lower_compare_present));
  if (!lower_compare_present) {
    return iree_ok_status();
  }
  bool upper_compare_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_compare_descriptor_if_present(
      context, compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OGT,
      &out_plan->upper_compare_descriptor, &upper_compare_present));
  if (!upper_compare_present) {
    return iree_ok_status();
  }
  bool select_present = false;
  loom_amdgpu_select_descriptors_t select_descriptors = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_select_vgpr_descriptors(
      context, &select_descriptors, &select_present));
  out_plan->select_register_descriptor = select_descriptors.register_descriptor;
  out_plan->select_src0_inline_descriptor =
      select_descriptors.src0_inline_descriptor;
  out_plan->select_src1_inline_descriptor =
      select_descriptors.src1_inline_descriptor;
  out_plan->select_src0_literal_descriptor =
      select_descriptors.src0_literal_descriptor;
  out_plan->select_src1_literal_descriptor =
      select_descriptors.src1_literal_descriptor;
  out_plan->select_src0_literal_src1_inline_descriptor =
      select_descriptors.src0_literal_src1_inline_descriptor;
  out_plan->select_src1_literal_src0_inline_descriptor =
      select_descriptors.src1_literal_src0_inline_descriptor;
  *out_present = select_present;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_clampf_number_descriptors(
    loom_low_lower_context_t* context, loom_amdgpu_clampf_plan_t* out_plan,
    bool* out_present) {
  *out_present = false;
  bool lower_register_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32,
      &out_plan->lower_bound_register_descriptor, &lower_register_present));
  if (!lower_register_present) {
    return iree_ok_status();
  }
  bool upper_register_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32,
      &out_plan->upper_bound_register_descriptor, &upper_register_present));
  if (!upper_register_present) {
    return iree_ok_status();
  }

  bool lower_literal_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_LIT,
      &out_plan->lower_bound_literal_descriptor, &lower_literal_present));
  bool upper_literal_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_LIT,
      &out_plan->upper_bound_literal_descriptor, &upper_literal_present));
  (void)lower_literal_present;
  (void)upper_literal_present;

  *out_present = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_value_id_t result, uint32_t lane_count, loom_op_kind_t compare_op_kind,
    loom_amdgpu_clampf_mode_t mode, loom_amdgpu_clampf_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_amdgpu_clampf_plan_t){0};
  *out_selected = false;
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }

  bool descriptors_present = false;
  switch (mode) {
    case LOOM_AMDGPU_CLAMPF_MODE_ORDERED: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_clampf_ordered_descriptors(
          context, compare_op_kind, out_plan, &descriptors_present));
      break;
    }
    case LOOM_AMDGPU_CLAMPF_MODE_NUMBER: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_clampf_number_descriptors(
          context, out_plan, &descriptors_present));
      break;
    }
    case LOOM_AMDGPU_CLAMPF_MODE_NONE:
      return iree_ok_status();
  }
  if (!descriptors_present) {
    return iree_ok_status();
  }

  out_plan->mode = mode;
  out_plan->value = value;
  out_plan->lower = lower;
  out_plan->upper = upper;
  out_plan->result = result;
  out_plan->lane_count = lane_count;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_clampf_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_scalar_clampf_value(source_op);
  const loom_value_id_t lower = loom_scalar_clampf_lower(source_op);
  const loom_value_id_t upper = loom_scalar_clampf_upper(source_op);
  const loom_value_id_t result = loom_scalar_clampf_result(source_op);
  const loom_type_t value_type = loom_module_value_type(module, value);
  if (!loom_amdgpu_type_is_f32(value_type) ||
      !loom_type_equal(loom_module_value_type(module, lower), value_type) ||
      !loom_type_equal(loom_module_value_type(module, upper), value_type) ||
      !loom_type_equal(loom_module_value_type(module, result), value_type)) {
    return iree_ok_status();
  }

  loom_amdgpu_clampf_mode_t mode = LOOM_AMDGPU_CLAMPF_MODE_NONE;
  switch (loom_scalar_clampf_mode(source_op)) {
    case LOOM_SCALAR_CLAMPF_MODE_ORDERED:
      mode = LOOM_AMDGPU_CLAMPF_MODE_ORDERED;
      break;
    case LOOM_SCALAR_CLAMPF_MODE_NUMBER:
      mode = LOOM_AMDGPU_CLAMPF_MODE_NUMBER;
      break;
    case LOOM_SCALAR_CLAMPF_MODE_IEEE:
    case LOOM_SCALAR_CLAMPF_MODE_COUNT_:
      return iree_ok_status();
  }
  return loom_amdgpu_select_clampf_plan(
      context, source_op, value, lower, upper, result, /*lane_count=*/1,
      LOOM_OP_SCALAR_CMPF, mode, out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_clampf_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_vector_clampf_value(source_op);
  const loom_value_id_t lower = loom_vector_clampf_lower(source_op);
  const loom_value_id_t upper = loom_vector_clampf_upper(source_op);
  const loom_value_id_t result = loom_vector_clampf_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t lane_count = loom_amdgpu_vector_f32_lane_count(result_type);
  if (lane_count == 0 ||
      !loom_type_equal(loom_module_value_type(module, value), result_type) ||
      !loom_type_equal(loom_module_value_type(module, lower), result_type) ||
      !loom_type_equal(loom_module_value_type(module, upper), result_type)) {
    return iree_ok_status();
  }

  loom_amdgpu_clampf_mode_t mode = LOOM_AMDGPU_CLAMPF_MODE_NONE;
  switch (loom_vector_clampf_mode(source_op)) {
    case LOOM_VECTOR_CLAMPF_MODE_ORDERED:
      mode = LOOM_AMDGPU_CLAMPF_MODE_ORDERED;
      break;
    case LOOM_VECTOR_CLAMPF_MODE_NUMBER:
      mode = LOOM_AMDGPU_CLAMPF_MODE_NUMBER;
      break;
    case LOOM_VECTOR_CLAMPF_MODE_IEEE:
    case LOOM_VECTOR_CLAMPF_MODE_COUNT_:
      return iree_ok_status();
  }
  return loom_amdgpu_select_clampf_plan(context, source_op, value, lower, upper,
                                        result, lane_count, LOOM_OP_VECTOR_CMPF,
                                        mode, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_slice_lane_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t unit_offset,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, unit_offset,
                                    lane_type, out_lane);
}

static iree_status_t loom_amdgpu_slice_source_lane_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t unit_offset,
    loom_type_t fallback_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  loom_type_t source_lane_type = fallback_lane_type;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_source);
  if (loom_type_is_register(source_type) &&
      loom_low_register_type_unit_count(source_type) > 1) {
    source_lane_type = loom_low_register_type_with_unit_count(source_type, 1);
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, unit_offset,
                                    source_lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_select_lane_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_named_attr_t* attrs, iree_host_size_t attr_count,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &lane_op));
  *out_result = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  return iree_ok_status();
}

typedef enum loom_amdgpu_select_immediate_value_role_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE = 1,
} loom_amdgpu_select_immediate_value_role_t;

typedef enum loom_amdgpu_select_immediate_operand_kind_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE = 1,
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE = 2,
} loom_amdgpu_select_immediate_operand_kind_t;

typedef enum loom_amdgpu_select_immediate_attr_kind_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32 = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE = 1,
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE = 2,
} loom_amdgpu_select_immediate_attr_kind_t;

typedef struct loom_amdgpu_select_immediate_attr_t {
  // Descriptor attribute receiving the selected source bits.
  loom_amdgpu_select_immediate_attr_kind_t kind;
  // Select source lane whose exact bits populate the attribute.
  loom_amdgpu_select_immediate_value_role_t value_role;
  // True when the attribute is encoded as an inline source constrained to
  // 0..64.
  bool requires_inline_range;
} loom_amdgpu_select_immediate_attr_t;

typedef struct loom_amdgpu_select_immediate_candidate_t {
  // Byte offset to the plan descriptor row selected by this immediate form.
  iree_host_size_t descriptor_offset;
  // Operand payloads consumed by the descriptor.
  loom_amdgpu_select_immediate_operand_kind_t operands[2];
  // Number of entries in operands.
  uint8_t operand_count;
  // Attribute payloads emitted with the descriptor.
  loom_amdgpu_select_immediate_attr_t attrs[2];
  // Number of entries in attrs.
  uint8_t attr_count;
} loom_amdgpu_select_immediate_candidate_t;

static const loom_amdgpu_select_immediate_candidate_t
    kLoomAmdgpuSelectImmediateCandidates[] = {
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src0_literal_src1_inline_descriptor),
            .operands = {LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION},
            .operand_count = 1,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                    },
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 2,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src1_literal_src0_inline_descriptor),
            .operands = {LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION},
            .operand_count = 1,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                    },
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 2,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src1_inline_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src0_inline_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src0_literal_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_select_plan_t,
                                          src1_literal_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                    },
                },
            .attr_count = 1,
        },
};

static const loom_low_lower_resolved_descriptor_t*
loom_amdgpu_select_immediate_candidate_descriptor(
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate) {
  const uint8_t* plan_bytes = (const uint8_t*)plan;
  const void* descriptor_bytes = plan_bytes + candidate->descriptor_offset;
  return (const loom_low_lower_resolved_descriptor_t*)descriptor_bytes;
}

static iree_string_view_t loom_amdgpu_select_immediate_attr_name(
    loom_amdgpu_select_immediate_attr_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32:
      return IREE_SV("imm32");
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE:
      return IREE_SV("false_value");
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE:
      return IREE_SV("true_value");
  }
  return iree_string_view_empty();
}

static bool loom_amdgpu_select_immediate_bits(
    loom_amdgpu_select_immediate_value_role_t role, bool false_is_exact,
    uint32_t false_bits, bool true_is_exact, uint32_t true_bits,
    uint32_t* out_bits) {
  switch (role) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE:
      *out_bits = false_bits;
      return false_is_exact;
    case LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE:
      *out_bits = true_bits;
      return true_is_exact;
  }
  *out_bits = 0;
  return false;
}

static bool loom_amdgpu_select_immediate_candidate_matches(
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate,
    bool false_is_exact, uint32_t false_bits, bool true_is_exact,
    uint32_t true_bits) {
  const loom_low_lower_resolved_descriptor_t* descriptor =
      loom_amdgpu_select_immediate_candidate_descriptor(plan, candidate);
  if (descriptor->descriptor == NULL) {
    return false;
  }
  for (uint8_t i = 0; i < candidate->attr_count; ++i) {
    uint32_t bits = 0;
    if (!loom_amdgpu_select_immediate_bits(candidate->attrs[i].value_role,
                                           false_is_exact, false_bits,
                                           true_is_exact, true_bits, &bits)) {
      return false;
    }
    if (candidate->attrs[i].requires_inline_range && bits > 64) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_select_immediate_materialize_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_select_immediate_operand_kind_t kind,
    const loom_amdgpu_vector_select_plan_t* plan,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_operand) {
  *out_operand = LOOM_VALUE_ID_INVALID;
  switch (kind) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION:
      *out_operand = condition;
      return iree_ok_status();
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
          context, source_op, low_false_value, plan->lane_count, lane,
          lane_type, out_operand));
      return loom_amdgpu_materialize_low_vgpr_b32(context, source_op,
                                                  *out_operand, out_operand);
    }
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
          context, source_op, low_true_value, plan->lane_count, lane, lane_type,
          out_operand));
      return loom_amdgpu_materialize_low_vgpr_b32(context, source_op,
                                                  *out_operand, out_operand);
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU select immediate operand kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_vector_select_immediate_candidate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    uint32_t false_bits, uint32_t true_bits, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[2] = {0};
  for (uint8_t i = 0; i < candidate->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_immediate_materialize_operand(
        context, source_op, candidate->operands[i], plan, low_false_value,
        low_true_value, condition, lane, lane_type, &operands[i]));
  }

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  for (uint8_t i = 0; i < candidate->attr_count; ++i) {
    const loom_amdgpu_select_immediate_attr_t* attr = &candidate->attrs[i];
    const uint32_t bits =
        attr->value_role == LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE
            ? false_bits
            : true_bits;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, loom_amdgpu_select_immediate_attr_name(attr->kind), bits,
        attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  }

  return loom_amdgpu_emit_select_lane_op(
      context, source_op,
      loom_amdgpu_select_immediate_candidate_descriptor(plan, candidate),
      operands, candidate->operand_count, attrs, attr_count, lane_type,
      out_result);
}

static iree_status_t loom_amdgpu_emit_vector_select_immediate_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_result, bool* out_emitted) {
  *out_result = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;
  if (!plan->allow_lane_immediates) {
    return iree_ok_status();
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t false_bits = 0;
  const bool false_is_exact = loom_amdgpu_source_lane_as_u32_bits(
      fact_table, module, plan->false_value, lane, &false_bits);
  uint32_t true_bits = 0;
  const bool true_is_exact = loom_amdgpu_source_lane_as_u32_bits(
      fact_table, module, plan->true_value, lane, &true_bits);

  if (false_is_exact && true_is_exact && false_bits == true_bits) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, low_true_value, plan->lane_count, lane, lane_type,
        out_result));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, *out_result, out_result));
    *out_emitted = true;
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuSelectImmediateCandidates); ++i) {
    const loom_amdgpu_select_immediate_candidate_t* candidate =
        &kLoomAmdgpuSelectImmediateCandidates[i];
    if (!loom_amdgpu_select_immediate_candidate_matches(
            plan, candidate, false_is_exact, false_bits, true_is_exact,
            true_bits)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_select_immediate_candidate(
        context, source_op, plan, candidate, low_false_value, low_true_value,
        condition, lane, lane_type, false_bits, true_bits, out_result));
    *out_emitted = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_amdgpu_select_source_lanes_have_same_bits(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    const loom_amdgpu_vector_select_plan_t* plan, uint32_t lane) {
  uint32_t false_bits = 0;
  if (!loom_amdgpu_source_lane_as_u32_bits(
          fact_table, module, plan->false_value, lane, &false_bits)) {
    return false;
  }
  uint32_t true_bits = 0;
  return loom_amdgpu_source_lane_as_u32_bits(
             fact_table, module, plan->true_value, lane, &true_bits) &&
         false_bits == true_bits;
}

typedef enum loom_amdgpu_compare_immediate_value_role_e {
  LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS = 0,
  LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS = 1,
} loom_amdgpu_compare_immediate_value_role_t;

typedef struct loom_amdgpu_compare_immediate_candidate_t {
  // Byte offset to the plan descriptor row selected by this immediate form.
  iree_host_size_t descriptor_offset;
  // Compare source lane encoded as an inline attribute.
  loom_amdgpu_compare_immediate_value_role_t inline_role;
  // Compare source lane supplied as a register operand.
  loom_amdgpu_compare_immediate_value_role_t operand_role;
} loom_amdgpu_compare_immediate_candidate_t;

static const loom_amdgpu_compare_immediate_candidate_t
    kLoomAmdgpuCompareImmediateCandidates[] = {
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_compare_plan_t,
                                          src1_inline_descriptor),
            .inline_role = LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS,
            .operand_role = LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS,
        },
        {
            .descriptor_offset = offsetof(loom_amdgpu_vector_compare_plan_t,
                                          src0_inline_descriptor),
            .inline_role = LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS,
            .operand_role = LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS,
        },
};

static const loom_low_lower_resolved_descriptor_t*
loom_amdgpu_compare_immediate_candidate_descriptor(
    const loom_amdgpu_vector_compare_plan_t* plan,
    const loom_amdgpu_compare_immediate_candidate_t* candidate) {
  const uint8_t* plan_bytes = (const uint8_t*)plan;
  const void* descriptor_bytes = plan_bytes + candidate->descriptor_offset;
  return (const loom_low_lower_resolved_descriptor_t*)descriptor_bytes;
}

static iree_string_view_t loom_amdgpu_compare_immediate_attr_name(
    loom_amdgpu_compare_immediate_value_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS:
      return IREE_SV("lhs");
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS:
      return IREE_SV("rhs");
  }
  return iree_string_view_empty();
}

static loom_value_id_t loom_amdgpu_compare_immediate_source_value(
    const loom_amdgpu_vector_compare_plan_t* plan,
    loom_amdgpu_compare_immediate_value_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS:
      return plan->lhs;
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS:
      return plan->rhs;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU compare immediate source role");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_value_id_t loom_amdgpu_compare_immediate_low_value(
    loom_value_id_t low_lhs, loom_value_id_t low_rhs,
    loom_amdgpu_compare_immediate_value_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_LHS:
      return low_lhs;
    case LOOM_AMDGPU_COMPARE_IMMEDIATE_VALUE_RHS:
      return low_rhs;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU compare immediate low value role");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_amdgpu_compare_immediate_candidate_matches(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    const loom_amdgpu_vector_compare_plan_t* plan,
    const loom_amdgpu_compare_immediate_candidate_t* candidate, uint32_t lane,
    uint32_t* out_inline_bits) {
  const loom_low_lower_resolved_descriptor_t* descriptor =
      loom_amdgpu_compare_immediate_candidate_descriptor(plan, candidate);
  if (descriptor->descriptor == NULL) {
    return false;
  }
  const loom_value_id_t source_value =
      loom_amdgpu_compare_immediate_source_value(plan, candidate->inline_role);
  return loom_amdgpu_source_lane_as_u32_bits(fact_table, module, source_value,
                                             lane, out_inline_bits) &&
         *out_inline_bits <= 64;
}

static iree_status_t loom_amdgpu_emit_vector_compare_immediate_candidate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan,
    const loom_amdgpu_compare_immediate_candidate_t* candidate,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, uint32_t lane,
    loom_type_t payload_lane_type, loom_type_t mask_lane_type,
    uint32_t inline_bits, loom_value_id_t* out_result) {
  loom_value_id_t lane_operand = loom_amdgpu_compare_immediate_low_value(
      low_lhs, low_rhs, candidate->operand_role);
  IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
      context, source_op, lane_operand, plan->lane_count, lane,
      payload_lane_type, &lane_operand));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, lane_operand, &lane_operand));

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, loom_amdgpu_compare_immediate_attr_name(candidate->inline_role),
      inline_bits, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {lane_operand};
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context,
      loom_amdgpu_compare_immediate_candidate_descriptor(plan, candidate),
      operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &mask_lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &lane_op));
  *out_result = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_compare_immediate_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, uint32_t lane, loom_type_t payload_lane_type,
    loom_type_t mask_lane_type, loom_value_id_t* out_result,
    bool* out_emitted) {
  *out_result = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuCompareImmediateCandidates); ++i) {
    const loom_amdgpu_compare_immediate_candidate_t* candidate =
        &kLoomAmdgpuCompareImmediateCandidates[i];
    uint32_t inline_bits = 0;
    if (!loom_amdgpu_compare_immediate_candidate_matches(
            fact_table, module, plan, candidate, lane, &inline_bits)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_compare_immediate_candidate(
        context, source_op, plan, candidate, low_lhs, low_rhs, lane,
        payload_lane_type, mask_lane_type, inline_bits, out_result));
    *out_emitted = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->rhs, &low_rhs));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    bool emitted_immediate = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_compare_immediate_lane(
        context, source_op, plan, low_lhs, low_rhs, i, lane_type,
        mask_lane_type, &lane_results[i], &emitted_immediate));
    if (emitted_immediate) {
      continue;
    }

    loom_value_id_t lane_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_lhs, lane_count, i, lane_type, &lane_lhs));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_lhs, &lane_lhs));
    loom_value_id_t lane_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_rhs, lane_count, i, lane_type, &lane_rhs));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_rhs, &lane_rhs));
    const loom_value_id_t operands[] = {
        lane_lhs,
        lane_rhs,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_cmpi(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

iree_status_t loom_amdgpu_lower_vector_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

iree_status_t loom_amdgpu_lower_scalar_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

static iree_status_t loom_amdgpu_emit_resolved_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_resolved_vgpr_binary_literal(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, uint32_t literal, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), literal, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_clampf_compare_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_lane_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &lane_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_clampf_select_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t false_value,
    loom_value_id_t true_value, loom_value_id_t true_source,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t true_bits = 0;
  if (plan->select_src1_inline_descriptor.descriptor != NULL &&
      loom_amdgpu_source_lane_as_u32_bits(fact_table, module, true_source, lane,
                                          &true_bits) &&
      true_bits <= 64) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("true_value"), true_bits,
                                    attrs, IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {false_value, condition};
    return loom_amdgpu_emit_select_lane_op(
        context, source_op, &plan->select_src1_inline_descriptor, operands,
        IREE_ARRAYSIZE(operands), attrs, attr_count, lane_type, out_value);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, true_value, &true_value));
  const loom_value_id_t operands[] = {
      false_value,
      true_value,
      condition,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->select_register_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &lane_type,
      1, /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

typedef enum loom_amdgpu_clampf_bound_kind_e {
  LOOM_AMDGPU_CLAMPF_BOUND_LOWER = 0,
  LOOM_AMDGPU_CLAMPF_BOUND_UPPER = 1,
} loom_amdgpu_clampf_bound_kind_t;

static iree_status_t loom_amdgpu_emit_clampf_number_bound_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t value,
    loom_value_id_t low_bound, loom_value_id_t bound_source, uint32_t lane,
    uint32_t lane_count, loom_type_t lane_type,
    loom_amdgpu_clampf_bound_kind_t bound_kind, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_low_lower_resolved_descriptor_t* literal_descriptor =
      bound_kind == LOOM_AMDGPU_CLAMPF_BOUND_LOWER
          ? &plan->lower_bound_literal_descriptor
          : &plan->upper_bound_literal_descriptor;
  uint32_t bound_bits = 0;
  if (literal_descriptor->descriptor != NULL &&
      loom_amdgpu_source_lane_as_u32_bits(fact_table, module, bound_source,
                                          lane, &bound_bits)) {
    return loom_amdgpu_emit_resolved_vgpr_binary_literal(
        context, source_op, literal_descriptor, value, bound_bits, lane_type,
        out_value);
  }

  loom_value_id_t bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
      context, source_op, low_bound, lane_count, lane, lane_type, &bound));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_materialize_low_vgpr_b32(context, source_op, bound, &bound));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op,
      bound_kind == LOOM_AMDGPU_CLAMPF_BOUND_LOWER
          ? &plan->lower_bound_register_descriptor
          : &plan->upper_bound_register_descriptor,
      value, bound, lane_type, out_value);
}

static iree_status_t loom_amdgpu_lower_clampf_ordered_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t value,
    loom_value_id_t lower, loom_value_id_t upper, uint32_t lane,
    loom_type_t lane_type, loom_type_t mask_lane_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t below_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_compare_lane(
      context, source_op, &plan->lower_compare_descriptor, value, lower,
      mask_lane_type, &below_lower));
  loom_value_id_t at_least_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_select_lane(
      context, source_op, plan, value, lower, plan->lower, below_lower, lane,
      lane_type, &at_least_lower));

  loom_value_id_t above_upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_compare_lane(
      context, source_op, &plan->upper_compare_descriptor, at_least_lower,
      upper, mask_lane_type, &above_upper));
  return loom_amdgpu_emit_clampf_select_lane(
      context, source_op, plan, at_least_lower, upper, plan->upper, above_upper,
      lane, lane_type, out_result);
}

static iree_status_t loom_amdgpu_lower_clampf_number_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t value,
    loom_value_id_t low_lower, loom_value_id_t low_upper, uint32_t lane,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t at_least_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_number_bound_lane(
      context, source_op, plan, value, low_lower, plan->lower, lane,
      plan->lane_count, lane_type, LOOM_AMDGPU_CLAMPF_BOUND_LOWER,
      &at_least_lower));
  return loom_amdgpu_emit_clampf_number_bound_lane(
      context, source_op, plan, at_least_lower, low_upper, plan->upper, lane,
      plan->lane_count, lane_type, LOOM_AMDGPU_CLAMPF_BOUND_UPPER, out_result);
}

iree_status_t loom_amdgpu_lower_clampf(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       const loom_amdgpu_clampf_plan_t* plan) {
  const bool has_valid_mode = plan->mode == LOOM_AMDGPU_CLAMPF_MODE_ORDERED ||
                              plan->mode == LOOM_AMDGPU_CLAMPF_MODE_NUMBER;
  IREE_ASSERT_TRUE(has_valid_mode);
  if (!has_valid_mode) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid AMDGPU clamp lowering plan");
  }
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->value, &low_value));
  loom_value_id_t low_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->lower, &low_lower));
  loom_value_id_t low_upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->upper, &low_upper));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_value, lane_count, i, lane_type, &lane_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_value, &lane_value));

    switch (plan->mode) {
      case LOOM_AMDGPU_CLAMPF_MODE_ORDERED: {
        loom_value_id_t lane_lower = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
            context, source_op, low_lower, lane_count, i, lane_type,
            &lane_lower));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
            context, source_op, lane_lower, &lane_lower));
        loom_value_id_t lane_upper = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
            context, source_op, low_upper, lane_count, i, lane_type,
            &lane_upper));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
            context, source_op, lane_upper, &lane_upper));
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_clampf_ordered_lane(
            context, source_op, plan, lane_value, lane_lower, lane_upper, i,
            lane_type, mask_lane_type, &lane_results[i]));
        break;
      }
      case LOOM_AMDGPU_CLAMPF_MODE_NUMBER: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_clampf_number_lane(
            context, source_op, plan, lane_value, low_lower, low_upper, i,
            lane_type, &lane_results[i]));
        break;
      }
      case LOOM_AMDGPU_CLAMPF_MODE_NONE: {
        break;
      }
    }
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_emit_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i1_mask_exec_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_type_t mask_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_op_t* exec_read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->mask_exec_read_descriptor,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &mask_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &exec_read_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(exec_read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i1_zero_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t lane_type, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, lane_type,
      &zero));
  const loom_value_id_t zero_lanes[] = {zero, zero};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), zero_lanes,
      IREE_ARRAYSIZE(zero_lanes), mask_type, source_op->location, &concat_op));
  *out_mask = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_emit_i1_mask_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_type_t lane_type, loom_type_t mask_type,
    const loom_amdgpu_vector_select_plan_t* plan,
    loom_value_id_t* out_low_value) {
  bool constant = false;
  if (loom_amdgpu_value_as_i1_constant(context, source_value, &constant)) {
    if (constant) {
      return loom_amdgpu_emit_i1_mask_exec_read(context, source_op, plan,
                                                mask_type, out_low_value);
    }
    return loom_amdgpu_emit_i1_zero_mask(context, source_op, lane_type,
                                         mask_type, out_low_value);
  }
  return loom_amdgpu_lookup_or_materialize_native_i1_mask(
      context, source_op, source_value, out_low_value);
}

static iree_status_t loom_amdgpu_emit_i1_mask_invert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t low_mask,
    loom_type_t mask_type, loom_value_id_t* out_inverse_mask) {
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_exec_read(
      context, source_op, plan, mask_type, &exec_mask));
  return loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->mask_xor_descriptor, low_mask, exec_mask,
      mask_type, out_inverse_mask);
}

static iree_status_t loom_amdgpu_lower_i1_mask_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->condition, &low_condition));
  bool true_constant = false;
  const bool true_is_constant = loom_amdgpu_value_as_i1_constant(
      context, plan->true_value, &true_constant);
  bool false_constant = false;
  const bool false_is_constant = loom_amdgpu_value_as_i1_constant(
      context, plan->false_value, &false_constant);
  if (plan->true_value == plan->false_value) {
    loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->true_value, lane_type, mask_type, plan,
        &low_value));
    return loom_low_lower_bind_value(context, plan->result, low_value);
  }

  loom_value_id_t low_true_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_false_value = LOOM_VALUE_ID_INVALID;

  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->true_value, lane_type, mask_type, plan,
        &low_true_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->false_value, lane_type, mask_type, plan,
        &low_false_value));
    loom_value_id_t lane_results[2] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(lane_results); ++i) {
      loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_true_value, IREE_ARRAYSIZE(lane_results), i,
          lane_type, &lane_true_value));
      loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_false_value, IREE_ARRAYSIZE(lane_results), i,
          lane_type, &lane_false_value));
      const loom_value_id_t operands[] = {
          lane_true_value,
          lane_false_value,
          low_condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->scc_descriptor, operands, IREE_ARRAYSIZE(operands),
          loom_named_attr_slice_empty(), &lane_type, 1,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &select_op));
      lane_results[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_concat_build(loom_low_lower_context_builder(context),
                              lane_results, IREE_ARRAYSIZE(lane_results),
                              mask_type, source_op->location, &concat_op));
    return loom_low_lower_bind_value(context, plan->result,
                                     loom_low_concat_result(concat_op));
  }

  IREE_ASSERT_EQ(plan->condition_kind,
                 LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK);
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_native_i1_mask(
      context, source_op, plan->condition, &low_condition));

  if (true_is_constant && false_is_constant) {
    if (true_constant == false_constant) {
      loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
          context, source_op, plan->true_value, lane_type, mask_type, plan,
          &result_mask));
      return loom_low_lower_bind_value(context, plan->result, result_mask);
    }
    if (true_constant && !false_constant) {
      return loom_low_lower_bind_value(context, plan->result, low_condition);
    }
    loom_value_id_t inverse_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_i1_mask_invert(context, source_op, plan, low_condition,
                                        mask_type, &inverse_condition));
    return loom_low_lower_bind_value(context, plan->result, inverse_condition);
  }

  if (plan->true_value == plan->condition ||
      (true_is_constant && true_constant)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->false_value, lane_type, mask_type, plan,
        &low_false_value));
    loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->mask_or_descriptor, low_condition,
        low_false_value, mask_type, &result_mask));
    return loom_low_lower_bind_value(context, plan->result, result_mask);
  }

  if (true_is_constant && !true_constant) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->false_value, lane_type, mask_type, plan,
        &low_false_value));
    loom_value_id_t inverse_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_i1_mask_invert(context, source_op, plan, low_condition,
                                        mask_type, &inverse_condition));
    if (false_is_constant && false_constant) {
      return loom_low_lower_bind_value(context, plan->result,
                                       inverse_condition);
    }
    loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->mask_and_descriptor, inverse_condition,
        low_false_value, mask_type, &result_mask));
    return loom_low_lower_bind_value(context, plan->result, result_mask);
  }

  if (false_is_constant && false_constant) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
        context, source_op, plan->true_value, lane_type, mask_type, plan,
        &low_true_value));
    loom_value_id_t inverse_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_i1_mask_invert(context, source_op, plan, low_condition,
                                        mask_type, &inverse_condition));
    loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->mask_or_descriptor, low_true_value,
        inverse_condition, mask_type, &result_mask));
    return loom_low_lower_bind_value(context, plan->result, result_mask);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
      context, source_op, plan->true_value, lane_type, mask_type, plan,
      &low_true_value));
  loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->mask_and_descriptor, low_condition,
      low_true_value, mask_type, &true_mask));
  if (plan->false_value == plan->condition ||
      (false_is_constant && !false_constant)) {
    return loom_low_lower_bind_value(context, plan->result, true_mask);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_emit_i1_mask_value(
      context, source_op, plan->false_value, lane_type, mask_type, plan,
      &low_false_value));
  loom_value_id_t inverse_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_invert(
      context, source_op, plan, low_condition, mask_type, &inverse_condition));
  loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->mask_and_descriptor, inverse_condition,
      low_false_value, mask_type, &false_mask));
  loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->mask_or_descriptor, true_mask, false_mask,
      mask_type, &result_mask));
  return loom_low_lower_bind_value(context, plan->result, result_mask);
}

iree_status_t loom_amdgpu_lower_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  if (plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK) {
    return loom_amdgpu_lower_i1_mask_select(context, source_op, plan);
  }

  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->condition, &low_condition));
  loom_value_id_t low_true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->true_value, &low_true_value));
  loom_value_id_t low_false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->false_value,
                                                   &low_false_value));

  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC) {
    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
    loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < lane_count; ++i) {
      loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_true_value, lane_count, i, lane_type,
          &lane_true_value));
      loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_false_value, lane_count, i, lane_type,
          &lane_false_value));
      if (lane_true_value == lane_false_value) {
        lane_results[i] = lane_true_value;
        continue;
      }
      const loom_value_id_t operands[] = {
          lane_true_value,
          lane_false_value,
          low_condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->scc_descriptor, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &select_op));
      lane_results[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
    if (lane_count == 1) {
      return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
    }
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, plan->result, &result_type));
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context), lane_results, lane_count,
        result_type, source_op->location, &concat_op));
    return loom_low_lower_bind_value(context, plan->result,
                                     loom_low_concat_result(concat_op));
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_condition = LOOM_VALUE_ID_INVALID;
    if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK) {
      lane_condition = low_condition;
    } else {
      const uint32_t condition_lane = i / plan->registers_per_condition_lane;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_condition, lane_count, condition_lane * 2u,
          mask_lane_type, &lane_condition));
    }

    bool emitted_immediate = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_select_immediate_lane(
        context, source_op, plan, low_false_value, low_true_value,
        lane_condition, i, lane_type, &lane_results[i], &emitted_immediate));
    if (emitted_immediate) {
      continue;
    }

    if (loom_amdgpu_select_source_lanes_have_same_bits(fact_table, module, plan,
                                                       i)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
          context, source_op, low_true_value, lane_count, i, lane_type,
          &lane_results[i]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, lane_results[i], &lane_results[i]));
      continue;
    }

    loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, low_true_value, lane_count, i, lane_type,
        &lane_true_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_true_value, &lane_true_value));
    loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, low_false_value, lane_count, i, lane_type,
        &lane_false_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_false_value, &lane_false_value));
    if (lane_false_value == lane_true_value) {
      lane_results[i] = lane_true_value;
      continue;
    }
    const loom_value_id_t operands[] = {
        lane_false_value,
        lane_true_value,
        lane_condition,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->register_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, lane_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}
