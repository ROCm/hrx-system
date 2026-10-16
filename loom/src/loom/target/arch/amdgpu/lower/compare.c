// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/compare.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static const loom_amdgpu_compare_descriptor_candidate_t*
loom_amdgpu_find_compare_descriptor_candidate(loom_op_kind_t op_kind,
                                              uint8_t predicate) {
  switch (op_kind) {
    case LOOM_OP_VECTOR_CMPI:
      if (predicate < LOOM_VECTOR_CMPI_PREDICATE_COUNT_) {
        return &kLoomAmdgpuVectorCmpiCompareDescriptorCandidates[predicate];
      }
      return NULL;
    case LOOM_OP_SCALAR_CMPF:
      if (predicate < LOOM_SCALAR_CMPF_PREDICATE_COUNT_) {
        return &kLoomAmdgpuScalarCmpfCompareDescriptorCandidates[predicate];
      }
      return NULL;
    case LOOM_OP_VECTOR_CMPF:
      if (predicate < LOOM_VECTOR_CMPF_PREDICATE_COUNT_) {
        return &kLoomAmdgpuVectorCmpfCompareDescriptorCandidates[predicate];
      }
      return NULL;
    default:
      return NULL;
  }
}

static iree_status_t loom_amdgpu_resolve_optional_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  bool present = false;
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, &present);
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

static bool loom_amdgpu_compare_immediate_enum_domain_contains(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate, uint32_t value) {
  if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_ENUM ||
      immediate->enum_domain_id >= descriptor_set->enum_domain_count) {
    return false;
  }
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[immediate->enum_domain_id];
  if ((uint64_t)domain->value_start + (uint64_t)domain->value_count >
      descriptor_set->enum_value_count) {
    return false;
  }
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const loom_low_enum_value_t* enum_value =
        &descriptor_set->enum_values[domain->value_start + i];
    if (enum_value->value >= 0 && (uint64_t)enum_value->value == value) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_compare_immediate_value_fits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_lower_resolved_descriptor_t* descriptor, uint32_t value) {
  if (descriptor->descriptor == NULL ||
      loom_low_descriptor_layout(descriptor_set, descriptor->descriptor)
              ->immediate_count != 1) {
    return false;
  }
  const uint32_t immediate_index =
      loom_low_descriptor_layout(descriptor_set, descriptor->descriptor)
          ->immediate_start;
  if (immediate_index >= descriptor_set->immediate_count) {
    return false;
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_index];
  switch (immediate->encoding_id) {
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_SOURCE_INLINE_U32:
      if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
        return value <= immediate->unsigned_max;
      }
      return loom_amdgpu_compare_immediate_enum_domain_contains(
          descriptor_set, immediate, value);
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_SOURCE_INLINE_F32:
      return loom_amdgpu_compare_immediate_enum_domain_contains(
          descriptor_set, immediate, value);
    default:
      return false;
  }
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
    const loom_low_descriptor_set_t* descriptor_set,
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
         loom_amdgpu_compare_immediate_value_fits(descriptor_set, descriptor,
                                                  *out_inline_bits);
}

struct loom_amdgpu_compare_lane_plan_t {
  // Exact source bits encoded by the selected instruction.
  uint32_t bits;
  // Logical lane selecting the inline form.
  uint8_t lane;
  // Row in kLoomAmdgpuCompareImmediateCandidates.
  uint8_t candidate;
};
static_assert(sizeof(loom_amdgpu_compare_lane_plan_t) == 8,
              "selected compare literals must stay compact");
static_assert(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES <= UINT8_MAX,
              "compare lane ordinals must fit in a byte");
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuCompareImmediateCandidates) <=
                  UINT8_MAX,
              "compare candidate ordinals must fit in a byte");

static iree_status_t loom_amdgpu_select_compare_lanes(
    loom_low_lower_context_t* context,
    loom_amdgpu_vector_compare_plan_t* plan) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_compare_lane_plan_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint8_t lane_count = 0;
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    for (uint32_t i = 0;
         i < IREE_ARRAYSIZE(kLoomAmdgpuCompareImmediateCandidates); ++i) {
      loom_amdgpu_compare_lane_plan_t selected = {.lane = lane, .candidate = i};
      if (loom_amdgpu_compare_immediate_candidate_matches(
              descriptor_set, fact_table, module, plan,
              &kLoomAmdgpuCompareImmediateCandidates[i], lane,
              &selected.bits)) {
        lanes[lane_count++] = selected;
        break;
      }
    }
  }
  if (lane_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_compare_lane_plan_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, lane_count * sizeof(*retained), (void**)&retained));
  memcpy(retained, lanes, lane_count * sizeof(*retained));
  plan->inline_lanes = retained;
  plan->inline_lane_count = lane_count;
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_compare_lanes(context, out_plan));
  *out_selected = true;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_range_type(
      context, lhs_lane_count * 2, &result_type));
  return loom_low_lower_plan_value_type(context, result, result_type);
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

static bool loom_amdgpu_vector_float_classification_values(
    const loom_op_t* source_op, loom_value_id_t* out_input,
    loom_value_id_t* out_result, uint32_t* out_class_mask) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_ISNANF:
      *out_input = loom_vector_isnanf_input(source_op);
      *out_result = loom_vector_isnanf_result(source_op);
      *out_class_mask = 0x003u;
      return true;
    case LOOM_OP_VECTOR_ISINFF:
      *out_input = loom_vector_isinff_input(source_op);
      *out_result = loom_vector_isinff_result(source_op);
      *out_class_mask = 0x204u;
      return true;
    case LOOM_OP_VECTOR_ISFINITEF:
      *out_input = loom_vector_isfinitef_input(source_op);
      *out_result = loom_vector_isfinitef_result(source_op);
      *out_class_mask = 0x1F8u;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_float_classification_descriptor_refs(
    loom_scalar_type_t element_type,
    loom_amdgpu_float_classification_form_t form,
    loom_amdgpu_descriptor_ref_t* out_low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_high_descriptor_ref) {
  *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (element_type) {
    case LOOM_SCALAR_TYPE_F16:
      switch (form) {
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_INLINE:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16_CLASSES_INLINE;
          *out_high_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16_INPUT_HIGH_CLASSES_INLINE;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16_CLASSES_LIT;
          *out_high_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16_INPUT_HIGH_CLASSES_LIT;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER:
          *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16;
          *out_high_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F16_INPUT_HIGH;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_NONE:
          return false;
      }
      return false;
    case LOOM_SCALAR_TYPE_F32:
      switch (form) {
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_INLINE:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F32_CLASSES_INLINE;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F32_CLASSES_LIT;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER:
          *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F32;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_NONE:
          return false;
      }
      return false;
    case LOOM_SCALAR_TYPE_F64:
      switch (form) {
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_INLINE:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F64_CLASSES_INLINE;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL:
          *out_low_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F64_CLASSES_LIT;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER:
          *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_CLASS_F64;
          return true;
        case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_NONE:
          return false;
      }
      return false;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_resolve_float_classification_descriptors(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_amdgpu_float_classification_form_t form,
    loom_low_lower_resolved_descriptor_t* out_low_descriptor,
    loom_low_lower_resolved_descriptor_t* out_high_descriptor,
    bool* out_present) {
  *out_low_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_high_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_present = false;
  loom_amdgpu_descriptor_ref_t low_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_descriptor_ref_t high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_float_classification_descriptor_refs(
          element_type, form, &low_descriptor_ref, &high_descriptor_ref)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, low_descriptor_ref, out_low_descriptor, out_present));
  if (!*out_present || high_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, high_descriptor_ref, out_high_descriptor, out_present);
}

iree_status_t loom_amdgpu_select_vector_float_classification_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_float_classification_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_float_classification_plan_t){0};
  *out_selected = false;
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  uint32_t class_mask = 0;
  if (!loom_amdgpu_vector_float_classification_values(source_op, &input,
                                                      &result, &class_mask)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_vector_storage_t input_storage = {0};
  if (!loom_amdgpu_type_vector_storage(loom_module_value_type(module, input),
                                       &input_storage) ||
      (input_storage.element_type != LOOM_SCALAR_TYPE_F16 &&
       input_storage.element_type != LOOM_SCALAR_TYPE_F32 &&
       input_storage.element_type != LOOM_SCALAR_TYPE_F64) ||
      input_storage.register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      loom_amdgpu_vector_i1_lane_count(loom_module_value_type(
          module, result)) != input_storage.element_count) {
    return iree_ok_status();
  }
  bool input_materializable = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_value_can_materialize_as_vgpr_registers(
      context, source_op, input, &input_materializable));
  if (!input_materializable) {
    return iree_ok_status();
  }

  loom_amdgpu_float_classification_form_t form =
      class_mask <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX
          ? LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_INLINE
          : LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL;
  loom_low_lower_resolved_descriptor_t low_descriptor = {0};
  loom_low_lower_resolved_descriptor_t high_descriptor = {0};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_float_classification_descriptors(
      context, input_storage.element_type, form, &low_descriptor,
      &high_descriptor, &descriptors_present));
  if (!descriptors_present &&
      form == LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL) {
    form = LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_float_classification_descriptors(
        context, input_storage.element_type, form, &low_descriptor,
        &high_descriptor, &descriptors_present));
    if (descriptors_present &&
        !loom_amdgpu_descriptor_set_has_ref(
            loom_low_lower_context_descriptor_set(context),
            LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32)) {
      descriptors_present = false;
    }
  }
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_float_classification_plan_t){
      .input = input,
      .low_descriptor = low_descriptor,
      .high_descriptor = high_descriptor,
      .result = result,
      .class_mask = class_mask,
      .lane_count = input_storage.element_count,
      .element_type = input_storage.element_type,
      .form = form,
  };
  *out_selected = true;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_range_type(
      context, input_storage.element_count * 2, &result_type));
  return loom_low_lower_plan_value_type(context, result, result_type);
}

static_assert((uint8_t)LOOM_SCALAR_CMPF_PREDICATE_OLT ==
                  (uint8_t)LOOM_VECTOR_CMPF_PREDICATE_OLT,
              "scalar and vector cmpf ordered-lt predicates must align");
static_assert((uint8_t)LOOM_SCALAR_CMPF_PREDICATE_OGT ==
                  (uint8_t)LOOM_VECTOR_CMPF_PREDICATE_OGT,
              "scalar and vector cmpf ordered-gt predicates must align");

// Generated rules own native compact clamps. Fallback recipes participate in
// legality and selection only when the target lacks a matching native form.
static bool loom_amdgpu_clampf_fallback_descriptors_available(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_op_kind_t compare_op_kind, loom_amdgpu_clampf_mode_t mode) {
  switch (mode) {
    case LOOM_AMDGPU_CLAMPF_MODE_ORDERED: {
      const loom_amdgpu_compare_descriptor_candidate_t* lower_candidate =
          loom_amdgpu_find_compare_descriptor_candidate(
              compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OLT);
      const loom_amdgpu_compare_descriptor_candidate_t* upper_candidate =
          loom_amdgpu_find_compare_descriptor_candidate(
              compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OGT);
      return lower_candidate != NULL && upper_candidate != NULL &&
             loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, lower_candidate->descriptor_ref) &&
             loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, upper_candidate->descriptor_ref) &&
             loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32);
    }
    case LOOM_AMDGPU_CLAMPF_MODE_NUMBER:
      return !loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAXMIN_NUM_F32) &&
             loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32) &&
             loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32);
    case LOOM_AMDGPU_CLAMPF_MODE_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_match_clampf_fallback(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_clampf_plan_t){0};
  loom_op_kind_t compare_op_kind = LOOM_OP_KIND_UNKNOWN;

  switch (source_op->kind) {
    case LOOM_OP_SCALAR_CLAMPF: {
      const loom_value_id_t value = loom_scalar_clampf_value(source_op);
      const loom_value_id_t lower = loom_scalar_clampf_lower(source_op);
      const loom_value_id_t upper = loom_scalar_clampf_upper(source_op);
      const loom_value_id_t result = loom_scalar_clampf_result(source_op);
      const loom_type_t value_type = loom_module_value_type(module, value);
      if (!loom_amdgpu_type_is_f32(value_type) ||
          !loom_type_equal(loom_module_value_type(module, lower), value_type) ||
          !loom_type_equal(loom_module_value_type(module, upper), value_type) ||
          !loom_type_equal(loom_module_value_type(module, result),
                           value_type)) {
        return false;
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
          return false;
      }
      *out_plan = (loom_amdgpu_clampf_plan_t){
          .value = value,
          .lower = lower,
          .upper = upper,
          .mode = mode,
          .result = result,
          .lane_count = 1,
      };
      compare_op_kind = LOOM_OP_SCALAR_CMPF;
      break;
    }
    case LOOM_OP_VECTOR_CLAMPF: {
      const loom_value_id_t value = loom_vector_clampf_value(source_op);
      const loom_value_id_t lower = loom_vector_clampf_lower(source_op);
      const loom_value_id_t upper = loom_vector_clampf_upper(source_op);
      const loom_value_id_t result = loom_vector_clampf_result(source_op);
      const loom_type_t result_type = loom_module_value_type(module, result);
      const uint32_t lane_count =
          loom_amdgpu_vector_f32_lane_count(result_type);
      if (lane_count == 0 ||
          lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
          !loom_type_equal(loom_module_value_type(module, value),
                           result_type) ||
          !loom_type_equal(loom_module_value_type(module, lower),
                           result_type) ||
          !loom_type_equal(loom_module_value_type(module, upper),
                           result_type)) {
        return false;
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
          return false;
      }
      *out_plan = (loom_amdgpu_clampf_plan_t){
          .value = value,
          .lower = lower,
          .upper = upper,
          .mode = mode,
          .result = result,
          .lane_count = lane_count,
      };
      compare_op_kind = LOOM_OP_VECTOR_CMPF;
      break;
    }
    default:
      return false;
  }
  return loom_amdgpu_clampf_fallback_descriptors_available(
      descriptor_set, compare_op_kind, out_plan->mode);
}

static iree_status_t loom_amdgpu_populate_clampf_fallback_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* plan) {
  switch (plan->mode) {
    case LOOM_AMDGPU_CLAMPF_MODE_ORDERED: {
      const loom_op_kind_t compare_op_kind =
          source_op->kind == LOOM_OP_SCALAR_CLAMPF ? LOOM_OP_SCALAR_CMPF
                                                   : LOOM_OP_VECTOR_CMPF;
      const loom_amdgpu_compare_descriptor_candidate_t* lower_candidate =
          loom_amdgpu_find_compare_descriptor_candidate(
              compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OLT);
      const loom_amdgpu_compare_descriptor_candidate_t* upper_candidate =
          loom_amdgpu_find_compare_descriptor_candidate(
              compare_op_kind, LOOM_SCALAR_CMPF_PREDICATE_OGT);
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
          context, lower_candidate->descriptor_ref,
          &plan->lower_compare_descriptor));
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
          context, upper_candidate->descriptor_ref,
          &plan->upper_compare_descriptor));
      bool select_descriptors_present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptors(
          context, LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER,
          LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL, &plan->select_descriptors,
          &select_descriptors_present));
      IREE_ASSERT(select_descriptors_present);
      return iree_ok_status();
    }
    case LOOM_AMDGPU_CLAMPF_MODE_NUMBER: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32,
          &plan->lower_bound_register_descriptor));
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32,
          &plan->upper_bound_register_descriptor));
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_optional_descriptor_ref(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_LIT,
          &plan->lower_bound_literal_descriptor));
      return loom_amdgpu_resolve_optional_descriptor_ref(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_LIT,
          &plan->upper_bound_literal_descriptor);
    }
    case LOOM_AMDGPU_CLAMPF_MODE_NONE:
      IREE_ASSERT_UNREACHABLE("selected AMDGPU clamp recipe has no mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("selected AMDGPU clamp recipe has unknown mode");
  IREE_BUILTIN_UNREACHABLE();
}

typedef enum loom_amdgpu_clampf_bound_kind_e {
  LOOM_AMDGPU_CLAMPF_BOUND_LOWER = 0,
  LOOM_AMDGPU_CLAMPF_BOUND_UPPER = 1,
} loom_amdgpu_clampf_bound_kind_t;

struct loom_amdgpu_clampf_literal_t {
  // Exact bound bits encoded by the selected instruction.
  uint32_t bits;
  // Logical lane selecting an immediate bound.
  uint8_t lane;
  // Lower or upper bound receiving the immediate.
  uint8_t bound;
};
static_assert(sizeof(loom_amdgpu_clampf_literal_t) == 8,
              "selected clamp literals must stay compact");
static_assert(2 * LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES <= UINT8_MAX,
              "selected clamp literal counts must fit in a byte");

static iree_status_t loom_amdgpu_select_clampf_literals(
    loom_low_lower_context_t* context, loom_amdgpu_clampf_plan_t* plan) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t sources[] = {plan->lower, plan->upper};
  const loom_low_lower_resolved_descriptor_t* descriptors[] = {
      &plan->lower_bound_literal_descriptor,
      &plan->upper_bound_literal_descriptor,
  };
  const bool is_ordered = plan->mode == LOOM_AMDGPU_CLAMPF_MODE_ORDERED;
  loom_amdgpu_clampf_literal_t
      literals[2 * LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint8_t literal_count = 0;
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    for (uint32_t bound = 0; bound < IREE_ARRAYSIZE(sources); ++bound) {
      const loom_low_lower_resolved_descriptor_t* descriptor =
          is_ordered ? &plan->select_descriptors.src1_inline_descriptor
                     : descriptors[bound];
      loom_amdgpu_clampf_literal_t selected = {.lane = lane, .bound = bound};
      if (descriptor->descriptor != NULL &&
          loom_amdgpu_source_lane_as_u32_bits(
              fact_table, module, sources[bound], lane, &selected.bits) &&
          (!is_ordered || selected.bits <= 64)) {
        literals[literal_count++] = selected;
      }
    }
  }
  if (literal_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_clampf_literal_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, literal_count * sizeof(*retained), (void**)&retained));
  memcpy(retained, literals, literal_count * sizeof(*retained));
  plan->literals = retained;
  plan->literal_count = literal_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_clampf_fallback_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_clampf_plan_t){0};
  *out_selected = false;
  if (!loom_amdgpu_match_clampf_fallback(
          loom_low_lower_context_module(context),
          loom_low_lower_context_descriptor_set(context), source_op,
          out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_populate_clampf_fallback_descriptors(
      context, source_op, out_plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_clampf_literals(context, out_plan));
  *out_selected = true;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, out_plan->lane_count, &result_type));
  return loom_low_lower_plan_value_type(context, out_plan->result, result_type);
}

iree_status_t loom_amdgpu_select_scalar_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_clampf_fallback_plan(context, source_op, out_plan,
                                                 out_selected);
}

iree_status_t loom_amdgpu_select_vector_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_clampf_fallback_plan(context, source_op, out_plan,
                                                 out_selected);
}

iree_status_t loom_amdgpu_low_legality_verify_clampf(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }

  loom_amdgpu_clampf_plan_t plan = {0};
  *out_handled = loom_amdgpu_match_clampf_fallback(
      loom_target_low_legality_module(context),
      loom_target_low_legality_descriptor_set(context), op, &plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_compare_immediate_candidate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan,
    const loom_amdgpu_compare_immediate_candidate_t* candidate,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, uint32_t lane,
    loom_type_t mask_lane_type, uint32_t inline_bits,
    loom_value_id_t* out_result) {
  loom_value_id_t lane_operand = loom_amdgpu_compare_immediate_low_value(
      low_lhs, low_rhs, candidate->operand_role);
  const loom_type_t payload_lane_type = loom_amdgpu_low_register_lane_type(
      loom_low_lower_context_module(context), lane_operand);
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, lane_operand, plan->lane_count, lane,
      payload_lane_type, &lane_operand));
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

static iree_status_t loom_amdgpu_lower_vector_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_lhs = loom_low_lower_lookup_value(context, plan->lhs);
  loom_value_id_t low_rhs = loom_low_lower_lookup_value(context, plan->rhs);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lhs_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_lhs);
  const loom_type_t rhs_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_rhs);
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t next_inline_lane = 0;
  for (uint32_t i = 0; i < lane_count; ++i) {
    if (next_inline_lane < plan->inline_lane_count &&
        plan->inline_lanes[next_inline_lane].lane == i) {
      const loom_amdgpu_compare_lane_plan_t* selected =
          &plan->inline_lanes[next_inline_lane++];
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_compare_immediate_candidate(
          context, source_op, plan,
          &kLoomAmdgpuCompareImmediateCandidates[selected->candidate], low_lhs,
          low_rhs, i, mask_lane_type, selected->bits, &lane_results[i]));
      continue;
    }

    loom_value_id_t lane_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_lhs, lane_count, i, lhs_lane_type, &lane_lhs));
    loom_value_id_t lane_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_rhs, lane_count, i, rhs_lane_type, &lane_rhs));
    loom_value_id_t operands[] = {
        lane_lhs,
        lane_rhs,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_legalize_vop3_scalar_sources(
        context, source_op, operands, IREE_ARRAYSIZE(operands)));
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lane_results, lane_count);
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

static iree_status_t loom_amdgpu_extract_float_classification_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_float_classification_plan_t* plan,
    loom_value_id_t low_input, uint32_t lane, loom_type_t vgpr_type,
    loom_type_t vgpr_x2_type, loom_value_id_t* out_lane) {
  uint32_t input_register_count = 0;
  uint32_t register_offset = 0;
  loom_type_t lane_type = vgpr_type;
  switch (plan->element_type) {
    case LOOM_SCALAR_TYPE_F16:
      input_register_count = (plan->lane_count + 1u) / 2u;
      register_offset = lane / 2u;
      break;
    case LOOM_SCALAR_TYPE_F32:
      input_register_count = plan->lane_count;
      register_offset = lane;
      break;
    case LOOM_SCALAR_TYPE_F64:
      input_register_count = plan->lane_count * 2u;
      register_offset = lane * 2u;
      lane_type = vgpr_x2_type;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("unselected AMDGPU float classification type");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (register_offset == 0 &&
      input_register_count == loom_low_register_type_unit_count(lane_type)) {
    *out_lane = low_input;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_input,
                                    register_offset, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_float_classification_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_float_classification_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t input, loom_value_id_t classes, loom_type_t mask_type,
    loom_value_id_t* out_result) {
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  loom_value_id_t operands[2] = {input, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  switch (plan->form) {
    case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_INLINE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
          context, IREE_SV("classes"), plan->class_mask, attrs,
          IREE_ARRAYSIZE(attrs), &attr_count));
      break;
    }
    case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_LITERAL: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
          context, IREE_SV("imm32"), plan->class_mask, attrs,
          IREE_ARRAYSIZE(attrs), &attr_count));
      break;
    }
    case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER:
      operands[1] = classes;
      operand_count = 2;
      break;
    case LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_NONE:
      IREE_ASSERT_UNREACHABLE("unselected AMDGPU float classification form");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* classify_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &classify_op));
  *out_result = loom_value_slice_get(loom_low_op_results(classify_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_float_classification(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_float_classification_plan_t* plan) {
  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_registers(
      context, source_op, plan->input, &low_input));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t vgpr_x2_type = loom_type_none();
  if (plan->element_type == LOOM_SCALAR_TYPE_F64) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  }
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t classes = LOOM_VALUE_ID_INVALID;
  if (plan->form == LOOM_AMDGPU_FLOAT_CLASSIFICATION_FORM_REGISTER) {
    loom_type_t sgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        plan->class_mask, sgpr_type, &classes));
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    loom_value_id_t lane_input = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_float_classification_lane(
        context, source_op, plan, low_input, i, vgpr_type, vgpr_x2_type,
        &lane_input));
    const loom_low_lower_resolved_descriptor_t* descriptor =
        plan->element_type == LOOM_SCALAR_TYPE_F16 && (i & 1u) != 0
            ? &plan->high_descriptor
            : &plan->low_descriptor;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_float_classification_lane(
        context, source_op, plan, descriptor, lane_input, classes, mask_type,
        &lane_results[i]));
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lane_results, plan->lane_count);
}

static iree_status_t loom_amdgpu_emit_clampf_select_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t false_value,
    loom_value_id_t true_value, loom_value_id_t condition,
    const loom_amdgpu_clampf_literal_t* literal, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (literal != NULL) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("true_value"), literal->bits, attrs,
        IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {false_value, condition};
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->select_descriptors.src1_inline_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &lane_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &select_op));
    *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, true_value, &true_value));
  return loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &plan->select_descriptors.register_descriptor,
      false_value, true_value, condition, lane_type, out_value);
}

static iree_status_t loom_amdgpu_emit_clampf_number_bound_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t value,
    loom_value_id_t low_bound, const loom_amdgpu_clampf_literal_t* literal,
    uint32_t lane, uint32_t lane_count, loom_type_t lane_type,
    loom_amdgpu_clampf_bound_kind_t bound_kind, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (literal != NULL) {
    const loom_low_lower_resolved_descriptor_t* literal_descriptor =
        bound_kind == LOOM_AMDGPU_CLAMPF_BOUND_LOWER
            ? &plan->lower_bound_literal_descriptor
            : &plan->upper_bound_literal_descriptor;
    return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
        context, source_op, literal_descriptor, value, literal->bits, lane_type,
        out_value);
  }

  loom_value_id_t bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
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
    loom_value_id_t lower, loom_value_id_t upper,
    const loom_amdgpu_clampf_literal_t* const literals[2],
    loom_type_t lane_type, loom_type_t mask_lane_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t below_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->lower_compare_descriptor, value, lower,
      mask_lane_type, &below_lower));
  loom_value_id_t at_least_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_select_lane(
      context, source_op, plan, value, lower, below_lower,
      literals[LOOM_AMDGPU_CLAMPF_BOUND_LOWER], lane_type, &at_least_lower));

  loom_value_id_t above_upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->upper_compare_descriptor, at_least_lower,
      upper, mask_lane_type, &above_upper));
  return loom_amdgpu_emit_clampf_select_lane(
      context, source_op, plan, at_least_lower, upper, above_upper,
      literals[LOOM_AMDGPU_CLAMPF_BOUND_UPPER], lane_type, out_result);
}

static iree_status_t loom_amdgpu_lower_clampf_number_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_clampf_plan_t* plan, loom_value_id_t value,
    loom_value_id_t low_lower, loom_value_id_t low_upper, uint32_t lane,
    const loom_amdgpu_clampf_literal_t* const literals[2],
    loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t at_least_lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_clampf_number_bound_lane(
      context, source_op, plan, value, low_lower,
      literals[LOOM_AMDGPU_CLAMPF_BOUND_LOWER], lane, plan->lane_count,
      lane_type, LOOM_AMDGPU_CLAMPF_BOUND_LOWER, &at_least_lower));
  return loom_amdgpu_emit_clampf_number_bound_lane(
      context, source_op, plan, at_least_lower, low_upper,
      literals[LOOM_AMDGPU_CLAMPF_BOUND_UPPER], lane, plan->lane_count,
      lane_type, LOOM_AMDGPU_CLAMPF_BOUND_UPPER, out_result);
}

iree_status_t loom_amdgpu_lower_clampf(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       const loom_amdgpu_clampf_plan_t* plan) {
  const bool has_valid_mode = plan->mode == LOOM_AMDGPU_CLAMPF_MODE_ORDERED ||
                              plan->mode == LOOM_AMDGPU_CLAMPF_MODE_NUMBER;
  IREE_ASSERT_TRUE(has_valid_mode);
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_value = loom_low_lower_lookup_value(context, plan->value);
  loom_value_id_t low_lower = loom_low_lower_lookup_value(context, plan->lower);
  loom_value_id_t low_upper = loom_low_lower_lookup_value(context, plan->upper);
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, &low_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_lower, &low_lower));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_upper, &low_upper));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t next_literal = 0;
  for (uint32_t i = 0; i < lane_count; ++i) {
    const loom_amdgpu_clampf_literal_t* literals[2] = {NULL, NULL};
    while (next_literal < plan->literal_count &&
           plan->literals[next_literal].lane == i) {
      const loom_amdgpu_clampf_literal_t* literal =
          &plan->literals[next_literal++];
      literals[literal->bound] = literal;
    }
    loom_value_id_t lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_value, lane_count, i, lane_type, &lane_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_value, &lane_value));

    switch (plan->mode) {
      case LOOM_AMDGPU_CLAMPF_MODE_ORDERED: {
        loom_value_id_t lane_lower = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
            context, source_op, low_lower, lane_count, i, lane_type,
            &lane_lower));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
            context, source_op, lane_lower, &lane_lower));
        loom_value_id_t lane_upper = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
            context, source_op, low_upper, lane_count, i, lane_type,
            &lane_upper));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
            context, source_op, lane_upper, &lane_upper));
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_clampf_ordered_lane(
            context, source_op, plan, lane_value, lane_lower, lane_upper,
            literals, lane_type, mask_lane_type, &lane_results[i]));
        break;
      }
      case LOOM_AMDGPU_CLAMPF_MODE_NUMBER: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_clampf_number_lane(
            context, source_op, plan, lane_value, low_lower, low_upper, i,
            literals, lane_type, &lane_results[i]));
        break;
      }
      case LOOM_AMDGPU_CLAMPF_MODE_NONE:
        IREE_ASSERT_UNREACHABLE("unselected AMDGPU clamp plan");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lane_results, lane_count);
}
