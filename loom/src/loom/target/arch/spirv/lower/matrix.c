// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/matrix.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/profile.h"

static bool loom_spirv_contract_numeric_scalar_type(
    loom_contract_numeric_type_t numeric_type,
    loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I8:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S8;
      return true;
    case LOOM_CONTRACT_NUMERIC_U8:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U8;
      return true;
    case LOOM_CONTRACT_NUMERIC_I16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S16;
      return true;
    case LOOM_CONTRACT_NUMERIC_U16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U16;
      return true;
    case LOOM_CONTRACT_NUMERIC_I32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S32;
      return true;
    case LOOM_CONTRACT_NUMERIC_U32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U32;
      return true;
    case LOOM_CONTRACT_NUMERIC_F16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16;
      return true;
    case LOOM_CONTRACT_NUMERIC_BF16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_BF16;
      return true;
    case LOOM_CONTRACT_NUMERIC_F32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F32;
      return true;
    case LOOM_CONTRACT_NUMERIC_F64:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F64;
      return true;
    default:
      return false;
  }
}

static bool loom_spirv_contract_numeric_is_signed_integer(
    loom_contract_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I4:
    case LOOM_CONTRACT_NUMERIC_I8:
    case LOOM_CONTRACT_NUMERIC_I16:
    case LOOM_CONTRACT_NUMERIC_I32:
      return true;
    default:
      return false;
  }
}

static bool loom_spirv_contract_dimension_u16(int64_t value,
                                              uint16_t* out_value) {
  if (value <= 0 || value > UINT16_MAX) {
    *out_value = 0;
    return false;
  }
  *out_value = (uint16_t)value;
  return true;
}

static loom_spirv_cooperative_matrix_operand_flags_t
loom_spirv_cooperative_matrix_operand_flags(
    const loom_contract_request_t* contract_request) {
  loom_spirv_cooperative_matrix_operand_flags_t flags = 0;
  if (loom_spirv_contract_numeric_is_signed_integer(
          contract_request->lhs.numeric_type)) {
    flags |= LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS;
  }
  if (loom_spirv_contract_numeric_is_signed_integer(
          contract_request->rhs.numeric_type)) {
    flags |= LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS;
  }
  if (loom_spirv_contract_numeric_is_signed_integer(
          contract_request->accumulator.numeric_type)) {
    flags |= LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS;
  }
  if (loom_spirv_contract_numeric_is_signed_integer(
          contract_request->result.numeric_type)) {
    flags |= LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS;
  }
  return flags;
}

static iree_string_view_t loom_spirv_matrix_rejection_key_from_contract(
    loom_contract_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("source.request_shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("source.static_shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("source.numeric_type");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("source.capability_class");
  }
  return IREE_SV("source.matrix_contract");
}

static iree_string_view_t loom_spirv_matrix_rejection_key_from_property(
    const loom_spirv_cooperative_diagnostic_t* diagnostic) {
  const loom_spirv_cooperative_rejection_flags_t flags =
      diagnostic->rejection_flags;
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE)) {
    return loom_spirv_feature_atom_name(diagnostic->feature_atom);
  }
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE)) {
    return IREE_SV("cooperative_matrix.shape");
  }
  if (iree_any_bit_set(flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE)) {
    return IREE_SV("cooperative_matrix.scalar_type");
  }
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_SCOPE)) {
    return IREE_SV("cooperative_matrix.scope");
  }
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_LAYOUT)) {
    return IREE_SV("cooperative_matrix.layout");
  }
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_STORAGE_CLASS)) {
    return IREE_SV("cooperative_matrix.storage_class");
  }
  if (iree_any_bit_set(flags, LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS)) {
    return IREE_SV("cooperative_matrix.operands");
  }
  if (iree_any_bit_set(flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK)) {
    return IREE_SV("cooperative_matrix.reference_fallback");
  }
  return IREE_SV("cooperative_matrix.property");
}

static iree_string_view_t loom_spirv_matrix_diagnostic_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_spirv_matrix_diagnostic_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_spirv_matrix_diagnostic_function_name(
    const loom_target_contract_query_environment_t* environment) {
  if (!loom_func_like_isa(environment->function)) {
    return IREE_SV("<module>");
  }
  return loom_spirv_matrix_diagnostic_symbol_name(
      environment->module, loom_func_like_callee(environment->function));
}

static void loom_spirv_matrix_diagnostic_make_context_params(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, iree_string_view_t matrix_constraint,
    loom_diagnostic_param_t* params) {
  params[0] = loom_param_string(loom_spirv_matrix_diagnostic_nonempty(
      environment->bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_spirv_matrix_diagnostic_nonempty(
      environment->bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_spirv_matrix_diagnostic_nonempty(
      environment->bundle->config->name, IREE_SV("<empty>")));
  params[3] = loom_param_string(
      loom_spirv_matrix_diagnostic_function_name(environment));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
  params[5] = loom_param_string(IREE_SV("spirv.cooperative_matrix"));
  params[6] = loom_param_string(matrix_constraint);
}

static iree_status_t loom_spirv_matrix_contract_query_reject(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, iree_string_view_t matrix_constraint,
    uint32_t source_rejection_bits, uint32_t target_rejection_bits,
    uint32_t missing_feature_bits,
    loom_target_contract_query_result_t* out_result) {
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  loom_spirv_matrix_diagnostic_make_context_params(environment, source_op,
                                                   matrix_constraint, params);
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_TARGET_039_REF,
      .params = params,
      .param_count = 7,
  };
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  out_result->source_rejection_bits = source_rejection_bits;
  out_result->target_rejection_bits = target_rejection_bits;
  out_result->missing_feature_bits = missing_feature_bits;
  out_result->rejection = rejection;
  return iree_ok_status();
}

static bool loom_spirv_cooperative_matrix_query_from_contract(
    const loom_contract_request_t* contract_request,
    loom_spirv_cooperative_matrix_query_t* out_query,
    loom_contract_diagnostic_t* out_diagnostic) {
  *out_query = (loom_spirv_cooperative_matrix_query_t){0};
  *out_diagnostic = (loom_contract_diagnostic_t){0};

  if (contract_request->kind != LOOM_CONTRACT_KIND_MATRIX_MULTIPLY) {
    out_diagnostic->rejection_bits = LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
    return false;
  }
  if (contract_request->capability_class !=
      LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX) {
    out_diagnostic->rejection_bits = LOOM_CONTRACT_REJECTION_CAPABILITY;
    return false;
  }

  if (!loom_spirv_contract_dimension_u16(contract_request->shape.m,
                                         &out_query->m_size) ||
      !loom_spirv_contract_dimension_u16(contract_request->shape.n,
                                         &out_query->n_size) ||
      !loom_spirv_contract_dimension_u16(contract_request->shape.k,
                                         &out_query->k_size)) {
    out_diagnostic->rejection_bits = LOOM_CONTRACT_REJECTION_SHAPE;
    return false;
  }

  if (!loom_spirv_contract_numeric_scalar_type(
          contract_request->lhs.numeric_type, &out_query->lhs_type) ||
      !loom_spirv_contract_numeric_scalar_type(
          contract_request->rhs.numeric_type, &out_query->rhs_type) ||
      !loom_spirv_contract_numeric_scalar_type(
          contract_request->accumulator.numeric_type,
          &out_query->accumulator_type) ||
      !loom_spirv_contract_numeric_scalar_type(
          contract_request->result.numeric_type, &out_query->result_type)) {
    out_diagnostic->rejection_bits = LOOM_CONTRACT_REJECTION_NUMERIC;
    return false;
  }

  out_query->scope = LOOM_SPIRV_SCOPE_SUBGROUP;
  out_query->layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR;
  out_query->storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER;
  out_query->operand_flags =
      loom_spirv_cooperative_matrix_operand_flags(contract_request);
  out_query->policy = contract_request->policy;
  return true;
}

static void loom_spirv_matrix_prepare_properties(
    const loom_target_contract_query_environment_t* environment,
    loom_spirv_cooperative_property_set_t* out_property_set) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_from_data(environment->target_data);
  if (profile != NULL && profile->cooperative_properties != NULL) {
    *out_property_set = *profile->cooperative_properties;
    return;
  }
  // Cooperative property selection only needs feature atom membership; full
  // SPIR-V capability/extension preparation is emission-owned.
  const loom_spirv_feature_set_t feature_set = {
      .atom_bits = (loom_spirv_feature_bits_t)
                       environment->bundle->config->contract_feature_bits,
  };
  loom_spirv_cooperative_property_set_prepare(&feature_set, out_property_set);
}

static iree_status_t loom_spirv_matrix_query_select_descriptor(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    const loom_spirv_cooperative_matrix_property_t* property,
    loom_target_contract_query_result_t* out_result) {
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(environment->descriptor_set,
                                            property->mul_add_descriptor_ref);
  if (descriptor == NULL) {
    return loom_spirv_matrix_contract_query_reject(
        environment, source_op, IREE_SV("descriptor_set_packet"), 0,
        LOOM_SPIRV_COOPERATIVE_REJECTION_DESCRIPTOR, 0, out_result);
  }
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor = descriptor;
  return iree_ok_status();
}

iree_status_t loom_spirv_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)environment;
  (void)rule;
  *out_options = (loom_contract_vector_mma_options_t){
      .k_group_size = 1,
      .fragment =
          (loom_contract_fragment_t){
              .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
              .subgroup_size = 0,
          },
      .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
      .policy = LOOM_LOWERING_POLICY_REFERENCE_ALLOWED,
  };
  return iree_ok_status();
}

iree_status_t loom_spirv_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  (void)rule;
  *out_result = loom_target_contract_query_result_empty();

  loom_spirv_cooperative_matrix_query_t query = {0};
  loom_contract_diagnostic_t contract_diagnostic = {0};
  if (!loom_spirv_cooperative_matrix_query_from_contract(
          contract_request, &query, &contract_diagnostic)) {
    return loom_spirv_matrix_contract_query_reject(
        environment, source_op,
        loom_spirv_matrix_rejection_key_from_contract(
            contract_diagnostic.rejection_bits),
        contract_diagnostic.rejection_bits, 0, 0, out_result);
  }

  loom_spirv_cooperative_property_set_t property_set = {0};
  loom_spirv_matrix_prepare_properties(environment, &property_set);
  loom_spirv_cooperative_diagnostic_t diagnostic = {0};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);
  if (property == NULL) {
    const uint32_t missing_feature_bits =
        iree_any_bit_set(diagnostic.rejection_flags,
                         LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE)
            ? (uint32_t)loom_spirv_feature_atom_bit(diagnostic.feature_atom)
            : 0;
    return loom_spirv_matrix_contract_query_reject(
        environment, source_op,
        loom_spirv_matrix_rejection_key_from_property(&diagnostic), 0,
        diagnostic.rejection_flags, missing_feature_bits, out_result);
  }

  return loom_spirv_matrix_query_select_descriptor(environment, source_op,
                                                   property, out_result);
}
