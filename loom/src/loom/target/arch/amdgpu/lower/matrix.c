// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix.h"

#include <stdint.h>

#include "loom/analysis/contract_vector.h"
#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/matrix/projection.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"

typedef struct loom_amdgpu_matrix_target_facts_t {
  // Generic vector.mma adapter options for this AMDGPU target.
  loom_contract_vector_mma_options_t options;
  // Matrix feature bits available on the selected processor.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Concrete wave size selected by the target processor.
  uint32_t wave_size;
} loom_amdgpu_matrix_target_facts_t;

static iree_status_t loom_amdgpu_matrix_target_facts_from_environment(
    const loom_target_contract_query_environment_t* environment,
    loom_amdgpu_matrix_target_facts_t* out_facts) {
  *out_facts = (loom_amdgpu_matrix_target_facts_t){0};

  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_ref(environment->module,
                                            environment->target_ref);
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU matrix lowering requires an AMDGPU "
                            "processor target record");
  }
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  (void)loom_amdgpu_matrix_feature_bits_from_profile(processor->features.matrix,
                                                     &feature_bits);
  if (environment->bundle == NULL || environment->bundle->snapshot == NULL ||
      environment->bundle->snapshot->subgroup_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU matrix lowering requires a fixed target "
                            "subgroup size");
  }
  const uint16_t wavefront_size =
      (uint16_t)environment->bundle->snapshot->subgroup_size;

  *out_facts = (loom_amdgpu_matrix_target_facts_t){
      .options =
          (loom_contract_vector_mma_options_t){
              .k_group_size = 1,
              .fragment =
                  (loom_contract_fragment_t){
                      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
                      .subgroup_size = wavefront_size,
                  },
              .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
              .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
          },
      .feature_bits = feature_bits,
      .wave_size = wavefront_size,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_matrix_select_contract(
    const loom_contract_request_t* contract_request,
    const loom_amdgpu_matrix_target_facts_t* target_facts,
    const loom_amdgpu_matrix_contract_descriptor_t** out_descriptor,
    loom_contract_diagnostic_t* out_contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_match_diagnostic) {
  *out_descriptor = NULL;
  if (out_contract_diagnostic != NULL) {
    *out_contract_diagnostic = (loom_contract_diagnostic_t){0};
  }
  if (out_match_diagnostic != NULL) {
    *out_match_diagnostic = (loom_amdgpu_matrix_contract_match_diagnostic_t){0};
  }

  loom_amdgpu_matrix_contract_match_request_t match_request = {0};
  if (!loom_amdgpu_matrix_contract_match_request_from_contract(
          contract_request, target_facts->feature_bits, target_facts->wave_size,
          &match_request, out_contract_diagnostic)) {
    return false;
  }

  *out_descriptor =
      loom_amdgpu_matrix_contract_select(&match_request, out_match_diagnostic);
  return *out_descriptor != NULL;
}

static iree_string_view_t loom_amdgpu_matrix_contract_rejection_key(
    loom_contract_diagnostic_t contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic) {
  const loom_contract_rejection_bits_t contract_bits =
      contract_diagnostic.rejection_bits;
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV("source.fragment_roles");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("source.fragment_shape");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
    return IREE_SV("source.fragment_schema");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("source.numeric_type");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV("source.fragment_ownership");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("source.capability_class");
  }
  if (iree_any_bit_set(contract_bits,
                       LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("source.request_shape");
  }

  const loom_amdgpu_matrix_contract_rejection_bits_t match_bits =
      match_diagnostic.rejection_bits;
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FAMILY)) {
    return IREE_SV("family");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE)) {
    return IREE_SV("tile_shape");
  }
  if (iree_any_bit_set(
          match_bits,
          LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RESULT_PAYLOAD)) {
    return IREE_SV("payload_shape");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND)) {
    return IREE_SV("scale_kind");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES)) {
    return IREE_SV("processor_features");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE)) {
    return IREE_SV("wave_size");
  }
  if (iree_any_bit_set(
          match_bits,
          LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS)) {
    const loom_amdgpu_matrix_contract_rejection_bits_t selector_bits =
        match_bits &
        (LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS |
         LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS);
    if (selector_bits ==
        (LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS |
         LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS)) {
      return IREE_SV("format_selector.matrix_and_scale");
    }
    if (iree_any_bit_set(
            selector_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS)) {
      return IREE_SV("format_selector.matrix");
    }
    return IREE_SV("format_selector.scale");
  }
  if (iree_any_bit_set(
          match_bits,
          LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_CLAMP |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SIGN_SELECT |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_AB_MODIFIERS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_C_MODIFIER |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_OPSEL |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS)) {
    return IREE_SV("required_operands");
  }
  return IREE_SV("descriptor_match");
}

static iree_string_view_t loom_amdgpu_matrix_diagnostic_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_amdgpu_matrix_diagnostic_symbol_name(
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

static iree_string_view_t loom_amdgpu_matrix_diagnostic_function_name(
    const loom_target_contract_query_environment_t* environment) {
  if (!loom_func_like_isa(environment->function)) {
    return IREE_SV("<module>");
  }
  return loom_amdgpu_matrix_diagnostic_symbol_name(
      environment->module, loom_func_like_callee(environment->function));
}

static void loom_amdgpu_matrix_diagnostic_make_context_params(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_diagnostic_param_t* params) {
  params[0] = loom_param_string(loom_amdgpu_matrix_diagnostic_nonempty(
      environment->bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_amdgpu_matrix_diagnostic_nonempty(
      environment->bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_amdgpu_matrix_diagnostic_nonempty(
      environment->bundle->config->name, IREE_SV("<empty>")));
  params[3] = loom_param_string(
      loom_amdgpu_matrix_diagnostic_function_name(environment));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
}

static iree_status_t loom_amdgpu_matrix_contract_query_reject_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_target_contract_query_outcome_t outcome,
    loom_contract_diagnostic_t contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic,
    loom_target_contract_query_result_t* out_result) {
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 8, sizeof(*params), (void**)&params));
  loom_amdgpu_matrix_diagnostic_make_context_params(environment, source_op,
                                                    params);
  params[5] = loom_param_string(loom_amdgpu_matrix_contract_rejection_key(
      contract_diagnostic, match_diagnostic));
  params[6] = loom_param_u32(contract_diagnostic.rejection_bits);
  params[7] = loom_param_u32(match_diagnostic.rejection_bits);
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_AMDGPU_021_REF,
      .params = params,
      .param_count = 8,
  };
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = outcome;
  out_result->source_rejection_bits = contract_diagnostic.rejection_bits;
  out_result->target_rejection_bits = match_diagnostic.rejection_bits;
  out_result->rejection = rejection;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_matrix_contract_query_reject_descriptor(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_target_contract_query_outcome_t outcome,
    const loom_amdgpu_matrix_fragment_layout_t* fragment_layout,
    iree_string_view_t descriptor_name,
    iree_string_view_t descriptor_constraint,
    loom_target_contract_query_result_t* out_result) {
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  loom_amdgpu_matrix_diagnostic_make_context_params(environment, source_op,
                                                    params);
  params[5] = loom_param_string(descriptor_name);
  params[6] = loom_param_string(descriptor_constraint);
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_AMDGPU_022_REF,
      .params = params,
      .param_count = 7,
  };
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = outcome;
  out_result->selected_matrix_fragment_layout = fragment_layout;
  out_result->rejection = rejection;
  return iree_ok_status();
}

static bool loom_amdgpu_matrix_format_selector_from_encoded_format(
    loom_value_fact_numeric_format_flags_t format, int64_t* out_value) {
  *out_value = 0;
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
      *out_value = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8:
      *out_value = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3:
      *out_value = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6:
      *out_value = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1:
      *out_value = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP4;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_matrix_scale_format_selector_from_encoded_format(
    loom_value_fact_numeric_format_flags_t format, int64_t* out_value) {
  *out_value = 0;
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0:
      *out_value = LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
      *out_value = LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3;
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_matrix_format_selector_attr(
    const loom_contract_operand_t* operand, iree_string_view_t field_name,
    int64_t* out_value) {
  const loom_value_fact_numeric_format_flags_t format =
      operand->encoded.target_schema.encoded_operand.element_format;
  if (!loom_amdgpu_matrix_format_selector_from_encoded_format(format,
                                                              out_value)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "selected AMDGPU matrix descriptor requires '%.*s', but the source "
        "matrix contract does not provide a supported matrix format selector",
        (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_matrix_scale_format_selector_attr(
    const loom_contract_operand_t* operand, iree_string_view_t field_name,
    int64_t* out_value) {
  const loom_value_fact_numeric_format_flags_t format =
      operand->encoded.target_schema.encoded_operand.scale_format;
  if (!loom_amdgpu_matrix_scale_format_selector_from_encoded_format(
          format, out_value)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "selected AMDGPU matrix descriptor requires '%.*s', but the source "
        "matrix contract does not provide a supported scale format selector",
        (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_matrix_attr_value(
    const loom_contract_request_t* contract_request,
    const loom_low_immediate_t* immediate, iree_string_view_t field_name,
    int64_t* out_value) {
  if (iree_string_view_equal(field_name, IREE_SV("matrix_a_fmt"))) {
    return loom_amdgpu_matrix_format_selector_attr(&contract_request->lhs,
                                                   field_name, out_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("matrix_b_fmt"))) {
    return loom_amdgpu_matrix_format_selector_attr(&contract_request->rhs,
                                                   field_name, out_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("matrix_a_scale_fmt"))) {
    return loom_amdgpu_matrix_scale_format_selector_attr(&contract_request->lhs,
                                                         field_name, out_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("matrix_b_scale_fmt"))) {
    return loom_amdgpu_matrix_scale_format_selector_attr(&contract_request->rhs,
                                                         field_name, out_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("matrix_a_scale")) ||
      iree_string_view_equal(field_name, IREE_SV("matrix_b_scale")) ||
      iree_string_view_equal(field_name, IREE_SV("matrix_a_reuse")) ||
      iree_string_view_equal(field_name, IREE_SV("matrix_b_reuse"))) {
    *out_value = 0;
    return iree_ok_status();
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "AMDGPU descriptor-matrix selected immediate '%.*s' is unmapped",
      (int)field_name.size, field_name.data);
}

iree_status_t loom_amdgpu_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)rule;
  *out_options = (loom_contract_vector_mma_options_t){0};
  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_target_facts_from_environment(
      environment, &target_facts));
  *out_options = target_facts.options;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  (void)rule;
  *out_result = loom_target_contract_query_result_empty();

  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_target_facts_from_environment(
      environment, &target_facts));
  loom_contract_diagnostic_t contract_diagnostic = {0};
  loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic = {0};
  const loom_amdgpu_matrix_contract_descriptor_t* contract_descriptor = NULL;
  if (!loom_amdgpu_matrix_select_contract(
          contract_request, &target_facts, &contract_descriptor,
          &contract_diagnostic, &match_diagnostic)) {
    return loom_amdgpu_matrix_contract_query_reject_contract(
        environment, source_op, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        contract_diagnostic, match_diagnostic, out_result);
  }
  if (contract_descriptor->low_descriptor_ref ==
      LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE) {
    const loom_amdgpu_matrix_fragment_layout_t* fragment_layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(
            contract_descriptor);
    return loom_amdgpu_matrix_contract_query_reject_descriptor(
        environment, source_op, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        fragment_layout, contract_descriptor->name,
        IREE_SV("target_low_descriptor_mapping"), out_result);
  }
  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      environment->descriptor_set, contract_descriptor->low_descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const loom_amdgpu_matrix_fragment_layout_t* fragment_layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(
            contract_descriptor);
    return loom_amdgpu_matrix_contract_query_reject_descriptor(
        environment, source_op, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        fragment_layout, contract_descriptor->name,
        IREE_SV("descriptor_set_packet"), out_result);
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(environment->descriptor_set,
                                            descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);

  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor = descriptor;
  out_result->selected_matrix_fragment_layout =
      loom_amdgpu_matrix_contract_descriptor_fragment_layout(
          contract_descriptor);
  out_result->source_rejection_bits = contract_diagnostic.rejection_bits;
  out_result->target_rejection_bits = match_diagnostic.rejection_bits;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_descriptor_matrix_attrs(
    void* user_data, loom_low_lower_context_t* context,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_request_t* contract_request,
    const loom_low_descriptor_t* descriptor,
    loom_named_attr_slice_t* out_attrs) {
  (void)user_data;
  (void)rule;
  *out_attrs = loom_named_attr_slice_empty();
  if (descriptor->immediate_count == 0) {
    return iree_ok_status();
  }

  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, descriptor->immediate_count, sizeof(*attrs), (void**)&attrs));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  IREE_ASSERT((uint64_t)descriptor->immediate_start +
                  (uint64_t)descriptor->immediate_count <=
              descriptor_set->immediate_count);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_row = descriptor->immediate_start + i;
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_row];
    const iree_string_view_t field_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context), field_name, &attrs[i].name_id));
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_matrix_attr_value(
        contract_request, immediate, field_name, &value));
    attrs[i].value = loom_attr_i64(value);
  }

  *out_attrs = loom_make_named_attr_slice(attrs, descriptor->immediate_count);
  return iree_ok_status();
}
