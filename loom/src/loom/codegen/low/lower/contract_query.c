// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/contract_query.h"

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"

static iree_status_t loom_low_lower_contract_query_make_rejection(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_diagnostic_t* diagnostic,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  if (diagnostic == NULL) {
    return iree_ok_status();
  }
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  if (diagnostic->param_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        environment->arena, diagnostic->param_count * sizeof(*params),
        (void**)&params));
    loom_low_lower_rule_materialize_diagnostic_params(
        match_context, rule_set, source_op, diagnostic, params);
  }
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = diagnostic->error_ref,
      .params = params,
      .param_count = diagnostic->param_count,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static void loom_low_lower_contract_query_adopt_case(
    const loom_target_contract_case_t* contract_case, uint16_t case_index,
    loom_target_contract_query_result_t* result) {
  if (result->binding_index == UINT16_MAX) {
    result->binding_index = contract_case->binding_index;
  }
  if (result->case_index == UINT16_MAX) {
    result->case_index = case_index;
  }
  if (result->rule_index == UINT16_MAX) {
    result->rule_index = contract_case->row_index;
  }
}

static iree_string_view_t loom_low_lower_contract_query_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_contract_query_symbol_name(
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

static iree_string_view_t loom_low_lower_contract_query_function_name(
    const loom_target_contract_query_environment_t* environment) {
  if (!loom_func_like_isa(environment->function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_contract_query_symbol_name(
      environment->module, loom_func_like_callee(environment->function));
}

static iree_string_view_t loom_low_lower_descriptor_matrix_source_key(
    loom_target_contract_descriptor_matrix_source_t source) {
  switch (source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return IREE_SV("vector.mma");
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return IREE_SV("<unknown>");
  }
}

static iree_string_view_t
loom_low_lower_descriptor_matrix_source_constraint_key(
    loom_target_contract_descriptor_matrix_source_t source,
    loom_contract_diagnostic_t diagnostic) {
  const loom_contract_rejection_bits_t bits = diagnostic.rejection_bits;
  if (source == LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA) {
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_ROLE)) {
      return IREE_SV("fragment_roles");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
      return IREE_SV("fragment_shape");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
      return IREE_SV("fragment_schema");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
      return IREE_SV("numeric_type");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
      return IREE_SV("fragment_ownership");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
      return IREE_SV("capability_class");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
      return IREE_SV("request_shape");
    }
  }
  return IREE_SV("matrix_contract_request");
}

static iree_status_t loom_low_lower_descriptor_matrix_make_rejection(
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_diagnostic_t* diagnostic,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  params[0] = loom_param_string(loom_low_lower_contract_query_nonempty(
      environment->bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_low_lower_contract_query_nonempty(
      environment->bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_low_lower_contract_query_nonempty(
      environment->bundle->config->name, IREE_SV("<empty>")));
  params[3] = loom_param_string(
      loom_low_lower_contract_query_function_name(environment));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
  params[5] = loom_param_string(
      loom_low_lower_descriptor_matrix_source_key(rule->source));
  params[6] =
      loom_param_string(loom_low_lower_descriptor_matrix_source_constraint_key(
          rule->source, *diagnostic));
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_TARGET_039_REF,
      .params = params,
      .param_count = 7,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_make_unsupported(
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_diagnostic_t* diagnostic,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_make_rejection(
      environment, rule, source_op, diagnostic, &rejection));
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  out_result->source_rejection_bits = diagnostic->rejection_bits;
  out_result->rejection = rejection;
  return iree_ok_status();
}

iree_status_t loom_low_lower_query_descriptor_matrix_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_descriptor_matrix_t* descriptor_matrix,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (descriptor_matrix->options == NULL || descriptor_matrix->query == NULL) {
    return iree_ok_status();
  }

  loom_contract_request_t request = {0};
  loom_contract_diagnostic_t diagnostic = {0};
  switch (rule->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA: {
      if (!loom_vector_mma_isa(source_op)) {
        return iree_ok_status();
      }
      loom_contract_vector_mma_options_t options = {0};
      IREE_RETURN_IF_ERROR(descriptor_matrix->options(
          descriptor_matrix->user_data, environment, rule, &options));
      if (!loom_contract_request_from_vector_mma_op(
              environment->module, environment->fact_table, source_op, &options,
              &request, &diagnostic)) {
        return loom_low_lower_descriptor_matrix_make_unsupported(
            environment, rule, source_op, &diagnostic, out_result);
      }
      break;
    }
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown descriptor-matrix source");
  }

  return descriptor_matrix->query(descriptor_matrix->user_data, environment,
                                  rule, source_op, &request, out_result);
}

static iree_status_t loom_low_lower_query_target_contract_index(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    const loom_low_lower_rule_match_context_t* match_context,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_index_t* index = options->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_selection = {0};
  uint16_t failed_binding_index = UINT16_MAX;
  uint16_t failed_case_index = UINT16_MAX;
  uint16_t failed_rule_set_index = UINT16_MAX;
  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    if (!loom_target_contract_fragment_queries_target(binding->fragment)) {
      continue;
    }
    if (contract_case->system ==
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
          environment, &options->descriptor_matrix, matrix_rule, source_op,
          out_result));
      if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
        loom_low_lower_contract_query_adopt_case(contract_case, case_index,
                                                 out_result);
        return iree_ok_status();
      }
      continue;
    }
    uint16_t rule_index = UINT16_MAX;
    if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                       &rule_index)) {
      continue;
    }
    const loom_low_lower_rule_set_t* rule_set =
        options->rule_sets.values[binding->rule_set_index];
    loom_low_lower_rule_selection_t selection = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_set_select_rule_range_with_match_context(
            match_context, rule_set, source_op, rule_index, 1, &selection));
    if (selection.rule != NULL) {
      const loom_low_lower_descriptor_ref_t descriptor_ref =
          loom_low_lower_rule_first_descriptor_ref(rule_set, selection.rule);
      const loom_low_descriptor_t* selected_descriptor = NULL;
      if (descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
            match_context, rule_set, descriptor_ref, &selected_descriptor));
        if (selected_descriptor == NULL) {
          const iree_string_view_t key =
              rule_set->descriptor_refs[descriptor_ref].key;
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "generated target-low contract selected missing descriptor "
              "'%.*s'",
              (int)key.size, key.data);
        }
      }
      *out_result = (loom_target_contract_query_result_t){
          .outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL,
          .binding_index = contract_case->binding_index,
          .case_index = case_index,
          .rule_set_index = binding->rule_set_index,
          .rule_index = selection.rule_index,
          .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
          .matched_guard_count = selection.rule->guard_count,
          .selected_descriptor = selected_descriptor,
          .source_rejection_bits = 0,
          .target_rejection_bits = 0,
          .missing_feature_bits = 0,
          .missing_fact_bits = 0,
          .rejection = NULL,
      };
      return iree_ok_status();
    }
    if (selection.has_source_op_span &&
        (failed_rule_set == NULL || selection.matched_guard_count >
                                        failed_selection.matched_guard_count)) {
      failed_rule_set = rule_set;
      failed_selection = selection;
      failed_binding_index = contract_case->binding_index;
      failed_case_index = case_index;
      failed_rule_set_index = binding->rule_set_index;
    }
  }

  if (failed_rule_set == NULL) {
    return iree_ok_status();
  }

  const loom_low_lower_diagnostic_t* diagnostic =
      loom_low_lower_rule_set_selection_diagnostic(failed_rule_set,
                                                   failed_selection);
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_contract_query_make_rejection(
      environment, match_context, failed_rule_set, source_op, diagnostic,
      &rejection));
  *out_result = (loom_target_contract_query_result_t){
      .outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
      .binding_index = failed_binding_index,
      .case_index = failed_case_index,
      .rule_set_index = failed_rule_set_index,
      .rule_index = UINT16_MAX,
      .diagnostic_index = failed_selection.diagnostic_index,
      .matched_guard_count = failed_selection.matched_guard_count,
      .selected_descriptor = NULL,
      .source_rejection_bits = 0,
      .target_rejection_bits = 0,
      .missing_feature_bits = 0,
      .missing_fact_bits = 0,
      .rejection = rejection,
  };
  return iree_ok_status();
}

iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();

  if (options->contract_index == NULL ||
      options->contract_index->case_count == 0) {
    return iree_ok_status();
  }

  loom_symbolic_expr_context_t expression_context;
  loom_symbolic_expr_context_t* expression_context_ptr = NULL;
  if (environment->arena && environment->fact_table) {
    loom_symbolic_expr_context_initialize(
        environment->module, environment->fact_table, environment->arena,
        &expression_context);
    expression_context_ptr = &expression_context;
  }

  const loom_low_lower_rule_match_context_t match_context = {
      .module = environment->module,
      .function = environment->function,
      .bundle = environment->bundle,
      .descriptor_set = environment->descriptor_set,
      .feature_bits = environment->bundle->config->contract_feature_bits,
      .map_value = options->map_value,
      .can_materialize = options->can_materialize,
      .descriptor_ref = options->descriptor_ref,
      .fact_table = environment->fact_table,
      .symbolic_expr_context = expression_context_ptr,
      .flags = LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY,
  };

  return loom_low_lower_query_target_contract_index(
      environment, options, source_op, &match_context, out_result);
}
