// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/contract_query.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/lower/contract_selection.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
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
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_diagnostic_params(
        match_context, rule_set, source_op, diagnostic, params));
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
    return loom_string_table_get(&module->strings, symbol->name_id);
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
    const loom_op_t* source_op, iree_string_view_t constraint,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  params[0] = loom_param_string(
      loom_low_lower_contract_query_nonempty(bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_low_lower_contract_query_nonempty(
      bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_low_lower_contract_query_nonempty(
      bundle->config->name, IREE_SV("<empty>")));
  params[3] = loom_param_string(
      loom_low_lower_contract_query_function_name(environment));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
  params[5] = loom_param_string(
      loom_low_lower_descriptor_matrix_source_key(rule->source));
  params[6] = loom_param_string(constraint);
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_TARGET_039_REF,
      .params = params,
      .param_count = 7,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

iree_status_t loom_low_lower_descriptor_matrix_reject(
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, iree_string_view_t constraint,
    loom_contract_rejection_bits_t source_rejection_bits,
    uint32_t target_rejection_bits, uint32_t missing_feature_bits,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_make_rejection(
      environment, rule, source_op, constraint, &rejection));
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  out_result->source_rejection_bits = source_rejection_bits;
  out_result->target_rejection_bits = target_rejection_bits;
  out_result->missing_feature_bits = missing_feature_bits;
  out_result->rejection = rejection;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_make_unsupported(
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_diagnostic_t* diagnostic,
    loom_target_contract_query_result_t* out_result) {
  return loom_low_lower_descriptor_matrix_reject(
      environment, rule, source_op,
      loom_low_lower_descriptor_matrix_source_constraint_key(rule->source,
                                                             *diagnostic),
      diagnostic->rejection_bits, 0, 0, out_result);
}

iree_status_t loom_low_lower_query_descriptor_matrix_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_descriptor_matrix_t* descriptor_matrix,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, loom_contract_request_t* out_request,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (out_request != NULL) {
    *out_request = (loom_contract_request_t){0};
  }
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
      IREE_ASSERT_UNREACHABLE("unknown descriptor-matrix source");
      IREE_BUILTIN_UNREACHABLE();
  }

  if (out_request != NULL) {
    *out_request = request;
  }

  return descriptor_matrix->query(descriptor_matrix->user_data, environment,
                                  rule, source_op, &request, out_result);
}

typedef enum loom_low_lower_contract_case_domain_e {
  LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_LOWERING = 0,
  LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_TARGET_QUERY = 1,
} loom_low_lower_contract_case_domain_t;

static iree_status_t loom_low_lower_match_contract_case(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op, uint16_t case_index,
    loom_low_lower_contract_case_domain_t domain,
    bool include_descriptor_matrix,
    loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_contract_case_match_t* out_match) {
  const loom_target_contract_index_t* index = options->contract_index;
  const loom_target_contract_case_t* contract_case = &index->cases[case_index];
  if (domain == LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_TARGET_QUERY &&
      !loom_low_lower_contract_case_queries_target(index, contract_case)) {
    out_match->kind = LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_SKIP;
    return iree_ok_status();
  }
  const loom_target_contract_binding_t* binding =
      &index->bindings[contract_case->binding_index];
  *out_match = (loom_low_lower_contract_case_match_t){
      .kind = LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_SKIP,
      .selection = {.rule_index = UINT16_MAX},
      .result = loom_target_contract_query_result_empty(),
  };
  if (contract_case->system == LOOM_TARGET_CONTRACT_SYSTEM_UNSUPPORTED) {
    out_match->kind = LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_UNSUPPORTED;
    return iree_ok_status();
  }
  if (contract_case->system == LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
    if (!include_descriptor_matrix) {
      return iree_ok_status();
    }
    out_match->kind = LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_DESCRIPTOR_MATRIX;
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
        &binding->fragment->descriptor_matrices[contract_case->row_index];
    IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
        environment, &options->descriptor_matrix, matrix_rule, source_op,
        &out_match->contract_request, &out_match->result));
    if (out_match->result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
      loom_low_lower_contract_query_adopt_case(contract_case, case_index,
                                               &out_match->result);
    }
    return iree_ok_status();
  }

  uint16_t rule_index = UINT16_MAX;
  if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                     &rule_index)) {
    return iree_ok_status();
  }
  out_match->kind = LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE;
  out_match->rule_set = options->rule_sets.values[binding->rule_set_index];
  match_context->policy_rule_set_ordinal =
      (uint16_t)(binding->rule_set_index + 1u);
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_set_select_rule_range_with_match_context(
          match_context, out_match->rule_set, source_op, rule_index, 1,
          &out_match->selection));
  return iree_ok_status();
}

// Returns true when two source-node rows resolve the same SSA connection.
// Node-local guards are deliberately excluded: resolution establishes the
// connected op once, while each candidate still evaluates its own guards.
static bool loom_low_lower_source_nodes_share_connection(
    const loom_low_lower_source_node_t* lhs,
    const loom_low_lower_source_node_t* rhs) {
  return lhs->relation == rhs->relation &&
         lhs->parent_node_index == rhs->parent_node_index &&
         lhs->source_op_kind == rhs->source_op_kind &&
         lhs->parent_value_ref_index == rhs->parent_value_ref_index &&
         lhs->node_value_ref_index == rhs->node_value_ref_index;
}

static iree_status_t loom_low_lower_visit_contract_cases(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_low_lower_contract_case_iteration_mode_t iteration_mode,
    loom_low_lower_contract_case_domain_t domain,
    loom_low_lower_contract_case_scope_t case_scope,
    loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_contract_case_callback_t callback,
    bool* out_used_candidate_index) {
  if (out_used_candidate_index != NULL) {
    *out_used_candidate_index = false;
  }
  if (options->contract_index == NULL || callback.fn == NULL) {
    return iree_ok_status();
  }
  const loom_target_contract_index_t* index = options->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  loom_low_lower_contract_case_iterator_t iterator;
  const bool used_candidate_index =
      loom_low_lower_contract_case_iterator_initialize(
          environment->module, index, op_entry, source_op,
          environment->vector_lane_projection, iteration_mode, &iterator);
  if (out_used_candidate_index != NULL) {
    *out_used_candidate_index = used_candidate_index;
  }
  const loom_low_lower_rule_set_t* cached_first_source_node_rule_set = NULL;
  const loom_low_lower_source_node_t* cached_first_source_node = NULL;
  const loom_op_t* resolved_first_source_node = NULL;
  uint16_t case_index = UINT16_MAX;
  while (loom_low_lower_contract_case_iterator_next(&iterator, &case_index)) {
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    if (domain == LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_TARGET_QUERY &&
        !loom_low_lower_contract_case_queries_target(index, contract_case)) {
      continue;
    }
    if (case_scope != LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_ALL) {
      uint16_t rule_index = UINT16_MAX;
      const bool has_lower_rule = loom_low_lower_contract_case_lower_rule_index(
          index, contract_case, &rule_index);
      bool is_source_graph = false;
      const loom_low_lower_rule_set_t* rule_set = NULL;
      if (has_lower_rule) {
        const loom_target_contract_binding_t* binding =
            &index->bindings[contract_case->binding_index];
        rule_set = options->rule_sets.values[binding->rule_set_index];
        is_source_graph = loom_low_lower_rule_source_node_count(
                              &rule_set->rules[rule_index]) != 0;
      }
      const bool select_source_graph =
          case_scope == LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SOURCE_GRAPH;
      if (has_lower_rule ? is_source_graph != select_source_graph
                         : select_source_graph) {
        continue;
      }
      if (select_source_graph) {
        const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
        const loom_low_lower_source_node_t* first_source_node =
            &rule_set
                 ->source_nodes[loom_low_lower_rule_source_node_start(rule)];
        if (cached_first_source_node_rule_set != rule_set ||
            !loom_low_lower_source_nodes_share_connection(
                cached_first_source_node, first_source_node)) {
          cached_first_source_node_rule_set = rule_set;
          cached_first_source_node = first_source_node;
          if (!loom_low_lower_rule_resolve_first_source_node(
                  match_context, rule_set, source_op, rule,
                  &resolved_first_source_node)) {
            continue;
          }
        } else if (resolved_first_source_node == NULL) {
          continue;
        }
        match_context->resolved_first_source_node = resolved_first_source_node;
      }
    }
    loom_low_lower_contract_case_match_t match;
    iree_status_t status = loom_low_lower_match_contract_case(
        environment, options, source_op, case_index, domain,
        /*include_descriptor_matrix=*/
        iteration_mode == LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
        match_context, &match);
    match_context->resolved_first_source_node = NULL;
    IREE_RETURN_IF_ERROR(status);
    if (match.kind != LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_SKIP) {
      bool stop = false;
      IREE_RETURN_IF_ERROR(
          callback.fn(callback.user_data, case_index, &match, &stop));
      if (stop) {
        break;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_visit_lowering_contract_cases(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_low_lower_contract_case_iteration_mode_t iteration_mode,
    loom_low_lower_contract_case_scope_t case_scope,
    loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_contract_case_callback_t callback,
    bool* out_used_candidate_index) {
  return loom_low_lower_visit_contract_cases(
      environment, options, source_op, iteration_mode,
      LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_LOWERING, case_scope, match_context,
      callback, out_used_candidate_index);
}

iree_status_t loom_low_lower_visit_target_contract_cases(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_low_lower_contract_case_iteration_mode_t iteration_mode,
    loom_low_lower_contract_case_scope_t case_scope,
    loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_contract_case_callback_t callback,
    bool* out_used_candidate_index) {
  return loom_low_lower_visit_contract_cases(
      environment, options, source_op, iteration_mode,
      LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_TARGET_QUERY, case_scope,
      match_context, callback, out_used_candidate_index);
}

static loom_target_contract_projection_flags_t
loom_low_lower_contract_projection_flags(
    const loom_low_lower_rule_set_t* rule_set, uint16_t rule_index) {
  loom_target_contract_projection_flags_t flags =
      LOOM_TARGET_CONTRACT_PROJECTION_FLAG_NONE;
  if (loom_low_lower_rule_requires_root_producer_closure(rule_set,
                                                         rule_index)) {
    flags |= LOOM_TARGET_CONTRACT_PROJECTION_FLAG_ROOT_PRODUCER_CLOSURE;
  }
  return flags;
}

static iree_status_t loom_low_lower_materialize_target_contract_case(
    const loom_low_lower_contract_query_options_t* options,
    const loom_target_contract_case_t* contract_case, uint16_t case_index,
    const loom_target_contract_binding_t* binding,
    loom_low_lower_rule_match_context_t* match_context, bool apply_accept_rule,
    loom_low_lower_contract_case_match_t* match, bool* out_accepted) {
  if (out_accepted != NULL) {
    *out_accepted = false;
  }
  IREE_ASSERT_EQ(match->kind, LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE);
  IREE_ASSERT(match->selection.rule != NULL);
  if (apply_accept_rule && options->accept_rule.fn != NULL) {
    bool accepted = false;
    IREE_RETURN_IF_ERROR(options->accept_rule.fn(options->accept_rule.user_data,
                                                 &match->selection, &accepted));
    if (!accepted) {
      return iree_ok_status();
    }
  }

  const loom_low_lower_descriptor_ref_t descriptor_ref =
      loom_low_lower_rule_primary_descriptor_ref(match->rule_set,
                                                 match->selection.rule);
  const loom_low_descriptor_t* selected_descriptor = NULL;
  if (descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
        match_context, match->rule_set, descriptor_ref, &selected_descriptor));
    IREE_ASSERT(selected_descriptor != NULL,
                "generated target-low contract selected a missing descriptor");
  }
  match->result = (loom_target_contract_query_result_t){
      .outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL,
      .projection_flags = loom_low_lower_contract_projection_flags(
          match->rule_set, match->selection.rule_index),
      .binding_index = contract_case->binding_index,
      .case_index = case_index,
      .rule_set_index = binding->rule_set_index,
      .rule_index = match->selection.rule_index,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
      .matched_guard_count = match->selection.rule->guard_count,
      .selected_descriptor = selected_descriptor,
      .source_rejection_bits = 0,
      .target_rejection_bits = 0,
      .missing_feature_bits = 0,
      .missing_fact_bits = 0,
      .rejection = NULL,
  };
  if (out_accepted != NULL) {
    *out_accepted = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_query_target_contract_index(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    const loom_low_lower_rule_match_context_t* match_context,
    loom_target_contract_query_result_t* out_result,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_target_contract_index_t* index = options->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  loom_low_lower_rule_match_context_t case_match_context = *match_context;
  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_failure_t best_failure = {0};
  uint16_t failed_binding_index = UINT16_MAX;
  uint16_t failed_case_index = UINT16_MAX;
  uint16_t failed_rule_set_index = UINT16_MAX;
  bool has_unsupported_marker = false;
  uint16_t unsupported_binding_index = UINT16_MAX;
  uint16_t unsupported_case_index = UINT16_MAX;
  loom_low_lower_contract_case_iteration_mode_t iteration_mode =
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES;
  while (true) {
    failed_rule_set = NULL;
    best_failure = (loom_low_lower_rule_failure_t){0};
    failed_binding_index = UINT16_MAX;
    failed_case_index = UINT16_MAX;
    failed_rule_set_index = UINT16_MAX;
    has_unsupported_marker = false;
    unsupported_binding_index = UINT16_MAX;
    unsupported_case_index = UINT16_MAX;
    loom_target_contract_query_result_t unavailable_result =
        loom_target_contract_query_result_empty();
    loom_low_lower_rule_selection_t unavailable_selection = {
        .rule_index = UINT16_MAX,
    };
    uint16_t declined_rule_count = 0;
    loom_low_lower_contract_case_iterator_t iterator;
    const bool used_candidates =
        loom_low_lower_contract_case_iterator_initialize(
            environment->module, index, op_entry, source_op,
            environment->vector_lane_projection, iteration_mode, &iterator);
    uint16_t case_index = UINT16_MAX;
    while (loom_low_lower_contract_case_iterator_next(&iterator, &case_index)) {
      const loom_target_contract_case_t* contract_case =
          &index->cases[case_index];
      const loom_target_contract_binding_t* binding =
          &index->bindings[contract_case->binding_index];
      const bool include_descriptor_matrix =
          iteration_mode != LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL ||
          options->accept_rule.fn != NULL;
      loom_low_lower_contract_case_match_t match;
      IREE_RETURN_IF_ERROR(loom_low_lower_match_contract_case(
          environment, options, source_op, case_index,
          LOOM_LOW_LOWER_CONTRACT_CASE_DOMAIN_TARGET_QUERY,
          include_descriptor_matrix, &case_match_context, &match));
      if (match.kind == LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_SKIP) {
        continue;
      }
      if (match.kind == LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_UNSUPPORTED) {
        if (!has_unsupported_marker) {
          has_unsupported_marker = true;
          unsupported_binding_index = contract_case->binding_index;
          unsupported_case_index = case_index;
        }
        continue;
      }
      if (match.kind == LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_DESCRIPTOR_MATRIX) {
        if (match.result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
          *out_result = match.result;
          return iree_ok_status();
        }
        continue;
      }
      loom_low_lower_rule_primary_descriptor_status_t primary_status =
          LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE;
      if (match.selection.rule != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_query_primary_descriptor(
            &case_match_context, match.rule_set, match.selection.rule,
            &primary_status));
      }
      if (match.selection.rule == NULL ||
          primary_status == LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_MISSING) {
        if (loom_low_lower_rule_failure_is_better(match.selection.failure,
                                                  best_failure)) {
          failed_rule_set = match.rule_set;
          best_failure = match.selection.failure;
          failed_binding_index = contract_case->binding_index;
          failed_case_index = case_index;
          failed_rule_set_index = binding->rule_set_index;
        }
        continue;
      }

      const bool has_unavailable_selection =
          unavailable_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
      if (has_unavailable_selection &&
          !loom_low_lower_rule_selections_have_same_source_graph(
              &unavailable_selection, &match.selection)) {
        continue;
      }
      if (has_unavailable_selection &&
          primary_status ==
              LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_FEATURES_UNAVAILABLE) {
        continue;
      }
      bool accepted = false;
      IREE_RETURN_IF_ERROR(loom_low_lower_materialize_target_contract_case(
          options, contract_case, case_index, binding, &case_match_context,
          /*apply_accept_rule=*/true, &match, &accepted));
      if (!accepted) {
        ++declined_rule_count;
        continue;
      }
      if (primary_status ==
          LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_FEATURES_UNAVAILABLE) {
        if (unavailable_result.outcome ==
            LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
          unavailable_result = match.result;
          unavailable_selection = match.selection;
        }
        continue;
      }
      if (iteration_mode == LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL &&
          options->accept_rule.fn == NULL) {
        IREE_ASSERT_UNREACHABLE(
            "generated contract candidate index omitted a matching rule");
        IREE_BUILTIN_UNREACHABLE();
      }
      *out_result = match.result;
      if (out_selection != NULL) {
        *out_selection = match.selection;
      }
      return iree_ok_status();
    }
    if (unavailable_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      *out_result = unavailable_result;
      if (out_selection != NULL) {
        *out_selection = unavailable_selection;
      }
      return iree_ok_status();
    }
    if (environment->defer_rejection_diagnostics) {
      // The candidate index excludes only impossible matches. A query rule
      // either failed or matched and was declined by the ownership filter;
      // the generated total distinguishes those outcomes without replaying
      // excluded rules just to rank their failure diagnostics.
      *out_result = loom_target_contract_query_result_empty();
      if (has_unsupported_marker ||
          declined_rule_count < op_entry.query_rule_count) {
        out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
      }
      return iree_ok_status();
    }
    if (!used_candidates ||
        iteration_mode == LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL) {
      break;
    }
    // Candidate failures are incomplete; repeat the authored order to retain
    // the exact best rejection when no candidate selected.
    iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL;
  }

  if (failed_rule_set == NULL) {
    if (has_unsupported_marker) {
      *out_result = loom_target_contract_query_result_empty();
      out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
      out_result->binding_index = unsupported_binding_index;
      out_result->case_index = unsupported_case_index;
    }
    return iree_ok_status();
  }

  const loom_low_lower_diagnostic_t* diagnostic =
      loom_low_lower_rule_set_failure_diagnostic(failed_rule_set, best_failure);
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
      .diagnostic_index = best_failure.diagnostic_index,
      .matched_guard_count = best_failure.matched_guard_count,
      .selected_descriptor = NULL,
      .source_rejection_bits = 0,
      .target_rejection_bits = 0,
      .missing_feature_bits = 0,
      .missing_fact_bits = 0,
      .rejection = rejection,
  };
  return iree_ok_status();
}

typedef struct loom_low_lower_contract_match_visit_state_t {
  // Query options owning the composed contract and lower-rule sets.
  const loom_low_lower_contract_query_options_t* options;
  // Match context shared across every generated case for one source op.
  loom_low_lower_rule_match_context_t* match_context;
  // Consumer of fully materialized legal matches.
  loom_low_lower_contract_match_callback_t callback;
} loom_low_lower_contract_match_visit_state_t;

static iree_status_t loom_low_lower_visit_target_contract_match_case(
    void* user_data, uint16_t case_index,
    const loom_low_lower_contract_case_match_t* case_match, bool* out_stop) {
  *out_stop = false;
  loom_low_lower_contract_match_visit_state_t* state =
      (loom_low_lower_contract_match_visit_state_t*)user_data;
  if (case_match->kind ==
      LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_DESCRIPTOR_MATRIX) {
    if (case_match->result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      const loom_low_lower_rule_selection_t selection = {
          .rule_index = UINT16_MAX,
      };
      return state->callback.fn(
          state->callback.user_data, &case_match->result,
          /*rule_set=*/NULL, &selection,
          LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE);
    }
    return iree_ok_status();
  }
  if (case_match->kind != LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE ||
      case_match->selection.rule == NULL) {
    return iree_ok_status();
  }

  loom_low_lower_rule_primary_descriptor_status_t primary_status =
      LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_query_primary_descriptor(
      state->match_context, case_match->rule_set, case_match->selection.rule,
      &primary_status));
  if (primary_status == LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_MISSING) {
    return iree_ok_status();
  }

  const loom_target_contract_index_t* index = state->options->contract_index;
  const loom_target_contract_case_t* contract_case = &index->cases[case_index];
  const loom_target_contract_binding_t* binding =
      &index->bindings[contract_case->binding_index];
  loom_low_lower_contract_case_match_t match = *case_match;
  IREE_RETURN_IF_ERROR(loom_low_lower_materialize_target_contract_case(
      state->options, contract_case, case_index, binding, state->match_context,
      /*apply_accept_rule=*/false, &match, /*out_accepted=*/NULL));
  return state->callback.fn(state->callback.user_data, &match.result,
                            match.rule_set, &match.selection, primary_status);
}

typedef struct loom_low_lower_contract_query_match_state_t {
  // Optional symbolic-expression context used when no view analysis exists.
  loom_symbolic_expr_context_t expression_context;
  // Canonical source-memory plan shared by every candidate for one root.
  loom_low_source_memory_access_plan_t source_memory_access;
  // Read-only fragment geometry shared by every candidate for one root.
  loom_low_source_memory_axis_byte_stride_t
      axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Lazy source-memory matching state shared by every candidate for one root.
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  // Fully initialized rule matching context borrowing the fields above.
  loom_low_lower_rule_match_context_t match_context;
} loom_low_lower_contract_query_match_state_t;

static void loom_low_lower_contract_query_match_state_initialize(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_low_lower_contract_query_match_state_t* out_state) {
  loom_symbolic_expr_context_t* expression_context = NULL;
  if (environment->view_regions != NULL) {
    expression_context = environment->view_regions->expression_context;
  } else if (environment->arena && environment->fact_table) {
    loom_symbolic_expr_context_initialize(
        environment->module, environment->value_domain, environment->fact_table,
        environment->arena, &out_state->expression_context);
    expression_context = &out_state->expression_context;
  }
  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &out_state->source_memory_access,
      &out_state->source_memory_state);
  if (loom_vector_memory_op_footprint_kind(environment->module, source_op) ==
      LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT) {
    out_state->source_memory_state.axis_byte_strides =
        out_state->axis_byte_strides;
  }
  out_state->match_context = (loom_low_lower_rule_match_context_t){
      .module = environment->module,
      .function = environment->function,
      .bundle = bundle,
      .descriptor_set = environment->descriptor_set,
      .feature_bits = bundle->config->contract_feature_bits,
      .vector_lane_projection = environment->vector_lane_projection,
      .map_value = options->map_value,
      .can_materialize = options->can_materialize,
      .descriptor_ref = options->descriptor_ref,
      .fact_table = environment->fact_table,
      .view_regions = environment->view_regions,
      .function_analysis = environment->function_analysis,
      .source_memory_state = &out_state->source_memory_state,
      .symbolic_expr_context = expression_context,
      .flags = LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY,
  };
}

iree_status_t loom_low_lower_visit_target_contract_matches(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op, loom_low_lower_contract_case_scope_t case_scope,
    loom_low_lower_contract_match_callback_t callback) {
  if (options->contract_index == NULL || callback.fn == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_contract_query_match_state_t state;
  loom_low_lower_contract_query_match_state_initialize(environment, options,
                                                       source_op, &state);
  loom_low_lower_contract_match_visit_state_t visit_state = {
      .options = options,
      .match_context = &state.match_context,
      .callback = callback,
  };
  return loom_low_lower_visit_target_contract_cases(
      environment, options, source_op,
      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, case_scope,
      &state.match_context,
      (loom_low_lower_contract_case_callback_t){
          .fn = loom_low_lower_visit_target_contract_match_case,
          .user_data = &visit_state,
      },
      /*out_used_candidate_index=*/NULL);
}

iree_status_t loom_low_lower_query_target_contract_with_selection(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op, loom_target_contract_query_result_t* out_result,
    loom_low_lower_rule_selection_t* out_selection) {
  *out_result = loom_target_contract_query_result_empty();
  if (out_selection != NULL) {
    *out_selection = (loom_low_lower_rule_selection_t){
        .rule_index = UINT16_MAX,
    };
  }

  if (options->contract_index == NULL ||
      options->contract_index->case_count == 0) {
    return iree_ok_status();
  }

  loom_low_lower_contract_query_match_state_t state;
  loom_low_lower_contract_query_match_state_initialize(environment, options,
                                                       source_op, &state);

  return loom_low_lower_query_target_contract_index(
      environment, options, source_op, &state.match_context, out_result,
      out_selection);
}

iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  return loom_low_lower_query_target_contract_with_selection(
      environment, options, source_op, out_result, /*out_selection=*/NULL);
}
