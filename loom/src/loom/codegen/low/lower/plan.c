// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/plan.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/function.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/storage.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"

bool loom_low_lower_structured_low_enabled(
    const loom_low_lower_context_t* context) {
  return context->options->control_flow_lowering ==
         LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
}

bool loom_low_lower_supported_structured_source_op(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_low_lower_structured_low_enabled(context)) {
    return false;
  }
  switch (source_op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
      return true;
    default:
      return false;
  }
}

bool loom_low_lower_op_is_structural(const loom_module_t* module,
                                     const loom_op_t* op) {
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits)) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_CFG_SWITCH:
    case LOOM_OP_FUNC_ADDRESS:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_CALL_INDIRECT:
    case LOOM_OP_FUNC_COMPARE_NULL:
    case LOOM_OP_FUNC_IMPORT_RESOLVED:
    case LOOM_OP_FUNC_NULL:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_SCHEDULE_FENCE:
    case LOOM_OP_SCF_YIELD:
      return true;
    default:
      return false;
  }
}

bool loom_low_lower_op_is_source_metadata(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
      return true;
    default:
      return false;
  }
}

bool loom_low_lower_op_uses_policy(const loom_module_t* module,
                                   const loom_op_t* op) {
  return !loom_low_lower_op_is_structural(module, op) &&
         !loom_low_lower_op_is_source_metadata(op->kind);
}

bool loom_low_lower_op_is_discardable_hint(const loom_module_t* module,
                                           const loom_op_t* op) {
  if (op->result_count != 0 || op->region_count != 0 ||
      op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  return iree_any_bit_set(traits, LOOM_TRAIT_HINT);
}

static void loom_low_lower_count_region_plan_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    iree_host_size_t* inout_plan_capacity) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            loom_low_lower_count_region_plan_ops(context, regions[i],
                                                 inout_plan_capacity);
          }
        }
        continue;
      }
      if (loom_low_lower_op_uses_policy(context->module, op)) {
        ++(*inout_plan_capacity);
      }
    }
  }
}

static iree_status_t loom_low_lower_prepare_plan(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  iree_host_size_t plan_capacity = 0;
  loom_low_lower_count_region_plan_ops(context, source_body, &plan_capacity);
  context->lowering.selected_plan_capacity = plan_capacity;
  context->lowering.selected_plan_count = 0;
  context->lowering.selected_plan_emit_index = 0;
  if (plan_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, plan_capacity, sizeof(*context->lowering.selected_plans),
      (void**)&context->lowering.selected_plans));
  if (context->options->table_arena != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->options->table_arena, plan_capacity,
        sizeof(*context->lowering.memory_access_records),
        (void**)&context->lowering.memory_access_records));
    context->lowering.memory_access_record_capacity = plan_capacity;
  }
  return iree_ok_status();
}

static void loom_low_lower_record_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t selected_plan) {
  IREE_ASSERT_LT(context->lowering.selected_plan_count,
                 context->lowering.selected_plan_capacity);
  context->lowering.selected_plans[context->lowering.selected_plan_count++] =
      selected_plan;
}

static void loom_low_lower_record_elided_hint_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .flags = LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_empty(),
               });
}

static iree_status_t loom_low_lower_try_select_op_callback(
    loom_low_lower_context_t* context,
    loom_low_lower_select_op_callback_t callback, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  if (callback.fn == NULL) {
    return iree_ok_status();
  }

  loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, context, source_op, &plan));
  if (loom_low_lower_plan_is_empty(plan)) {
    return iree_ok_status();
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .plan = plan,
               });
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_selected_rule_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t rule_selection,
    const loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  const loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_resolve_emit_program(
      context, rule_set_index, rule_set, rule_selection.rule, &resolved_emits));
  const loom_low_source_memory_access_plan_t* source_memory_access = NULL;
  if (rule_selection.uses_source_memory_access) {
    IREE_ASSERT(source_memory_state != NULL);
    IREE_ASSERT_EQ(source_memory_state->source_op, source_op);
    IREE_ASSERT(source_memory_state->plan_available);
    loom_low_source_memory_access_plan_t* retained_source_memory_access = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_source_memory_access),
        (void**)&retained_source_memory_access));
    *retained_source_memory_access = *source_memory_state->access_plan;
    source_memory_access = retained_source_memory_access;
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
                   .rule_set_index = rule_set_index,
                   .rule_index = rule_selection.rule_index,
                   .rule_set = rule_set,
                   .rule = rule_selection.rule,
                   .resolved_emits = resolved_emits,
                   .source_memory_access = source_memory_access,
                   .plan = loom_low_lower_plan_empty(),
               });
  return iree_ok_status();
}

static bool loom_low_lower_rule_selection_is_better_failure(
    const loom_low_lower_rule_set_t* failed_rule_set,
    loom_low_lower_rule_selection_t failed_rule_selection,
    loom_low_lower_rule_selection_t rule_selection) {
  return rule_selection.has_source_op_span &&
         (failed_rule_set == NULL ||
          (rule_selection.source_memory_compatible &&
           !failed_rule_selection.source_memory_compatible) ||
          (rule_selection.source_memory_compatible ==
               failed_rule_selection.source_memory_compatible &&
           rule_selection.matched_guard_count >
               failed_rule_selection.matched_guard_count));
}

static iree_status_t loom_low_lower_query_environment_from_context(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_target_contract_query_environment_t* out_environment) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_environment = (loom_target_contract_query_environment_t){
      .module = context->module,
      .function = context->source_function,
      .target_facts = context->options->target_facts,
      .descriptor_set = descriptor_set,
      .fact_table = context->lowering.fact_table,
      .value_domain = &context->lowering.value_domain,
      .view_regions = view_regions,
      .arena = &context->function_arena,
      .target_state_allocator =
          {
              .fn = loom_low_lower_contract_query_get_or_allocate_target_state,
              .user_data = context,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_contract_query_rejection(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    return loom_low_lower_emit_error_ref(
        context, source_op, result->rejection->error_ref,
        result->rejection->params, result->rejection->param_count);
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_record_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_contract_request_t* contract_request,
    const loom_target_contract_query_result_t* query_result) {
  loom_low_lower_descriptor_matrix_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  plan_data->source = matrix_rule->source;
  if (query_result->selected_descriptor == NULL) {
    IREE_ASSERT_UNREACHABLE("descriptor-matrix legal query has no descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  plan_data->descriptor.descriptor = query_result->selected_descriptor;
  plan_data->contract_request = *contract_request;
  plan_data->attrs = loom_named_attr_slice_empty();
  plan_data->native_contraction_facts =
      query_result->selected_native_contraction_facts;
  if (plan_data->descriptor.descriptor->immediate_count != 0) {
    if (context->policy->descriptor_matrix.attrs == NULL) {
      IREE_ASSERT_UNREACHABLE("descriptor-matrix policy has no attrs callback");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(context->policy->descriptor_matrix.attrs(
        context->policy->descriptor_matrix.user_data, context, matrix_rule,
        &plan_data->contract_request, plan_data->descriptor.descriptor,
        &plan_data->attrs));
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = query_result->rule_index,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_make(source_op->kind, plan_data),
               });
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op_from_contract_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_rule_set_t** inout_failed_rule_set,
    loom_low_lower_rule_selection_t* inout_failed_rule_selection,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    bool* out_selected) {
  *out_selected = false;
  const loom_target_contract_index_t* index = context->contract_set->index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, /*view_regions=*/NULL, source_memory_state, &match_context);
  bool view_regions_resolved = false;

  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    if (contract_case->system ==
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      loom_target_contract_query_result_t query_result =
          loom_target_contract_query_result_empty();
      loom_contract_request_t contract_request = {0};
      loom_target_contract_query_environment_t environment = {0};
      IREE_RETURN_IF_ERROR(loom_low_lower_query_environment_from_context(
          context, context->descriptor_set, &environment));
      IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
          &environment, &context->policy->descriptor_matrix, matrix_rule,
          source_op, &contract_request, &query_result));
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        query_result.rule_index = contract_case->row_index;
        IREE_RETURN_IF_ERROR(loom_low_lower_record_descriptor_matrix_plan(
            context, source_op, matrix_rule, &contract_request, &query_result));
        *out_selected = true;
        return iree_ok_status();
      }
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
          query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_contract_query_rejection(
            context, source_op, &query_result));
        *out_selected = true;
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
        context->contract_set->rule_sets.values[binding->rule_set_index];
    if (rule_set->source_memory_count != 0 && !view_regions_resolved) {
      IREE_RETURN_IF_ERROR(loom_low_lower_context_view_regions(
          context, &match_context.view_regions));
      view_regions_resolved = true;
    }
    IREE_ASSERT(rule_set->source_memory_count == 0 ||
                source_memory_state->access_plan != NULL);
    loom_low_lower_rule_selection_t rule_selection = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_set_select_rule_range_with_match_context(
            &match_context, rule_set, source_op, rule_index, 1,
            &rule_selection));
    if (rule_selection.rule != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
          context, source_op, binding->rule_set_index, rule_set, rule_selection,
          source_memory_state));
      *out_selected = true;
      return iree_ok_status();
    }
    if (loom_low_lower_rule_selection_is_better_failure(
            *inout_failed_rule_set, *inout_failed_rule_selection,
            rule_selection)) {
      *inout_failed_rule_set = rule_set;
      *inout_failed_rule_selection = rule_selection;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  if (source_op->region_count != 0) {
    if (loom_low_lower_supported_structured_source_op(context, source_op)) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(source_op->region_count),
    };
    return loom_low_lower_emit_target_context_error(context, source_op,
                                                    LOOM_ERR_TARGET_030, params,
                                                    IREE_ARRAYSIZE(params));
  }
  if (loom_low_lower_op_is_structural(context->module, source_op)) {
    return iree_ok_status();
  }
  if (loom_low_lower_op_is_source_metadata(source_op->kind)) {
    return iree_ok_status();
  }

  bool selected_callback = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->preselect_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_rule_selection = {0};
  loom_low_source_memory_access_plan_t source_memory_access;
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &source_memory_access, &source_memory_state);
  bool selected_rule = false;
  if (context->contract_set->index->case_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_op_from_contract_index(
        context, source_op, &failed_rule_set, &failed_rule_selection,
        &source_memory_state, &selected_rule));
    if (selected_rule) {
      return iree_ok_status();
    }
    if (failed_rule_set != NULL) {
      return loom_low_lower_rule_set_emit_selection_failure(
          context, failed_rule_set, source_op, failed_rule_selection,
          &source_memory_state);
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->select_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  if (failed_rule_set != NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(
        context, failed_rule_set, source_op, failed_rule_selection,
        &source_memory_state);
  }
  if (loom_low_lower_op_is_discardable_hint(context->module, source_op)) {
    loom_low_lower_record_elided_hint_plan(context, source_op);
    return iree_ok_status();
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static void loom_low_lower_planning_scope_begin(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = true;
}

static void loom_low_lower_planning_scope_end(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = false;
  iree_arena_reset(&context->planning_arena);
}

static iree_status_t loom_low_lower_plan_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_op_t* block_arg_context_op, bool skip_entry_block_args) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    if (!(skip_entry_block_args && block_index == 0)) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        loom_type_t low_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
            context, block_arg_context_op, block->arg_ids[i], &low_type));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_lower_planning_scope_begin(context);
      iree_status_t status = loom_low_lower_plan_op(context, op);
      loom_low_lower_planning_scope_end(context);
      IREE_RETURN_IF_ERROR(status);
      if (loom_low_lower_context_should_stop(context)) {
        return iree_ok_status();
      }
      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_lower_plan_region(context, regions[i], op,
                                       /*skip_entry_block_args=*/false));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_select_plans(loom_low_lower_context_t* context,
                                          loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(loom_low_lower_prepare_plan(context, source_body));
  return loom_low_lower_plan_region(context, source_body,
                                    context->source_function.op,
                                    /*skip_entry_block_args=*/true);
}
