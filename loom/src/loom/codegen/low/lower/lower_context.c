// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/memory_access_ir.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_lower_target_state_record_t {
  // Target-owned static key identifying this function-local state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the lowering arena.
  void* data;
} loom_low_lower_target_state_record_t;

typedef struct loom_low_lower_module_target_state_record_t {
  // Target-owned static key identifying this module-scope state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the module-state arena.
  void* data;
} loom_low_lower_module_target_state_record_t;

struct loom_low_lower_module_state_t {
  // Arena used for module-scope target state records and payloads.
  iree_arena_allocator_t* arena;
  // Module-scope target state records keyed by target-owned static storage.
  loom_low_lower_module_target_state_record_t* target_state_records;
  // Number of populated target_state_records entries.
  iree_host_size_t target_state_record_count;
  // Number of allocated target_state_records entries.
  iree_host_size_t target_state_record_capacity;
};

static iree_string_view_t loom_low_lower_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_symbol_name(
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

iree_string_view_t loom_low_lower_context_function_name(
    const loom_low_lower_context_t* context) {
  if (!loom_func_like_isa(context->source_function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_symbol_name(
      context->module, loom_func_like_callee(context->source_function));
}

static iree_string_view_t loom_low_lower_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_nonempty(bundle->name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_lower_context_target_key(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_target_key(loom_low_lower_context_bundle(context));
}

static iree_string_view_t loom_low_lower_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_nonempty(bundle->export_plan->name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_lower_context_export_name(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_export_name(loom_low_lower_context_bundle(context));
}

static iree_string_view_t loom_low_lower_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_lower_context_config_key(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_config_key(loom_low_lower_context_bundle(context));
}

bool loom_low_lower_context_should_stop(
    const loom_low_lower_context_t* context) {
  return context->options->max_errors != 0 &&
         context->result->error_count >= context->options->max_errors;
}

static iree_status_t loom_low_lower_emit(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         const loom_error_def_t* error,
                                         const loom_diagnostic_param_t* params,
                                         iree_host_size_t param_count) {
  if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_ERROR) {
    if (loom_low_lower_context_should_stop(context)) {
      return iree_ok_status();
    }
    ++context->result->error_count;
  } else if (loom_error_def_severity(error) == LOOM_DIAGNOSTIC_REMARK) {
    ++context->result->remark_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = source_op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(context->options->emitter, &emission);
}

iree_status_t loom_low_lower_emit_source_type_unsupported(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_type_t actual_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_type(actual_type),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_033, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_lower_emit_register_width_relation_unsupported(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t actual_type, uint32_t result_unit_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(actual_type),
      loom_param_u32(loom_low_register_type_unit_count(actual_type)),
      loom_param_u32(result_unit_count),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_066, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_lower_emit_branch_constraint(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t branch_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(branch_constraint),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_034, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_lower_emit_branch_condition_type_unsupported(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t actual_type, iree_string_view_t type_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(actual_type),
      loom_param_string(type_constraint),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_035, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_lower_emit_error_ref(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_error_ref_t error_ref, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_error_def_t* error = loom_error_catalog_lookup_ref(
      context->options->policy->error_catalog, error_ref);
  IREE_ASSERT(error != NULL);
  return loom_low_lower_emit(context, source_op, error, params, param_count);
}

static void loom_low_lower_make_target_context_params(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  params[0] = loom_param_string(loom_low_lower_target_key(bundle));
  params[1] = loom_param_string(loom_low_lower_export_name(bundle));
  params[2] = loom_param_string(loom_low_lower_config_key(bundle));
  params[3] = loom_param_string(loom_low_lower_context_function_name(context));
  params[4] = loom_param_string(loom_op_name(context->module, source_op));
}

iree_status_t loom_low_lower_emit_target_context_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* extra_params,
    iree_host_size_t extra_param_count) {
  IREE_ASSERT_LE(extra_param_count, 4);
  loom_diagnostic_param_t params[LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT + 4];
  loom_low_lower_make_target_context_params(context, source_op, params);
  for (iree_host_size_t i = 0; i < extra_param_count; ++i) {
    params[LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT + i] = extra_params[i];
  }
  return loom_low_lower_emit(
      context, source_op, error, params,
      LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT + extra_param_count);
}

iree_status_t loom_low_lower_emit_no_target_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_001, /*extra_params=*/NULL,
      /*extra_param_count=*/0);
}

loom_module_t* loom_low_lower_context_module(
    loom_low_lower_context_t* context) {
  return context->module;
}

loom_builder_t* loom_low_lower_context_builder(
    loom_low_lower_context_t* context) {
  return &context->builder;
}

loom_func_like_t loom_low_lower_context_source_function(
    const loom_low_lower_context_t* context) {
  return context->source_function;
}

bool loom_low_lower_context_source_is_retained(
    const loom_low_lower_context_t* context) {
  const loom_symbol_ref_t source_ref =
      loom_func_like_callee(context->source_function);
  return loom_symbol_ref_is_valid(source_ref) && source_ref.module_id == 0 &&
         source_ref.symbol_id < context->module->symbols.count &&
         iree_any_bit_set(
             context->module->symbols.entries[source_ref.symbol_id].flags,
             LOOM_SYMBOL_FLAG_RETAIN);
}

uint16_t loom_low_lower_source_memory_root_argument_index(
    const loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_plan) {
  if (context == NULL || source_plan == NULL ||
      source_plan->root_value_id == LOOM_VALUE_ID_INVALID ||
      source_plan->root_value_id >= context->module->values.count ||
      !loom_func_like_isa(context->source_function)) {
    return UINT16_MAX;
  }
  loom_region_t* body = loom_func_like_body(context->source_function);
  if (body == NULL || body->block_count == 0) {
    return UINT16_MAX;
  }
  const loom_value_t* root_value =
      loom_module_value(context->module, source_plan->root_value_id);
  if (!loom_value_is_block_arg(root_value) ||
      loom_value_def_block(root_value) != loom_region_const_entry_block(body)) {
    return UINT16_MAX;
  }
  return loom_value_def_index(root_value);
}

const loom_low_lower_abi_argument_t* loom_low_lower_context_argument_map(
    const loom_low_lower_context_t* context, uint16_t* out_argument_count) {
  IREE_ASSERT_ARGUMENT(out_argument_count);
  *out_argument_count = context->lowering.argument_map_count;
  return context->lowering.argument_map;
}

const loom_local_value_domain_t* loom_low_lower_context_value_domain(
    const loom_low_lower_context_t* context) {
  return &context->lowering.value_domain;
}

loom_op_t* loom_low_lower_context_low_function(
    const loom_low_lower_context_t* context) {
  return context->low_func_op;
}

uint32_t loom_low_lower_context_error_count(
    const loom_low_lower_context_t* context) {
  return context->result->error_count;
}

loom_target_low_legality_diagnostic_flags_t
loom_low_lower_context_diagnostic_flags(
    const loom_low_lower_context_t* context) {
  return context->options->legality_diagnostic_flags;
}

bool loom_low_lower_context_wants_report_rows(
    const loom_low_lower_context_t* context) {
  return !iree_allocator_is_null(context->options->report_allocator);
}

const loom_target_bundle_t* loom_low_lower_context_bundle(
    const loom_low_lower_context_t* context) {
  return loom_target_facts_bundle(context->options->target_facts);
}

const loom_target_facts_t* loom_low_lower_context_target_facts(
    const loom_low_lower_context_t* context) {
  return context->options->target_facts;
}

const loom_low_descriptor_set_t* loom_low_lower_context_descriptor_set(
    const loom_low_lower_context_t* context) {
  return context->descriptor_set;
}

const loom_value_fact_table_t* loom_low_lower_context_fact_table(
    const loom_low_lower_context_t* context) {
  return context->lowering.fact_table;
}

loom_condition_query_t* loom_low_lower_context_condition_query(
    loom_low_lower_context_t* context) {
  return &context->lowering.condition_query;
}

loom_symbolic_expr_context_t* loom_low_lower_context_symbolic_expr_context(
    loom_low_lower_context_t* context) {
  loom_low_lower_function_analysis_t* analysis =
      &context->lowering.function_analysis;
  if (analysis->phase < LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EXPRESSIONS) {
    loom_symbolic_expr_context_initialize(
        loom_low_lower_context_module(context),
        loom_low_lower_context_fact_table(context), &context->function_arena,
        &analysis->expression_context);
    analysis->phase = LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EXPRESSIONS;
  }
  return &analysis->expression_context;
}

loom_sanitizer_reporting_mode_t loom_low_lower_context_sanitizer_reporting_mode(
    const loom_low_lower_context_t* context) {
  return context->options->sanitizer_reporting_mode;
}

iree_status_t loom_low_lower_context_view_regions(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t** out_view_regions) {
  *out_view_regions = NULL;
  loom_low_lower_function_analysis_t* analysis =
      &context->lowering.function_analysis;
  if (analysis->phase < LOOM_LOW_LOWER_FUNCTION_ANALYSIS_VIEW_REGIONS) {
    loom_symbolic_expr_context_t* expression_context =
        loom_low_lower_context_symbolic_expr_context(context);
    IREE_RETURN_IF_ERROR(loom_view_region_table_initialize(
        &context->lowering.value_domain, expression_context,
        &analysis->view_regions));
    IREE_RETURN_IF_ERROR(
        loom_view_region_table_analyze(&analysis->view_regions));
    analysis->phase = LOOM_LOW_LOWER_FUNCTION_ANALYSIS_VIEW_REGIONS;
  }
  *out_view_regions = &analysis->view_regions;
  return iree_ok_status();
}

iree_host_size_t loom_low_lower_context_selected_plan_count(
    const loom_low_lower_context_t* context) {
  return context->lowering.selected_plan_count;
}

loom_low_lower_selected_plan_view_t loom_low_lower_context_selected_plan_view(
    const loom_low_lower_context_t* context, iree_host_size_t index) {
  IREE_ASSERT_LT(index, context->lowering.selected_plan_count);
  return (loom_low_lower_selected_plan_view_t){
      .source_op = context->lowering.selected_plans[index].source_op,
      .plan = context->lowering.selected_plans[index].plan,
      .elided = iree_any_bit_set(context->lowering.selected_plans[index].flags,
                                 LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED),
  };
}

iree_arena_allocator_t* loom_low_lower_context_function_arena(
    loom_low_lower_context_t* context) {
  return &context->function_arena;
}

iree_arena_allocator_t* loom_low_lower_context_emission_arena(
    loom_low_lower_context_t* context) {
  IREE_ASSERT(context->emission_arena_active,
              "emission storage requested outside low-IR construction");
  return &context->emission_arena;
}

loom_low_lower_module_state_t* loom_low_lower_context_module_state(
    loom_low_lower_context_t* context) {
  return context->module_state;
}

iree_status_t loom_low_lower_allocate_function_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(&context->function_arena, count,
                                   element_size, out_ptr);
}

iree_status_t loom_low_lower_allocate_planning_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(context->planning_arena_active,
              "planning storage requested outside source-op selection");
  return iree_arena_allocate_array(&context->planning_arena, count,
                                   element_size, out_ptr);
}

iree_status_t loom_low_lower_allocate_emission_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(context->emission_arena_active,
              "emission storage requested outside low-IR construction");
  return iree_arena_allocate_array(&context->emission_arena, count,
                                   element_size, out_ptr);
}

iree_status_t loom_low_lower_allocate_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data) {
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  return loom_low_lower_allocate_function_array(context, 1, data_length,
                                                out_data);
}

iree_status_t loom_low_lower_module_state_create(
    iree_arena_allocator_t* arena,
    loom_low_lower_module_state_t** out_module_state) {
  IREE_ASSERT_ARGUMENT(out_module_state);
  *out_module_state = NULL;
  loom_low_lower_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*module_state), (void**)&module_state));
  memset(module_state, 0, sizeof(*module_state));
  module_state->arena = arena;
  *out_module_state = module_state;
  return iree_ok_status();
}

iree_status_t loom_low_lower_module_state_get_or_allocate(
    loom_low_lower_module_state_t* module_state, const void* key,
    iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT(key != NULL);
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  for (iree_host_size_t i = 0; i < module_state->target_state_record_count;
       ++i) {
    loom_low_lower_module_target_state_record_t* record =
        &module_state->target_state_records[i];
    if (record->key != key) continue;
    IREE_ASSERT_EQ(record->data_length, data_length);
    *out_data = record->data;
    return iree_ok_status();
  }

  if (module_state->target_state_record_count ==
      module_state->target_state_record_capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(module_state->target_state_record_count, 1,
                                    &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        module_state->arena, module_state->target_state_record_count,
        minimum_capacity, sizeof(*module_state->target_state_records),
        &module_state->target_state_record_capacity,
        (void**)&module_state->target_state_records));
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(module_state->arena, data_length, &data));
  memset(data, 0, data_length);
  const iree_host_size_t record_index =
      module_state->target_state_record_count++;
  loom_low_lower_module_target_state_record_t* record =
      &module_state->target_state_records[record_index];
  *record = (loom_low_lower_module_target_state_record_t){
      .key = key,
      .data_length = data_length,
      .data = data,
  };
  *out_data = data;
  return iree_ok_status();
}

iree_status_t loom_low_lower_module_state_allocate(
    loom_low_lower_module_state_t* module_state, iree_host_size_t byte_length,
    void** out_ptr) {
  *out_ptr = NULL;
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  if (byte_length == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate(module_state->arena, byte_length, out_ptr);
}

iree_status_t loom_low_lower_module_state_allocate_array(
    loom_low_lower_module_state_t* module_state, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  return iree_arena_allocate_array(module_state->arena, count, element_size,
                                   out_ptr);
}

iree_status_t loom_low_lower_get_or_allocate_target_state(
    loom_low_lower_context_t* context, const void* key,
    iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT(key != NULL);
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  for (iree_host_size_t i = 0; i < context->lowering.target_state_record_count;
       ++i) {
    loom_low_lower_target_state_record_t* record =
        &context->lowering.target_state_records[i];
    if (record->key != key) continue;
    IREE_ASSERT_EQ(record->data_length, data_length);
    *out_data = record->data;
    return iree_ok_status();
  }

  if (context->lowering.target_state_record_count ==
      context->lowering.target_state_record_capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(context->lowering.target_state_record_count,
                                    1, &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &context->function_arena, context->lowering.target_state_record_count,
        minimum_capacity, sizeof(*context->lowering.target_state_records),
        &context->lowering.target_state_record_capacity,
        (void**)&context->lowering.target_state_records));
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_function_array(context, 1, data_length, &data));
  memset(data, 0, data_length);
  const iree_host_size_t record_index =
      context->lowering.target_state_record_count++;
  loom_low_lower_target_state_record_t* record =
      &context->lowering.target_state_records[record_index];
  *record = (loom_low_lower_target_state_record_t){
      .key = key,
      .data_length = data_length,
      .data = data,
  };
  *out_data = data;
  return iree_ok_status();
}

iree_status_t loom_low_lower_contract_query_get_or_allocate_target_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data) {
  return loom_low_lower_get_or_allocate_target_state(
      (loom_low_lower_context_t*)user_data, key, data_length, out_data);
}

iree_status_t loom_low_lower_get_or_allocate_module_target_state(
    loom_low_lower_context_t* context, const void* key,
    iree_host_size_t data_length, void** out_data) {
  return loom_low_lower_module_state_get_or_allocate(context->module_state, key,
                                                     data_length, out_data);
}

static iree_status_t loom_low_lower_replace_value_binding(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id) {
  IREE_ASSERT_LT(low_value_id, context->module->values.count);
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  IREE_ASSERT_NE(context->lowering.value_map[source_ordinal],
                 LOOM_VALUE_ID_INVALID);
  context->lowering.value_map[source_ordinal] = low_value_id;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

loom_region_t* loom_low_lower_context_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
}

void loom_low_lower_context_emission_scope_begin(
    loom_low_lower_context_t* context) {
  context->emission_arena_active = true;
}

void loom_low_lower_context_emission_scope_end(
    loom_low_lower_context_t* context) {
  context->emission_arena_active = false;
  iree_arena_reset(&context->emission_arena);
}

iree_status_t loom_low_lower_interpose_entry_block(
    loom_low_lower_context_t* context, const loom_type_t* target_arg_types,
    uint16_t target_arg_count,
    loom_low_lower_entry_interposition_t* out_interposition) {
  IREE_ASSERT(target_arg_count == 0 || target_arg_types != NULL);
  IREE_ASSERT(out_interposition != NULL);
  *out_interposition = (loom_low_lower_entry_interposition_t){0};

  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_ASSERT(low_body != NULL);
  loom_block_t* setup_block = loom_region_entry_block(low_body);
  if (context->builder.ip.block != setup_block ||
      context->builder.ip.before_op != NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source-to-low entry interposition requires the end of the physical "
        "entry block");
  }
  if (context->lowering.block_map == NULL ||
      context->lowering.block_map[0] != setup_block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source-to-low entry block has already been interposed");
  }

  loom_block_t* body_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      context->module, low_body, (uint16_t)(setup_block->region_index + 1),
      &body_block));
  body_block->label_id = setup_block->label_id;
  setup_block->label_id = LOOM_STRING_ID_INVALID;

  const uint16_t forwarded_arg_count = setup_block->arg_count;
  loom_value_id_t* forwarded_args = NULL;
  if (forwarded_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, forwarded_arg_count, sizeof(*forwarded_args),
        (void**)&forwarded_args));
  }
  for (uint16_t i = 0; i < forwarded_arg_count; ++i) {
    const loom_value_id_t setup_arg = loom_block_arg_id(setup_block, i);
    forwarded_args[i] = setup_arg;
    loom_value_id_t body_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        &context->builder, body_block,
        loom_module_value_type(context->module, setup_arg), &body_arg));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(context->module, setup_arg, body_arg));
  }

  loom_value_id_t* target_args = NULL;
  if (target_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, target_arg_count, sizeof(*target_args), (void**)&target_args));
  }
  for (uint16_t i = 0; i < target_arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        &context->builder, body_block, target_arg_types[i], &target_args[i]));
  }

  loom_region_t* source_body = loom_func_like_body(context->source_function);
  IREE_ASSERT(source_body != NULL);
  loom_block_t* source_entry_block = loom_region_entry_block(source_body);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &source_argument_count);
  IREE_ASSERT_EQ(source_entry_block->arg_count, source_argument_count);
  uint16_t direct_argument_index = 0;
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      continue;
    }
    IREE_ASSERT_LT(direct_argument_index, forwarded_arg_count);
    IREE_RETURN_IF_ERROR(loom_low_lower_replace_value_binding(
        context, source_arguments[i],
        loom_block_arg_id(body_block, direct_argument_index)));
    ++direct_argument_index;
  }
  IREE_ASSERT_EQ(direct_argument_index, forwarded_arg_count);

  context->lowering.block_map[0] = body_block;
  *out_interposition = (loom_low_lower_entry_interposition_t){
      .setup_block = setup_block,
      .body_block = body_block,
      .forwarded_args = forwarded_args,
      .forwarded_arg_count = forwarded_arg_count,
      .target_args = target_args,
      .target_arg_count = target_arg_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_lower_append_low_block(loom_low_lower_context_t* context,
                                              loom_block_t** out_block) {
  *out_block = NULL;
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_ASSERT(low_body != NULL);
  return loom_region_append_block(context->module, low_body, out_block);
}

static uint16_t loom_low_lower_source_block_index(
    loom_low_lower_context_t* context, const loom_block_t* source_block) {
  loom_region_t* source_body = loom_func_like_body(context->source_function);
  uint16_t source_index = 0;
  IREE_ASSERT(source_body != NULL);
  const bool found_source_index =
      loom_region_try_block_index(source_body, source_block, &source_index);
  IREE_ASSERT(found_source_index);
  (void)found_source_index;
  return source_index;
}

iree_status_t loom_low_lower_lookup_successor_dest(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint16_t successor_index, loom_block_t** out_low_dest) {
  *out_low_dest = NULL;
  IREE_ASSERT(source_terminator != NULL);
  IREE_ASSERT_LT(successor_index, source_terminator->successor_count);

  const uint16_t source_index = loom_low_lower_source_block_index(
      context, source_terminator->parent_block);
  const loom_low_lower_successor_interpositions_t* interpositions =
      &context->lowering.successor_interpositions[source_index];
  if (interpositions->low_dests != NULL &&
      interpositions->low_dests[successor_index] != NULL) {
    *out_low_dest = interpositions->low_dests[successor_index];
    return iree_ok_status();
  }

  loom_block_t* const* source_successors =
      loom_op_const_successors(source_terminator);
  return loom_low_lower_lookup_block(
      context, source_successors[successor_index], out_low_dest);
}

iree_status_t loom_low_lower_materialize_structural_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_host_size_t operand_index, loom_value_id_t source_value_id,
    loom_type_t required_low_type, loom_value_id_t* inout_low_value_id) {
  if (context->policy->materialize_structural_operand.fn != NULL) {
    loom_value_id_t materialized_low_value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(context->policy->materialize_structural_operand.fn(
        context->policy->materialize_structural_operand.user_data, context,
        source_op, operand_index, source_value_id, *inout_low_value_id,
        required_low_type, &materialized_low_value_id));
    *inout_low_value_id = materialized_low_value_id;
  }

  const loom_type_t materialized_type =
      loom_module_value_type(context->module, *inout_low_value_id);
  IREE_ASSERT(loom_type_equal(materialized_type, required_low_type),
              "lowering policy materialized a structural operand with the "
              "wrong type");
  return iree_ok_status();
}

iree_status_t loom_low_lower_remap_successor_args(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint16_t successor_index, loom_block_t* low_dest,
    const loom_value_id_t* source_args, uint16_t source_arg_count,
    loom_value_id_t** out_low_args) {
  *out_low_args = NULL;
  if (source_arg_count == 0) {
    IREE_ASSERT_EQ(low_dest->arg_count, 0);
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(source_arg_count, low_dest->arg_count);

  loom_value_id_t* low_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_arg_count, sizeof(*low_args), (void**)&low_args));
  for (uint16_t i = 0; i < source_arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_args[i], &low_args[i]));

    const loom_type_t required_type =
        loom_block_arg_type(context->module, low_dest, i);
    const loom_type_t actual_type =
        loom_module_value_type(context->module, low_args[i]);
    if (!loom_type_equal(actual_type, required_type) &&
        context->policy->materialize_branch_arg.fn != NULL) {
      IREE_RETURN_IF_ERROR(context->policy->materialize_branch_arg.fn(
          context->policy->materialize_branch_arg.user_data, context,
          source_terminator, successor_index, i, source_args[i], low_args[i],
          required_type, &low_args[i]));
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_structural_operand(
        context, source_terminator, i, source_args[i], required_type,
        &low_args[i]));
  }
  *out_low_args = low_args;
  return iree_ok_status();
}

iree_status_t loom_low_lower_interpose_successor_dest(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint16_t successor_index, loom_block_t* interposed_low_block,
    loom_block_t** out_previous_low_dest) {
  *out_previous_low_dest = NULL;
  IREE_ASSERT(source_terminator != NULL);
  IREE_ASSERT_LT(successor_index, source_terminator->successor_count);
  IREE_ASSERT(interposed_low_block != NULL);
  if (loom_cfg_br_isa(source_terminator)) {
    IREE_ASSERT_EQ(loom_cfg_br_args(source_terminator).count,
                   interposed_low_block->arg_count);
  } else {
    IREE_ASSERT_EQ(interposed_low_block->arg_count, 0);
  }

  loom_block_t* previous_low_dest = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
      context, source_terminator, successor_index, &previous_low_dest));

  const uint16_t source_index = loom_low_lower_source_block_index(
      context, source_terminator->parent_block);
  loom_low_lower_successor_interpositions_t* interpositions =
      &context->lowering.successor_interpositions[source_index];
  if (interpositions->low_dests == NULL) {
    IREE_ASSERT_EQ(interpositions->low_dest_count, 0);
    interpositions->low_dest_count = source_terminator->successor_count;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, interpositions->low_dest_count,
        sizeof(*interpositions->low_dests),
        (void**)&interpositions->low_dests));
    memset(interpositions->low_dests, 0,
           interpositions->low_dest_count * sizeof(*interpositions->low_dests));
  }
  IREE_ASSERT_EQ(interpositions->low_dest_count,
                 source_terminator->successor_count);
  interpositions->low_dests[successor_index] = interposed_low_block;
  *out_previous_low_dest = previous_low_dest;
  return iree_ok_status();
}

iree_status_t loom_low_lower_set_branch_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_terminator,
                                             loom_low_lower_plan_t plan) {
  IREE_ASSERT(source_terminator != NULL);
  IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(plan));
  const uint16_t source_index = loom_low_lower_source_block_index(
      context, source_terminator->parent_block);
  loom_low_lower_plan_t* branch_plan =
      &context->lowering.branch_plans[source_index];
  IREE_ASSERT(loom_low_lower_plan_is_empty(*branch_plan));
  *branch_plan = plan;
  return iree_ok_status();
}

bool loom_low_lower_lookup_branch_plan(loom_low_lower_context_t* context,
                                       const loom_op_t* source_terminator,
                                       loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (source_terminator == NULL || context->lowering.branch_plans == NULL) {
    return false;
  }
  const uint16_t source_index = loom_low_lower_source_block_index(
      context, source_terminator->parent_block);
  *out_plan = context->lowering.branch_plans[source_index];
  return !loom_low_lower_plan_is_empty(*out_plan);
}

iree_status_t loom_low_lower_lookup_block(loom_low_lower_context_t* context,
                                          const loom_block_t* source_block,
                                          loom_block_t** out_low_block) {
  *out_low_block = NULL;
  const uint16_t source_index =
      loom_low_lower_source_block_index(context, source_block);
  IREE_ASSERT(context->lowering.block_map[source_index] != NULL);
  *out_low_block = context->lowering.block_map[source_index];
  return iree_ok_status();
}

iree_status_t loom_low_lower_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_build_register_type(context->descriptor_set, reg_class_id,
                                      unit_count, out_type);
}

iree_status_t loom_low_lower_make_typed_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t value_type, loom_type_t* out_type) {
  return loom_low_build_typed_register_type(
      loom_low_lower_context_module(context), context->descriptor_set,
      reg_class_id, unit_count, value_type, out_type);
}

iree_status_t loom_low_lower_emit_resolved_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  return loom_low_build_resolved_descriptor_op(
      &context->builder, context->descriptor_set, descriptor->descriptor,
      operands, operand_count, attrs, result_types, result_count, tied_results,
      tied_result_count, location, out_op);
}

iree_status_t loom_low_lower_emit_resolved_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_low_build_resolved_descriptor_const(
      &context->builder, context->descriptor_set, descriptor->descriptor, attrs,
      result_type, location, out_op);
}

iree_status_t loom_low_lower_record_memory_access_summary(
    loom_low_lower_context_t* context, loom_op_t* low_op,
    const loom_low_memory_access_summary_t* summary,
    loom_low_lower_memory_access_record_flags_t flags) {
  if (iree_any_bit_set(flags, ~LOOM_LOW_LOWER_MEMORY_ACCESS_RECORD_PRESERVE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported low memory access record flags");
  }
  if (iree_any_bit_set(flags, LOOM_LOW_LOWER_MEMORY_ACCESS_RECORD_PRESERVE)) {
    IREE_RETURN_IF_ERROR(
        loom_low_memory_access_ir_attach(context->module, low_op, summary));
  }
  if (context->options->table_arena == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(context->lowering.memory_access_record_count,
                 context->lowering.memory_access_record_capacity);

  loom_low_memory_access_record_t* record =
      &context->lowering.memory_access_records
           [context->lowering.memory_access_record_count++];
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  uint16_t block_index = LOOM_BLOCK_REGION_INDEX_INVALID;
  IREE_ASSERT(low_body != NULL);
  const bool found_block_index =
      loom_region_try_block_index(low_body, low_op->parent_block, &block_index);
  IREE_ASSERT(found_block_index);
  (void)found_block_index;
  *record = (loom_low_memory_access_record_t){
      .position =
          {
              .block_index = block_index,
              .block_ordinal = low_op->block_ordinal,
          },
      .op = low_op,
      .summary = *summary,
  };
  if (summary->byte_interval != NULL) {
    record->byte_interval = *summary->byte_interval;
    record->summary.byte_interval = &record->byte_interval;
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_record_source_memory_access(
    loom_low_lower_context_t* context, loom_op_t* low_op,
    const loom_low_source_memory_access_plan_t* source_plan,
    loom_low_lower_memory_access_record_flags_t flags) {
  loom_low_byte_interval_t byte_interval = {0};
  loom_low_memory_access_summary_t summary = {0};
  loom_low_source_memory_access_plan_make_summary(source_plan, &byte_interval,
                                                  &summary);
  return loom_low_lower_record_memory_access_summary(context, low_op, &summary,
                                                     flags);
}

iree_status_t loom_low_lower_map_type(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  *out_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      context->policy->map_type.fn(context->policy->map_type.user_data, context,
                                   source_op, source_type, out_low_type));
  return iree_ok_status();
}

iree_status_t loom_low_lower_map_value(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_value_id_t source_value_id,
                                       loom_type_t* out_low_type) {
  *out_low_type = loom_type_none();
  IREE_ASSERT_LT(source_value_id, context->module->values.count);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source_value_id);
  if (context->policy->map_value.fn == NULL) {
    return loom_low_lower_map_type(context, source_op, source_type,
                                   out_low_type);
  }
  return context->policy->map_value.fn(context->policy->map_value.user_data,
                                       context, source_op, source_value_id,
                                       source_type, out_low_type);
}

iree_status_t loom_low_lower_lookup_value(loom_low_lower_context_t* context,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(source_value_id != LOOM_VALUE_ID_INVALID &&
                  source_value_id < context->module->values.count,
              "source-to-low lookup for invalid source value");
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  loom_value_id_t low_value_id = context->lowering.value_map[source_ordinal];
  IREE_ASSERT(low_value_id != LOOM_VALUE_ID_INVALID,
              "source-to-low missing target value binding");
  IREE_ASSERT(low_value_id != LOOM_LOW_LOWER_VALUE_ID_ELIDED,
              "source-to-low requested elided source value");
  *out_low_value_id = low_value_id;
  return iree_ok_status();
}

bool loom_low_lower_source_value_has_low_mapping(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  const loom_value_id_t low_value_id =
      context->lowering.value_map[source_ordinal];
  return low_value_id != LOOM_VALUE_ID_INVALID &&
         low_value_id != LOOM_LOW_LOWER_VALUE_ID_ELIDED;
}

iree_status_t loom_low_lower_copy_value_name(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
                                             loom_value_id_t low_value_id) {
  return loom_module_overwrite_value_name(context->module, source_value_id,
                                          low_value_id);
}

iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id) {
  IREE_ASSERT_LT(low_value_id, context->module->values.count);
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  loom_value_id_t existing = context->lowering.value_map[source_ordinal];
  IREE_ASSERT(existing == LOOM_VALUE_ID_INVALID || existing == low_value_id);
  context->lowering.value_map[source_ordinal] = low_value_id;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

iree_status_t loom_low_lower_bind_value_alias(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t result_value_id) {
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  return loom_low_lower_bind_value(context, result_value_id, low_value_id);
}

iree_status_t loom_low_lower_elide_value(loom_low_lower_context_t* context,
                                         loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  loom_value_id_t existing = context->lowering.value_map[source_ordinal];
  IREE_ASSERT(existing == LOOM_VALUE_ID_INVALID ||
              existing == LOOM_LOW_LOWER_VALUE_ID_ELIDED);
  context->lowering.value_map[source_ordinal] = LOOM_LOW_LOWER_VALUE_ID_ELIDED;
  return iree_ok_status();
}
