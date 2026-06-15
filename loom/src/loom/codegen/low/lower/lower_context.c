// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/lower/lower_internal.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"

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
  return loom_low_lower_target_key(context->options->bundle);
}

static iree_string_view_t loom_low_lower_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_nonempty(bundle->export_plan->name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_lower_context_export_name(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_export_name(context->options->bundle);
}

static iree_string_view_t loom_low_lower_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_lower_context_config_key(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_config_key(context->options->bundle);
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
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    if (loom_low_lower_context_should_stop(context)) {
      return iree_ok_status();
    }
    ++context->result->error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_REMARK) {
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

#define LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT 5

static void loom_low_lower_make_target_context_params(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_diagnostic_param_t* params) {
  params[0] =
      loom_param_string(loom_low_lower_target_key(context->options->bundle));
  params[1] =
      loom_param_string(loom_low_lower_export_name(context->options->bundle));
  params[2] =
      loom_param_string(loom_low_lower_config_key(context->options->bundle));
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

const loom_target_bundle_t* loom_low_lower_context_bundle(
    const loom_low_lower_context_t* context) {
  return context->options->bundle;
}

loom_symbol_ref_t loom_low_lower_context_target_ref(
    const loom_low_lower_context_t* context) {
  return context->options->target_ref;
}

const loom_low_descriptor_set_t* loom_low_lower_context_descriptor_set(
    const loom_low_lower_context_t* context) {
  return context->descriptor_set;
}

const loom_value_fact_table_t* loom_low_lower_context_fact_table(
    const loom_low_lower_context_t* context) {
  return context->lowering.fact_table;
}

iree_status_t loom_low_lower_context_view_regions(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t** out_view_regions) {
  *out_view_regions = NULL;
  if (!context->lowering.view_regions_initialized) {
    IREE_RETURN_IF_ERROR(loom_view_region_table_initialize(
        context->lowering.fact_table, &context->lowering.value_domain,
        &context->arena, &context->lowering.view_regions));
    context->lowering.view_regions_initialized = true;
  }
  if (!context->lowering.view_regions_analyzed) {
    IREE_RETURN_IF_ERROR(
        loom_view_region_table_analyze(&context->lowering.view_regions));
    context->lowering.view_regions_analyzed = true;
  }
  *out_view_regions = &context->lowering.view_regions;
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

iree_status_t loom_low_lower_allocate_scratch_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(&context->arena, count, element_size,
                                   out_ptr);
}

iree_status_t loom_low_lower_allocate_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data) {
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  return loom_low_lower_allocate_scratch_array(context, 1, data_length,
                                               out_data);
}

static loom_region_t* loom_low_lower_context_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
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
    uint8_t successor_index, loom_block_t** out_low_dest) {
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

iree_status_t loom_low_lower_remap_successor_args(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint8_t successor_index, loom_block_t* low_dest,
    const loom_value_id_t* source_args, uint16_t source_arg_count,
    loom_value_id_t** out_low_args) {
  *out_low_args = NULL;
  if (source_arg_count == 0) {
    IREE_ASSERT_EQ(low_dest->arg_count, 0);
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(source_arg_count, low_dest->arg_count);

  loom_value_id_t* low_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_arg_count, sizeof(*low_args), (void**)&low_args));
  for (uint16_t i = 0; i < source_arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_args[i], &low_args[i]));

    const loom_type_t required_type =
        loom_block_arg_type(context->module, low_dest, i);
    const loom_type_t actual_type =
        loom_module_value_type(context->module, low_args[i]);
    if (loom_type_equal(actual_type, required_type)) {
      continue;
    }

    if (context->policy->materialize_branch_arg.fn == NULL) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "lowering policy produced a branch payload type mismatch");
    }
    IREE_RETURN_IF_ERROR(context->policy->materialize_branch_arg.fn(
        context->policy->materialize_branch_arg.user_data, context,
        source_terminator, successor_index, i, source_args[i], low_args[i],
        required_type, &low_args[i]));

    const loom_type_t materialized_type =
        loom_module_value_type(context->module, low_args[i]);
    if (!loom_type_equal(materialized_type, required_type)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "lowering policy materialized a branch payload with the wrong type");
    }
  }
  *out_low_args = low_args;
  return iree_ok_status();
}

iree_status_t loom_low_lower_interpose_successor_dest(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint8_t successor_index, loom_block_t* interposed_low_block,
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
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
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

iree_status_t loom_low_lower_resolve_descriptor_row(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){
      .descriptor = NULL,
      .opcode_id = LOOM_STRING_ID_INVALID,
  };
  if (loom_low_descriptor_set_descriptor_ordinal(context->descriptor_set,
                                                 descriptor) ==
      LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "generated target-low policy references a "
                            "descriptor row outside the active descriptor set");
  }

  iree_string_view_t key = loom_low_descriptor_set_string(
      context->descriptor_set, descriptor->key_string_offset);
  IREE_ASSERT_FALSE(iree_string_view_is_empty(key));
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(context->module, key, &opcode_id));
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){
      .descriptor = descriptor,
      .opcode_id = opcode_id,
  };
  return iree_ok_status();
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
      descriptor->opcode_id, operands, operand_count, attrs, result_types,
      result_count, tied_results, tied_result_count, location, out_op);
}

iree_status_t loom_low_lower_emit_resolved_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_low_build_resolved_descriptor_const(
      &context->builder, context->descriptor_set, descriptor->descriptor,
      descriptor->opcode_id, attrs, result_type, location, out_op);
}

iree_status_t loom_low_lower_record_memory_access_summary(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_memory_access_summary_t* summary) {
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
      .summary = *summary,
  };
  if (summary->byte_interval != NULL) {
    record->byte_interval = *summary->byte_interval;
    record->summary.byte_interval = &record->byte_interval;
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_record_source_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_source_memory_access_plan_t* source_plan) {
  loom_low_byte_interval_t byte_interval = {0};
  loom_low_memory_access_summary_t summary = {0};
  loom_low_source_memory_access_plan_make_summary(source_plan, &byte_interval,
                                                  &summary);
  return loom_low_lower_record_memory_access_summary(context, low_op, &summary);
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
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  loom_value_id_t low_value_id = context->lowering.value_map[source_ordinal];
  IREE_ASSERT(low_value_id != LOOM_VALUE_ID_INVALID);
  IREE_ASSERT(low_value_id != LOOM_LOW_LOWER_VALUE_ID_ELIDED);
  *out_low_value_id = low_value_id;
  return iree_ok_status();
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

static iree_status_t loom_low_lower_create_derived_symbol(
    loom_low_lower_context_t* context, iree_string_view_t base_name,
    iree_string_view_t suffix, bool append_index, uint32_t index,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  IREE_ASSERT_FALSE(iree_string_view_is_empty(base_name));
  IREE_ASSERT_FALSE(iree_string_view_equal(base_name, IREE_SV("<unnamed>")));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_string_builder_append_string(&builder, base_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, suffix);
  }
  if (iree_status_is_ok(status) && append_index) {
    status = iree_string_builder_append_format(&builder, "%u", (unsigned)index);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = loom_module_intern_string(
        context->module,
        iree_make_string_view(iree_string_builder_buffer(&builder),
                              iree_string_builder_size(&builder)),
        &name_id);
  }
  iree_string_builder_deinitialize(&builder);
  IREE_RETURN_IF_ERROR(status);

  if (loom_module_find_symbol(context->module, name_id) !=
      LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target-low derived symbol already exists");
  }

  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(context->module, name_id, &symbol_id));
  *out_symbol_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  return iree_ok_status();
}

iree_status_t loom_low_lower_create_function_symbol(
    loom_low_lower_context_t* context, iree_string_view_t suffix,
    bool append_index, uint32_t index, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  IREE_ASSERT(context->low_func_op != NULL);
  loom_func_like_t low_function =
      loom_func_like_cast(context->module, context->low_func_op);
  iree_string_view_t low_function_name = loom_low_lower_symbol_name(
      context->module, loom_func_like_callee(low_function));
  return loom_low_lower_create_derived_symbol(
      context, low_function_name, suffix, append_index, index, out_symbol_ref);
}
