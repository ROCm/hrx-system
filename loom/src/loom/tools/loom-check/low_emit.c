// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/low_emit.h"

#include <inttypes.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/tools/loom-check/diagnostics.h"

iree_status_t loom_check_low_emit_parse_schedule_strategy(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_strategy_t* out_strategy) {
  if (iree_string_view_equal(value, IREE_SV("source"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("pressure"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("latency-hiding"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("resource-stall"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "%.*s option 'strategy' expected 'source', 'pressure', "
      "'latency-hiding', or 'resource-stall', got '%.*s'",
      (int)option_scope.size, option_scope.data, (int)value.size, value.data);
}

iree_status_t loom_check_low_emit_parse_allocation_budget(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count) {
  if (*budget_count >= budget_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many %.*s allocation budgets",
                            (int)option_scope.size, option_scope.data);
  }
  iree_string_view_t register_class = iree_string_view_empty();
  iree_string_view_t budget_text = iree_string_view_empty();
  iree_string_view_split(token, '=', &register_class, &budget_text);
  register_class = iree_string_view_trim(register_class);
  budget_text = iree_string_view_trim(budget_text);
  if (iree_string_view_is_empty(register_class) ||
      iree_string_view_is_empty(budget_text)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s allocation budget must have the form <register-class>=<units>",
        (int)option_scope.size, option_scope.data);
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %.*s allocation budget '%.*s'",
                            (int)option_scope.size, option_scope.data,
                            (int)budget_text.size, budget_text.data);
  }
  budgets[(*budget_count)++] = (loom_low_allocation_budget_t){
      .register_class = register_class,
      .max_units = max_units,
  };
  return iree_ok_status();
}

static iree_status_t loom_check_low_emit_parse_location_kind(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_allocation_location_kind_t* out_location_kind) {
  if (iree_string_view_equal(value, IREE_SV("physical_register"))) {
    *out_location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("target_id"))) {
    *out_location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "%.*s fixed allocation location expected 'physical_register' or "
      "'target_id', got '%.*s'",
      (int)option_scope.size, option_scope.data, (int)value.size, value.data);
}

iree_status_t loom_check_low_emit_parse_fixed_value_spec(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_capacity, iree_host_size_t* fixed_spec_count) {
  if (*fixed_spec_count >= fixed_spec_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many %.*s fixed allocation values",
                            (int)option_scope.size, option_scope.data);
  }

  iree_string_view_t value_ref = iree_string_view_empty();
  iree_string_view_t kind_and_range = iree_string_view_empty();
  iree_string_view_split(value, ':', &value_ref, &kind_and_range);
  value_ref = iree_string_view_trim(value_ref);
  kind_and_range = iree_string_view_trim(kind_and_range);

  iree_string_view_t kind_text = iree_string_view_empty();
  iree_string_view_t range_text = iree_string_view_empty();
  iree_string_view_split(kind_and_range, ':', &kind_text, &range_text);
  kind_text = iree_string_view_trim(kind_text);
  range_text = iree_string_view_trim(range_text);

  iree_string_view_t base_text = iree_string_view_empty();
  iree_string_view_t count_text = iree_string_view_empty();
  iree_string_view_split(range_text, ':', &base_text, &count_text);
  base_text = iree_string_view_trim(base_text);
  count_text = iree_string_view_trim(count_text);

  if (!iree_string_view_starts_with(value_ref, IREE_SV("%")) ||
      value_ref.size == 1 || iree_string_view_is_empty(kind_text) ||
      iree_string_view_is_empty(base_text) ||
      iree_string_view_is_empty(count_text) ||
      iree_string_view_find_char(count_text, ':', 0) != IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s fixed allocation must have the form "
        "fixed=%%value:<physical_register|target_id>:<base>:<count>",
        (int)option_scope.size, option_scope.data);
  }

  loom_low_allocation_location_kind_t location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_location_kind(
      kind_text, option_scope, &location_kind));

  uint32_t location_base = 0;
  if (!iree_string_view_atoi_uint32(base_text, &location_base)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %.*s fixed allocation base '%.*s'",
                            (int)option_scope.size, option_scope.data,
                            (int)base_text.size, base_text.data);
  }
  uint32_t location_count = 0;
  if (!iree_string_view_atoi_uint32(count_text, &location_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %.*s fixed allocation count '%.*s'",
                            (int)option_scope.size, option_scope.data,
                            (int)count_text.size, count_text.data);
  }
  if (location_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s fixed allocation count must be non-zero",
                            (int)option_scope.size, option_scope.data);
  }

  fixed_specs[(*fixed_spec_count)++] = (loom_check_low_emit_fixed_value_spec_t){
      .value_name = iree_string_view_substr(value_ref, 1, IREE_HOST_SIZE_MAX),
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
  };
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_parse_allocation_option(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count,
    loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_capacity, iree_host_size_t* fixed_spec_count) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("fixed"))) {
    return loom_check_low_emit_parse_fixed_value_spec(
        value, option_scope, fixed_specs, fixed_spec_capacity,
        fixed_spec_count);
  }
  return loom_check_low_emit_parse_allocation_budget(
      token, option_scope, budgets, budget_capacity, budget_count);
}

iree_status_t loom_check_low_emit_find_low_function_def(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, loom_op_t** out_low_function) {
  *out_low_function = NULL;
  loom_string_id_t name_id = loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
    };
    return loom_check_diagnostic_collector_emit_case_source(
        diagnostic_collector, test_case, filename, LOOM_EMITTER_PASS,
        LOOM_ERR_SYMBOL_002, params, IREE_ARRAYSIZE(params));
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
    };
    return loom_check_diagnostic_collector_emit_case_source(
        diagnostic_collector, test_case, filename, LOOM_EMITTER_PASS,
        LOOM_ERR_SYMBOL_002, params, IREE_ARRAYSIZE(params));
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!loom_low_function_def_isa(symbol->defining_op) ||
      !loom_low_function_body(symbol->defining_op)) {
    iree_string_view_t actual_kind = IREE_SV("<unknown>");
    if (symbol->definition) {
      actual_kind = loom_symbol_definition_descriptor_name(symbol->definition);
    } else if (symbol->defining_op) {
      actual_kind = loom_op_name(module, symbol->defining_op);
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
        loom_param_string(actual_kind),
        loom_param_string(IREE_SV("low function body")),
    };
    const loom_diagnostic_emission_t emission = {
        .op = symbol->defining_op,
        .error = LOOM_ERR_SYMBOL_003,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  *out_low_function = symbol->defining_op;
  return iree_ok_status();
}

static iree_status_t loom_check_low_emit_consider_fixed_value_match(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t value_name, bool* found, bool* ambiguous,
    loom_value_id_t* out_value_id, loom_value_id_t* out_ambiguous_value_id) {
  if (value_id >= module->values.count) {
    return iree_ok_status();
  }

  bool matches = false;
  uint32_t numeric_value_id = 0;
  if (iree_string_view_atoi_uint32(value_name, &numeric_value_id) &&
      numeric_value_id == value_id) {
    matches = true;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!matches && value->name_id != LOOM_STRING_ID_INVALID &&
      value->name_id < module->strings.count &&
      iree_string_view_equal(module->strings.entries[value->name_id],
                             value_name)) {
    matches = true;
  }
  if (!matches) {
    return iree_ok_status();
  }

  if (*found && *out_value_id != value_id) {
    *ambiguous = true;
    *out_ambiguous_value_id = value_id;
    return iree_ok_status();
  }
  *found = true;
  *out_value_id = value_id;
  return iree_ok_status();
}

static iree_status_t loom_check_low_emit_find_value_in_region(
    const loom_module_t* module, const loom_region_t* region,
    iree_string_view_t value_name, bool* found, bool* ambiguous,
    loom_value_id_t* out_value_id, loom_value_id_t* out_ambiguous_value_id) {
  for (iree_host_size_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      IREE_RETURN_IF_ERROR(loom_check_low_emit_consider_fixed_value_match(
          module, loom_block_arg_id(block, arg_index), value_name, found,
          ambiguous, out_value_id, out_ambiguous_value_id));
      if (*ambiguous) return iree_ok_status();
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        IREE_RETURN_IF_ERROR(loom_check_low_emit_consider_fixed_value_match(
            module, results[result_index], value_name, found, ambiguous,
            out_value_id, out_ambiguous_value_id));
        if (*ambiguous) return iree_ok_status();
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        IREE_RETURN_IF_ERROR(loom_check_low_emit_find_value_in_region(
            module, regions[region_index], value_name, found, ambiguous,
            out_value_id, out_ambiguous_value_id));
        if (*ambiguous) return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_low_emit_emit_unresolved_fixed_value_selector(
    const loom_module_t* module, const loom_op_t* low_function,
    iree_string_view_t value_name, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(module, low_function)),
      loom_param_string(value_name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = low_function,
      .error = LOOM_ERR_BACKEND_037,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_check_low_emit_emit_ambiguous_fixed_value_selector(
    const loom_module_t* module, const loom_op_t* low_function,
    iree_string_view_t value_name, loom_value_id_t first_value_id,
    loom_value_id_t second_value_id, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(module, low_function)),
      loom_param_string(value_name),
      loom_param_u32(first_value_id),
      loom_param_u32(second_value_id),
  };
  const loom_diagnostic_emission_t emission = {
      .op = low_function,
      .error = LOOM_ERR_BACKEND_038,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_check_low_emit_find_value_in_low_function(
    const loom_module_t* module, const loom_op_t* low_function,
    iree_string_view_t value_name, iree_diagnostic_emitter_t emitter,
    loom_value_id_t* out_value_id, bool* out_resolved) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  *out_resolved = false;
  const loom_region_t* body = loom_low_function_const_body(low_function);
  bool found = false;
  bool ambiguous = false;
  loom_value_id_t ambiguous_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_value_in_region(
      module, body, value_name, &found, &ambiguous, out_value_id,
      &ambiguous_value_id));
  if (ambiguous) {
    return loom_check_low_emit_emit_ambiguous_fixed_value_selector(
        module, low_function, value_name, *out_value_id, ambiguous_value_id,
        emitter);
  }
  if (!found) {
    return loom_check_low_emit_emit_unresolved_fixed_value_selector(
        module, low_function, value_name, emitter);
  }
  *out_resolved = true;
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_resolve_fixed_value_specs(
    loom_module_t* module, loom_op_t* low_function,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count, iree_diagnostic_emitter_t emitter,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count, bool* out_resolved,
    iree_arena_allocator_t* arena) {
  *out_fixed_values = NULL;
  *out_fixed_value_count = 0;
  *out_resolved = true;
  if (fixed_spec_count == 0) {
    return iree_ok_status();
  }
  if (!loom_low_function_def_isa(low_function) ||
      !loom_low_function_body(low_function)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low fixed allocation requires a low function body");
  }

  loom_low_allocation_fixed_value_t* fixed_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, fixed_spec_count, sizeof(*fixed_values), (void**)&fixed_values));
  for (iree_host_size_t i = 0; i < fixed_spec_count; ++i) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    bool value_resolved = false;
    IREE_RETURN_IF_ERROR(loom_check_low_emit_find_value_in_low_function(
        module, low_function, fixed_specs[i].value_name, emitter, &value_id,
        &value_resolved));
    if (!value_resolved) {
      *out_resolved = false;
      continue;
    }
    fixed_values[i] = (loom_low_allocation_fixed_value_t){
        .value_id = value_id,
        .location_kind = fixed_specs[i].location_kind,
        .location_base = fixed_specs[i].location_base,
        .location_count = fixed_specs[i].location_count,
    };
  }
  if (!*out_resolved) {
    return iree_ok_status();
  }
  *out_fixed_values = fixed_values;
  *out_fixed_value_count = fixed_spec_count;
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t function_symbol_name,
    loom_low_schedule_strategy_t schedule_strategy,
    const loom_low_allocation_budget_t* allocation_budgets,
    iree_host_size_t allocation_budget_count,
    const loom_check_low_emit_fixed_value_spec_t* allocation_fixed_specs,
    iree_host_size_t allocation_fixed_spec_count,
    loom_low_schedule_pair_affinity_list_t schedule_pair_affinities,
    const loom_low_storage_lease_provider_t* storage_lease_provider,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    loom_low_emission_frame_t* out_frame) {
  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t emitter = {0};
  if (request->diagnostic_collector != NULL) {
    emitter = (iree_diagnostic_emitter_t){
        .fn = loom_check_diagnostic_emitter_capture_emit,
        .user_data = &diagnostic_capture,
    };
  }
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      request->module, function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  bool fixed_values_resolved = false;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_resolve_fixed_value_specs(
      request->module, low_function, allocation_fixed_specs,
      allocation_fixed_spec_count, emitter, &fixed_values, &fixed_value_count,
      &fixed_values_resolved, request->case_arena));
  if (!fixed_values_resolved) {
    return iree_ok_status();
  }

  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &request->low_registry->registry,
      .schedule_strategy = schedule_strategy,
      .schedule_pair_affinities = schedule_pair_affinities,
      .allocation_budgets = allocation_budgets,
      .allocation_budget_count = allocation_budget_count,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .storage_lease_provider = storage_lease_provider,
  };
  frame_options.emitter = emitter;
  *out_frame = (loom_low_emission_frame_t){0};
  if (spill_free_options != NULL) {
    return loom_low_emission_frame_build_spill_free(
        request->module, low_function, &frame_options, spill_free_options,
        request->case_arena, out_frame);
  }
  return loom_low_emission_frame_build(request->module, low_function,
                                       &frame_options, request->case_arena,
                                       out_frame);
}
