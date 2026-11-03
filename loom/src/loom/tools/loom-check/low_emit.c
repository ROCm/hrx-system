// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/low_emit.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/operand_ties.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
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

iree_status_t loom_check_low_emit_parse_schedule_diagnostics(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_diagnostic_flags_t* out_flags) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS;
  } else if (iree_string_view_equal(value, IREE_SV("resources"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS;
  } else if (iree_string_view_equal(value, IREE_SV("hazards"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS;
  } else if (iree_string_view_equal(value, IREE_SV("candidates"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("model"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    *out_flags = LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS |
                 LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS |
                 LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS |
                 LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS |
                 LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s option 'diagnostics' expected 'none', 'pressure', 'resources', "
        "'hazards', 'candidates', 'model', or 'all', got '%.*s'",
        (int)option_scope.size, option_scope.data, (int)value.size, value.data);
  }
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_parse_allocation_diagnostics(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_allocation_diagnostic_flags_t* out_flags) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("predicted-spills"))) {
    *out_flags = LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS;
  } else if (iree_string_view_equal(value, IREE_SV("copy-decisions"))) {
    *out_flags = LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("placement-decisions"))) {
    *out_flags = LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    *out_flags = LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS |
                 LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS |
                 LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s allocation diagnostics expected 'none', 'predicted-spills', "
        "'copy-decisions', 'placement-decisions', or 'all', got '%.*s'",
        (int)option_scope.size, option_scope.data, (int)value.size, value.data);
  }
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_find_low_function_def(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_test_case_t* test_case, iree_string_view_t filename,
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

static void loom_check_low_emit_consider_value_match(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t value_name,
    loom_check_low_emit_value_resolution_t* resolution) {
  if (value_id >= module->values.count) {
    return;
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
      iree_string_view_equal(
          loom_string_table_get(&module->strings, value->name_id),
          value_name)) {
    matches = true;
  }
  if (!matches) {
    return;
  }

  if (resolution->kind == LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_RESOLVED &&
      resolution->value_id != value_id) {
    resolution->kind = LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS;
    resolution->ambiguous_value_id = value_id;
    return;
  }
  resolution->kind = LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_RESOLVED;
  resolution->value_id = value_id;
}

static void loom_check_low_emit_resolve_value_in_region(
    const loom_module_t* module, const loom_region_t* region,
    iree_string_view_t value_name,
    loom_check_low_emit_value_resolution_t* resolution) {
  for (iree_host_size_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_check_low_emit_consider_value_match(
          module, loom_block_arg_id(block, arg_index), value_name, resolution);
      if (resolution->kind == LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS) {
        return;
      }
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        loom_check_low_emit_consider_value_match(module, results[result_index],
                                                 value_name, resolution);
        if (resolution->kind ==
            LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS) {
          return;
        }
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        loom_check_low_emit_resolve_value_in_region(
            module, regions[region_index], value_name, resolution);
        if (resolution->kind ==
            LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS) {
          return;
        }
      }
    }
  }
}

void loom_check_low_emit_resolve_function_value(
    const loom_module_t* module, const loom_op_t* low_function,
    iree_string_view_t value_name,
    loom_check_low_emit_value_resolution_t* out_resolution) {
  *out_resolution = (loom_check_low_emit_value_resolution_t){
      .kind = LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_NOT_FOUND,
      .value_id = LOOM_VALUE_ID_INVALID,
      .ambiguous_value_id = LOOM_VALUE_ID_INVALID,
  };
  loom_check_low_emit_resolve_value_in_region(
      module, loom_low_function_const_body(low_function), value_name,
      out_resolution);
}

iree_status_t loom_check_low_emit_materialize_operand_ties(
    loom_module_t* module, loom_op_t* function,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena) {
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, arena);
  loom_region_t* body = loom_low_function_body(function);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; iree_status_is_ok(status) && i < body->block_count;
       ++i) {
    for (loom_op_t* op = body->blocks[i]->first_op;
         iree_status_is_ok(status) && op != NULL;) {
      loom_op_t* next = op->next_op;
      loom_op_t* replacement = NULL;
      status = loom_low_operand_ties_materialize(&rewriter, descriptor_set,
                                                 /*memory_accesses=*/NULL, op,
                                                 &replacement);
      op = next;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module,
    iree_string_view_t function_symbol_name,
    const loom_low_emission_frame_options_t* options,
    const loom_low_emission_frame_spill_free_options_t* spill_free_options,
    loom_low_emission_frame_t* out_frame, bool* out_accepted) {
  *out_accepted = false;
  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = native_module->module,
      .source_resolver = native_module->source_resolver,
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
      native_module->module, function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }

  loom_low_emission_frame_options_t frame_options = *options;
  frame_options.descriptor_registry = &request->low_registry->registry;
  frame_options.emitter = emitter;
  loom_low_resolved_target_t resolved_target = {0};
  if (frame_options.resolved_target == NULL) {
    loom_symbol_fact_table_t symbol_facts;
    loom_symbol_fact_table_initialize(&symbol_facts, request->case_arena);
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        native_module->module, &symbol_facts, low_function,
        frame_options.function_target_facts, frame_options.descriptor_registry,
        emitter, &resolved_target));
    if (resolved_target.descriptor_set == NULL) {
      return iree_ok_status();
    }
    frame_options.resolved_target = &resolved_target;
  }
  // Direct Low emission skips optimization, but still realizes the same
  // physical operand requirements as the shipping preparation pipeline.
  IREE_RETURN_IF_ERROR(loom_check_low_emit_materialize_operand_ties(
      native_module->module, low_function,
      frame_options.resolved_target->descriptor_set, request->case_arena));
  *out_frame = (loom_low_emission_frame_t){0};
  if (spill_free_options != NULL) {
    return loom_low_emission_frame_build_spill_free(
        native_module->module, low_function, &frame_options, spill_free_options,
        request->case_arena, out_frame, out_accepted);
  }
  return loom_low_emission_frame_build(native_module->module, low_function,
                                       &frame_options, request->case_arena,
                                       out_frame, out_accepted);
}
