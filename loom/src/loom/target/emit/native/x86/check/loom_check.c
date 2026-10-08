// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/check/loom_check.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/x86/abi.h"
#include "loom/target/emit/native/x86/function.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef struct loom_x86_loom_check_emit_options_t {
  // Module-local low.func.def symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_list_t allocation_fixed_values;
} loom_x86_loom_check_emit_options_t;

static bool loom_x86_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("x86-frame"));
}

static iree_status_t loom_x86_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_x86_loom_check_emit_options_t* options,
    bool* out_matched) {
  *out_matched = false;
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate x86 frame option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("x86 frame"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_parse_option(
    iree_string_view_t token, loom_x86_loom_check_emit_options_t* options) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_x86_loom_check_parse_key_value_option(token, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("x86 frame"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, &options->allocation_fixed_values);
}

static iree_status_t loom_x86_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_x86_loom_check_emit_options_t* out_options) {
  *out_options = (loom_x86_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 frame requires a low function symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(out_options->function_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 frame low function symbol name is "
                            "required");
  }

  IREE_RETURN_IF_ERROR(loom_check_low_emit_fixed_value_spec_list_initialize(
      option_text, request->case_arena, &out_options->allocation_fixed_values));
  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_x86_loom_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_emit_frame(
    const loom_low_emission_frame_t* frame,
    const loom_x86_function_abi_t* function_abi,
    const loom_x86_module_abi_t* module_abi, iree_string_builder_t* builder,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena) {
  loom_x86_function_t function;
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_x86_function_prepare(
      frame, function_abi, module_abi, emitter, arena, &accepted, &function));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "callee-preserved:"));
  if (!function.saved_registers) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " none"));
  }
  static const char* const register_names[] = {
      "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
      "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
  };
  const char* separator = " ";
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(register_names); ++i) {
    if (function.saved_registers & (1u << i)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%s%s", separator, register_names[i]));
      separator = ", ";
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  if (function.stack.allocation_size) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "stack: %u bytes, alignment %u\n",
        function.stack.allocation_size, function.stack.alignment));
  }
  if (function.stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "realignment: %s, saved RSP at +%u\n",
        register_names[function.stack.realignment.scratch_register],
        function.stack.realignment.saved_pointer_offset));
  }
  if (function.may_dirty_upper_vector_state &&
      function.upper_vector_call_cleanup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "upper-state-call-cleanups: %zu\n",
        function.upper_vector_call_cleanup_count));
  }
  if (function.may_dirty_upper_vector_state &&
      !function.has_upper_vector_result) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, "upper-state-cleanup: vzeroupper\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_prepare_abis(
    const loom_check_emit_provider_request_t* request,
    loom_x86_module_abi_t* out_module_abi) {
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < request->module->symbols.count; ++i) {
    const loom_op_t* op = request->module->symbols.entries[i].defining_op;
    function_count +=
        op != NULL && (loom_low_func_def_isa(op) || loom_low_func_decl_isa(op));
  }
  IREE_RETURN_IF_ERROR(loom_x86_module_abi_initialize(
      request->module->symbols.count, function_count, request->case_arena,
      out_module_abi));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, request->case_arena);
  iree_host_size_t function_index = 0;
  for (loom_symbol_id_t symbol_id = 0;
       symbol_id < request->module->symbols.count; ++symbol_id) {
    loom_op_t* op = request->module->symbols.entries[symbol_id].defining_op;
    if (op == NULL ||
        (!loom_low_func_def_isa(op) && !loom_low_func_decl_isa(op))) {
      continue;
    }
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        request->module, &symbol_facts, op, NULL,
        &request->low_registry->registry, (iree_diagnostic_emitter_t){0},
        &target));
    if (target.descriptor_set == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 frame function has no resolved target");
    }
    bool supported = false;
    iree_string_view_t constraint = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_function_abi_prepare(
        request->module, loom_func_like_cast(request->module, op), &target,
        request->case_arena, &supported, &constraint,
        &out_module_abi->functions[function_index]));
    if (!supported) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "%.*s",
                              (int)constraint.size, constraint.data);
    }
    loom_x86_module_abi_bind(out_module_abi, function_index++, symbol_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_x86_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_x86_loom_check_parse_emit_options(request, &options));

  loom_x86_module_abi_t module_abi = {0};
  IREE_RETURN_IF_ERROR(loom_x86_loom_check_prepare_abis(request, &module_abi));
  const loom_string_id_t function_name_id =
      loom_module_lookup_string(request->module, options.function_symbol_name);
  const loom_symbol_id_t function_symbol_id =
      function_name_id == LOOM_STRING_ID_INVALID
          ? LOOM_SYMBOL_ID_INVALID
          : loom_module_find_symbol(request->module, function_name_id);
  const loom_x86_function_abi_t* function_abi =
      function_symbol_id == LOOM_SYMBOL_ID_INVALID
          ? NULL
          : loom_x86_module_abi_lookup(
                &module_abi,
                (loom_symbol_ref_t){.symbol_id = function_symbol_id});

  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {0};
  // The fixture supplies incoming value locations. Reserve RSP just as the
  // object provider does before allocation; the native envelope uses it.
  loom_low_allocation_reserved_range_t reserved_ranges[3] = {
      {{0}},
  };
  const iree_host_size_t reserved_range_count =
      function_abi != NULL
          ? loom_x86_function_reserved_ranges(function_abi, reserved_ranges)
          : 0;
  const loom_low_emission_frame_options_t frame_options = {
      .resolved_target = function_abi != NULL ? &function_abi->target : NULL,
      .call_contracts =
          {
              .validate = loom_x86_function_call_contract_validate,
              .query = loom_x86_function_call_contract,
              .query_common_clobbers = loom_x86_function_common_call_clobbers,
              .user_data = &module_abi,
          },
      .schedule_strategy = options.schedule_strategy,
      .allocation_budgets = options.allocation_budgets,
      .allocation_budget_count = options.allocation_budget_count,
      .synchronous_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK |
                                    LOOM_LOW_STORAGE_SPACE_SET_PRIVATE |
                                    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH,
      .allocation_entry_locations =
          function_abi != NULL ? function_abi->call_contract.arguments : NULL,
      .allocation_entry_location_count =
          function_abi != NULL ? function_abi->call_contract.argument_count : 0,
      .allocation_exit_locations =
          function_abi != NULL ? function_abi->call_contract.results : NULL,
      .allocation_exit_location_count =
          function_abi != NULL ? function_abi->call_contract.result_count : 0,
      .allocation_reserved_ranges = reserved_ranges,
      .allocation_reserved_range_count = reserved_range_count,
  };
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, &frame_options,
      options.allocation_fixed_values.specs,
      options.allocation_fixed_values.count, &spill_free_options, &frame,
      &frame_accepted));
  if (request->diagnostic_collector != NULL &&
      loom_check_diagnostic_collector_has_error(
          request->diagnostic_collector)) {
    return iree_ok_status();
  }
  if (!frame_accepted) {
    return iree_ok_status();
  }
  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
  };
  IREE_ASSERT_NE(function_abi, NULL);
  return loom_x86_loom_check_emit_frame(
      &frame, function_abi, &module_abi, &request->result->actual_output,
      (iree_diagnostic_emitter_t){
          .fn = loom_check_diagnostic_emitter_capture_emit,
          .user_data = &capture,
      },
      request->case_arena);
}

static iree_status_t loom_x86_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "x86-frame");
}

const loom_check_emit_provider_t loom_x86_native_loom_check_emit_provider = {
    .name = IREE_SVL("x86-native"),
    .match = loom_x86_loom_check_emit_provider_matches,
    .execute = loom_x86_loom_check_emit_provider_execute,
    .append_names = loom_x86_loom_check_emit_provider_append_names,
};
