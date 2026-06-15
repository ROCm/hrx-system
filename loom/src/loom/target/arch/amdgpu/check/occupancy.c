// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/occupancy.h"

#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef struct loom_amdgpu_occupancy_check_emit_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame construction.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Optional occupancy diagnostic feedback requested by the RUN line.
  loom_amdgpu_occupancy_diagnostic_flags_t occupancy_diagnostic_flags;
  // True once a diagnostics option has been parsed.
  bool has_occupancy_diagnostics_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_t allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |allocation_fixed_value_specs|.
  iree_host_size_t allocation_fixed_value_spec_count;
} loom_amdgpu_occupancy_check_emit_options_t;

static bool loom_amdgpu_occupancy_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-occupancy-json")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-occupancy"));
}

static iree_status_t loom_amdgpu_occupancy_check_parse_diagnostics(
    iree_string_view_t value,
    loom_amdgpu_occupancy_check_emit_options_t* options) {
  if (options->has_occupancy_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate AMDGPU occupancy option 'diagnostics'");
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    options->occupancy_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("summary")) ||
             iree_string_view_equal(value, IREE_SV("all"))) {
    options->occupancy_diagnostic_flags =
        LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU occupancy option 'diagnostics' expected 'none', 'summary', or "
        "'all', got '%.*s'",
        (int)value.size, value.data);
  }
  options->has_occupancy_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_check_parse_key_value_option(
    iree_string_view_t name, iree_string_view_t value,
    loom_amdgpu_occupancy_check_emit_options_t* options, bool* out_matched) {
  *out_matched = true;
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU occupancy option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("AMDGPU occupancy"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_amdgpu_occupancy_check_parse_diagnostics(value, options);
  }
  *out_matched = false;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_check_parse_option(
    iree_string_view_t token,
    loom_amdgpu_occupancy_check_emit_options_t* options) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);

  bool matched = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_check_parse_key_value_option(
      name, value, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("AMDGPU occupancy"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, options->allocation_fixed_value_specs,
      IREE_ARRAYSIZE(options->allocation_fixed_value_specs),
      &options->allocation_fixed_value_spec_count);
}

static iree_status_t loom_amdgpu_occupancy_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_amdgpu_occupancy_check_emit_options_t* out_options) {
  *out_options = (loom_amdgpu_occupancy_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
  };

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@")) ||
      symbol_name.size == 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU occupancy requires a low function symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_occupancy_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_amdgpu_occupancy_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_occupancy_check_parse_emit_options(request, &options));

  loom_low_emission_frame_t frame = {0};
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.allocation_budgets, options.allocation_budget_count,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count,
      loom_low_schedule_pair_affinity_list_empty(), &storage_lease_provider,
      /*spill_free_options=*/NULL, &frame));

  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const loom_amdgpu_occupancy_options_t occupancy_options = {
      .emitter =
          {
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &diagnostic_capture,
          },
      .diagnostic_flags = options.occupancy_diagnostic_flags,
  };
  loom_amdgpu_occupancy_table_t table = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_build(
      &frame.allocation, &occupancy_options, request->case_arena, &table));
  return loom_amdgpu_occupancy_format_json(&table,
                                           &request->result->actual_output);
}

static iree_status_t loom_amdgpu_occupancy_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "amdgpu-occupancy-json, amdgpu-occupancy");
}

const loom_check_emit_provider_t
    loom_amdgpu_occupancy_loom_check_emit_provider = {
        .name = IREE_SVL("amdgpu-occupancy"),
        .match = loom_amdgpu_occupancy_check_emit_provider_matches,
        .execute = loom_amdgpu_occupancy_check_emit_provider_execute,
        .append_names = loom_amdgpu_occupancy_check_emit_provider_append_names,
};
