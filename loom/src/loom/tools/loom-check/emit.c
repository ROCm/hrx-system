// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_json.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/packet_json.h"
#include "loom/codegen/low/schedule/json.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/error/source.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/pipeline.h"
#include "loom/target/low_packet_diagnostics.h"
#include "loom/target/provider.h"
#include "loom/tools/loom-check/analysis.h"
#include "loom/tools/loom-check/comparison.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/low_emit.h"
#include "loom/tools/loom-check/low_report.h"
#include "loom/util/stream.h"

typedef enum loom_check_emit_format_e {
  LOOM_CHECK_EMIT_LIVENESS_JSON = 0,
  LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON,
  LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON,
  LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY,
  LOOM_CHECK_EMIT_LOW_PACKET_JSON,
  LOOM_CHECK_EMIT_LOW_COMPILE_REPORT,
  LOOM_CHECK_EMIT_STORAGE_INTERFERENCE,
} loom_check_emit_format_t;

typedef struct loom_check_emit_low_allocation_summary_row_t {
  // Number of assignments mapped to this descriptor register class.
  iree_host_size_t assignment_count;
  // Total allocation units requested by the assignments in this class.
  uint64_t unit_count;
  // One-past-highest target-local ID used by this class.
  uint64_t target_id_count;
  // One-past-highest physical register used by this class.
  uint64_t physical_register_count;
  // One-past-highest spill slot used by this class.
  uint64_t spill_slot_count;
  // Number of assignments with no concrete location.
  iree_host_size_t unassigned_count;
} loom_check_emit_low_allocation_summary_row_t;

static const iree_string_view_t kLoomCheckEmitCoreTargetNames[] = {
    IREE_SVL("liveness-json"),        IREE_SVL("liveness"),
    IREE_SVL("low-schedule-json"),    IREE_SVL("low-schedule"),
    IREE_SVL("low-allocation-json"),  IREE_SVL("low-allocation-summary"),
    IREE_SVL("low-allocation"),       IREE_SVL("low-packet-json"),
    IREE_SVL("low-packet"),           IREE_SVL("low-compile-report"),
    IREE_SVL("storage-interference"),
};

typedef struct loom_check_emit_request_t {
  // Serialized target form to produce before comparison.
  loom_check_emit_format_t format;
  // Canonical/user-facing emit target name used in diagnostics.
  iree_string_view_t emit_target_name;
  // Module-local function symbol name used by analysis dumps.
  iree_string_view_t analysis_symbol_name;
  // Low allocation diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t low_allocation_diagnostic_flags;
  // True once a low allocation diagnostics option has been parsed.
  bool has_low_allocation_diagnostics_option;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t low_schedule_diagnostic_flags;
  // True once a low scheduler diagnostics option has been parsed.
  bool has_low_schedule_diagnostics_option;
  // Low scheduler candidate-selection strategy requested by the RUN line.
  loom_low_schedule_strategy_t low_schedule_strategy;
  // True once a low scheduler strategy option has been parsed.
  bool has_low_schedule_strategy_option;
  // Target-low packet diagnostics requested by the RUN line.
  loom_target_low_packet_diagnostic_flags_t low_packet_diagnostic_flags;
  // True once a low packet diagnostics option has been parsed.
  bool has_low_packet_diagnostics_option;
  // True when the emit target should run but suppress comparable output.
  bool suppress_actual_output;
  // True once an output option has been parsed.
  bool has_output_option;
} loom_check_emit_request_t;

static iree_status_t loom_check_emit_parse_json_output_option(
    iree_string_view_t target_name, iree_string_view_t value,
    loom_check_emit_request_t* request) {
  if (request->has_output_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate %.*s option 'output'",
                            (int)target_name.size, target_name.data);
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    request->suppress_actual_output = false;
  } else if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->suppress_actual_output = true;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s option 'output' expected 'json' or 'none', "
                            "got '%.*s'",
                            (int)target_name.size, target_name.data,
                            (int)value.size, value.data);
  }
  request->has_output_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_allocation_option(
    iree_string_view_t target_name, iree_string_view_t token,
    loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    if (!iree_string_view_equal(target_name, IREE_SV("low-allocation-json"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%.*s does not accept option 'output'; use low-allocation-json for "
          "JSON output",
          (int)target_name.size, target_name.data);
    }
    return loom_check_emit_parse_json_output_option(target_name, value,
                                                    request);
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown %.*s option '%.*s'", (int)target_name.size,
                            target_name.data, (int)name.size, name.data);
  }
  if (request->has_low_allocation_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate %.*s option 'diagnostics'",
                            (int)target_name.size, target_name.data);
  }
  IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_allocation_diagnostics(
      value, target_name, &request->low_allocation_diagnostic_flags));
  request->has_low_allocation_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_allocation_options(
    iree_string_view_t target_name, iree_string_view_t text,
    loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(loom_check_emit_parse_low_allocation_option(
          target_name, token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_schedule_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_emit_parse_json_output_option(
        IREE_SV("low-schedule-json"), value, request);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (request->has_low_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-json option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("low-schedule-json"), &request->low_schedule_strategy));
    request->has_low_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low-schedule-json option '%.*s'",
                            (int)name.size, name.data);
  }
  if (request->has_low_schedule_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate low-schedule-json option 'diagnostics'");
  }
  IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_diagnostics(
      value, IREE_SV("low-schedule-json"),
      &request->low_schedule_diagnostic_flags));
  request->has_low_schedule_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_schedule_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_low_schedule_option(token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_packet_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_emit_parse_json_output_option(IREE_SV("low-packet-json"),
                                                    value, request);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (request->has_low_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-packet-json option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("low-packet-json"), &request->low_schedule_strategy));
    request->has_low_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    if (request->has_low_packet_diagnostics_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-packet-json option "
                              "'diagnostics'");
    }
    if (iree_string_view_equal(value, IREE_SV("none"))) {
      request->low_packet_diagnostic_flags = 0;
    } else if (iree_string_view_equal(value, IREE_SV("packets"))) {
      request->low_packet_diagnostic_flags =
          LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_TARGET_PACKETS;
    } else if (iree_string_view_equal(value, IREE_SV("all"))) {
      request->low_packet_diagnostic_flags =
          LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_ALL;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-packet-json option 'diagnostics' expected 'none', 'packets', "
          "or 'all', got '%.*s'",
          (int)value.size, value.data);
    }
    request->has_low_packet_diagnostics_option = true;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown low-packet-json option '%.*s'",
                          (int)name.size, name.data);
}

static iree_status_t loom_check_emit_parse_low_packet_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_low_packet_option(token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static void loom_check_emit_split_target(
    iree_string_view_t emit_target, iree_string_view_t* out_target_name,
    iree_string_view_t* out_target_options) {
  emit_target = iree_string_view_trim(emit_target);
  iree_string_view_split(emit_target, ' ', out_target_name, out_target_options);
  *out_target_name = iree_string_view_trim(*out_target_name);
  *out_target_options = iree_string_view_trim(*out_target_options);
}

static bool loom_check_emit_core_target_matches(
    iree_string_view_t target_name) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomCheckEmitCoreTargetNames); ++i) {
    if (iree_string_view_equal(target_name, kLoomCheckEmitCoreTargetNames[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_emit_append_supported_target_names(
    const loom_check_environment_t* environment,
    iree_string_builder_t* output) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomCheckEmitCoreTargetNames); ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, kLoomCheckEmitCoreTargetNames[i]));
  }
  if (environment == NULL || environment->emit_providers.providers == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < environment->emit_providers.provider_count;
       ++i) {
    const loom_check_emit_provider_t* provider =
        environment->emit_providers.providers[i];
    if (provider == NULL || provider->append_names == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    IREE_RETURN_IF_ERROR(provider->append_names(provider, output));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_fail_unknown_target(
    const loom_check_environment_t* environment, iree_string_view_t target_name,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &result->detail,
      "unknown emit target '%.*s'; supported emit targets are ",
      (int)target_name.size, target_name.data));
  IREE_RETURN_IF_ERROR(loom_check_emit_append_supported_target_names(
      environment, &result->detail));
  return iree_string_builder_append_cstring(&result->detail, "\n");
}

static iree_status_t loom_check_emit_parse_request(
    iree_string_view_t emit_target, loom_check_emit_request_t* out_request) {
  *out_request = (loom_check_emit_request_t){
      .format = LOOM_CHECK_EMIT_LIVENESS_JSON,
      .emit_target_name = IREE_SV("emit"),
      .analysis_symbol_name = iree_string_view_empty(),
      .low_allocation_diagnostic_flags = 0,
      .has_low_allocation_diagnostics_option = false,
      .low_schedule_diagnostic_flags = 0,
      .has_low_schedule_diagnostics_option = false,
      .low_schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .has_low_schedule_strategy_option = false,
      .low_packet_diagnostic_flags = 0,
      .has_low_packet_diagnostics_option = false,
      .suppress_actual_output = false,
      .has_output_option = false,
  };
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t target_options = iree_string_view_empty();
  loom_check_emit_split_target(emit_target, &target_name, &target_options);
  if (!iree_string_view_is_empty(target_name)) {
    out_request->emit_target_name = target_name;
  }

  if (iree_string_view_equal(target_name, IREE_SV("liveness-json")) ||
      iree_string_view_equal(target_name, IREE_SV("liveness")) ||
      iree_string_view_equal(target_name, IREE_SV("storage-interference"))) {
    if (!iree_string_view_starts_with(target_options, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s requires a function symbol name",
                              (int)target_name.size, target_name.data);
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(target_options, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function symbol name is required");
    }
    out_request->format =
        iree_string_view_equal(target_name, IREE_SV("storage-interference"))
            ? LOOM_CHECK_EMIT_STORAGE_INTERFERENCE
            : LOOM_CHECK_EMIT_LIVENESS_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-schedule-json")) ||
             iree_string_view_equal(target_name, IREE_SV("low-schedule"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(target_options, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-schedule-json requires a low function "
                              "symbol name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_low_schedule_options(option_text, out_request));
    out_request->format = LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-allocation-json")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("low-allocation-summary")) ||
             iree_string_view_equal(target_name, IREE_SV("low-allocation"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(target_options, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s requires a low function symbol name",
                              (int)target_name.size, target_name.data);
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(loom_check_emit_parse_low_allocation_options(
        target_name, option_text, out_request));
    out_request->format =
        iree_string_view_equal(target_name, IREE_SV("low-allocation-json"))
            ? LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON
            : LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name, IREE_SV("low-packet-json")) ||
             iree_string_view_equal(target_name, IREE_SV("low-packet"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(target_options, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-packet-json requires a low function symbol "
                              "name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_low_packet_options(option_text, out_request));
    out_request->format = LOOM_CHECK_EMIT_LOW_PACKET_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-compile-report"))) {
    iree_string_view_t symbol_name;
    iree_string_view_t options;
    iree_string_view_split(target_options, ' ', &symbol_name, &options);
    if (!iree_string_view_consume_prefix(&symbol_name, IREE_SV("@")) ||
        iree_string_view_is_empty(symbol_name) ||
        !iree_string_view_is_empty(iree_string_view_trim(options))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-compile-report requires one low function "
                              "symbol name and no options");
    }
    out_request->analysis_symbol_name = symbol_name;
    out_request->format = LOOM_CHECK_EMIT_LOW_COMPILE_REPORT;
    return iree_ok_status();
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown emit target '%.*s'", (int)target_name.size,
                            target_name.data);
  }
}

static iree_status_t loom_check_emit_finish_status_failure(
    iree_status_t failure_status, iree_string_view_t emit_target_name,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  iree_status_t status = iree_string_builder_append_format(
      &result->detail,
      "emit target '%.*s' failed: ", (int)emit_target_name.size,
      emit_target_name.data);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_status(&result->detail, failure_status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, "\n");
  }
  iree_status_free(failure_status);
  return status;
}

static iree_status_t loom_check_emit_finish_diagnostics_and_compare_output(
    loom_check_diagnostic_collector_t* collector,
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result));
  if (result->raw_outcome != LOOM_CHECK_PASS || !result->has_actual_output) {
    return iree_ok_status();
  }
  return loom_check_compare_output(test_case, allocator, result);
}

static iree_status_t loom_check_emit_symbol_not_found(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    iree_string_view_t symbol_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
  };
  return loom_check_diagnostic_collector_emit_case_source(
      diagnostic_collector, test_case, filename, LOOM_EMITTER_PASS,
      LOOM_ERR_SYMBOL_002, params, IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_check_emit_symbol_actual_kind(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol->definition) {
    return loom_symbol_definition_descriptor_name(symbol->definition);
  }
  if (symbol->defining_op) {
    return loom_op_name(module, symbol->defining_op);
  }
  return IREE_SV("<unknown>");
}

static iree_status_t loom_check_emit_symbol_kind_mismatch(
    const loom_module_t* module, const loom_symbol_t* symbol,
    iree_string_view_t symbol_name, iree_string_view_t expected_kind,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
      loom_param_string(loom_check_emit_symbol_actual_kind(module, symbol)),
      loom_param_string(expected_kind),
  };
  const loom_diagnostic_emission_t emission = {
      .op = symbol->defining_op,
      .error = LOOM_ERR_SYMBOL_003,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_check_emit_find_func_like(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, loom_func_like_t* out_function) {
  *out_function = (loom_func_like_t){0};
  loom_string_id_t name_id = loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loom_check_emit_symbol_not_found(diagnostic_collector, test_case,
                                            filename, symbol_name);
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loom_check_emit_symbol_not_found(diagnostic_collector, test_case,
                                            filename, symbol_name);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function) || !loom_func_like_body(function)) {
    return loom_check_emit_symbol_kind_mismatch(
        module, symbol, symbol_name, IREE_SV("function body"), emitter);
  }
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_function_analysis(
    loom_module_t* module, iree_string_view_t symbol_name,
    loom_check_emit_format_t format, const loom_test_case_t* test_case,
    iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_func_like_t function = {0};
  IREE_RETURN_IF_ERROR(
      loom_check_emit_find_func_like(module, symbol_name, test_case, filename,
                                     diagnostic_collector, emitter, &function));
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  if (format == LOOM_CHECK_EMIT_STORAGE_INTERFERENCE) {
    return loom_check_emit_storage_interference(
        module, function, analysis_arena, &result->actual_output);
  }
  return loom_check_emit_liveness(module, function, analysis_arena,
                                  &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_schedule_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_schedule_diagnostic_flags_t diagnostic_flags,
    loom_low_schedule_strategy_t strategy, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* analysis_arena, loom_check_result_t* result) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      module, symbol_name, test_case, filename, diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  loom_low_schedule_options_t options = {
      .emitter = emitter,
      .diagnostic_flags = diagnostic_flags,
      .flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS |
               LOOM_LOW_SCHEDULE_FLAG_RETAIN_PRESSURE_STEPS |
               LOOM_LOW_SCHEDULE_FLAG_RETAIN_DEPENDENCIES,
      .strategy = strategy,
  };
  loom_low_function_model_t model = {0};
  loom_low_schedule_table_t table = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, low_function,
      /*function_target_facts=*/NULL, descriptor_registry, emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REQUIRE_SCHEDULE_MODEL, analysis_arena,
      &model);
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    status =
        loom_low_schedule_function(&model, &options, analysis_arena, &table);
  }
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    status = loom_low_schedule_format_json(&table, &result->actual_output);
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

static const char* loom_check_emit_low_allocation_mode_name(
    uint8_t allocation_mode) {
  switch (allocation_mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return "virtual";
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return "assigned";
    case LOOM_LOW_ALLOCATION_FIXED:
      return "fixed";
    default:
      return "unknown";
  }
}

static void loom_check_emit_low_allocation_summary_record_range(
    uint64_t* inout_count, const loom_low_allocation_assignment_t* assignment) {
  const uint64_t location_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (location_end > *inout_count) {
    *inout_count = location_end;
  }
}

static void loom_check_emit_low_allocation_summary_record_assignment(
    const loom_low_allocation_assignment_t* assignment,
    loom_check_emit_low_allocation_summary_row_t* row) {
  ++row->assignment_count;
  row->unit_count += assignment->unit_count;
  switch (assignment->location_kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
      loom_check_emit_low_allocation_summary_record_range(
          &row->physical_register_count, assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
      loom_check_emit_low_allocation_summary_record_range(&row->target_id_count,
                                                          assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      loom_check_emit_low_allocation_summary_record_range(
          &row->spill_slot_count, assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
    default:
      ++row->unassigned_count;
      break;
  }
}

static iree_status_t loom_check_emit_build_low_allocation_table(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    bool* out_built, loom_low_allocation_table_t* out_table) {
  *out_built = false;
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      module, symbol_name, test_case, filename, diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  bool synthesis_admitted = false;
  IREE_RETURN_IF_ERROR(loom_low_diagnostic_admit_allocation_synthesis(
      module, low_function, emitter, &synthesis_admitted));
  if (!synthesis_admitted) {
    return iree_ok_status();
  }
  loom_low_allocation_options_t options = {
      .emitter = emitter,
  };
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, low_function,
      /*function_target_facts=*/NULL, descriptor_registry, emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, analysis_arena, &model);
  if (iree_status_is_ok(status)) {
    status =
        loom_low_allocate_function(&model, &options, analysis_arena, out_table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_diagnostics_emit(out_table, diagnostic_flags,
                                                  emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_built = true;
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

static iree_status_t loom_check_emit_write_low_allocation_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_emit_build_low_allocation_table(
      module, symbol_name, descriptor_registry, test_case, filename,
      diagnostic_collector, diagnostic_flags, emitter, analysis_arena, &built,
      &table));
  if (!built) {
    return iree_ok_status();
  }
  return loom_low_allocation_format_json(&table, &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_allocation_summary(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_emit_build_low_allocation_table(
      module, symbol_name, descriptor_registry, test_case, filename,
      diagnostic_collector, diagnostic_flags, emitter, analysis_arena, &built,
      &table));
  if (!built || table.error_count != 0) {
    return iree_ok_status();
  }

  const iree_host_size_t register_class_count =
      table.target.descriptor_set->reg_class_count;
  loom_check_emit_low_allocation_summary_row_t* rows = NULL;
  if (register_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        analysis_arena, register_class_count, sizeof(*rows), (void**)&rows));
    memset(rows, 0, register_class_count * sizeof(*rows));
  }

  for (iree_host_size_t i = 0; i < table.assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &table.assignments[i];
    IREE_ASSERT(assignment->descriptor_reg_class_id < register_class_count);
    loom_check_emit_low_allocation_summary_record_assignment(
        assignment, &rows[assignment->descriptor_reg_class_id]);
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(
          &result->actual_output,
          "low-allocation @%.*s\n"
          "target: %.*s\n"
          "descriptor_set: %.*s\n"
          "mode: %s\n"
          "assignments: %" PRIhsz "\n"
          "remarks: %" PRIhsz "\n"
          "spills: %" PRIhsz "\n"
          "copies: coalesced=%" PRIhsz " materialized=%" PRIhsz "\n",
          (int)symbol_name.size, symbol_name.data,
          (int)table.target.target_name.size, table.target.target_name.data,
          (int)table.target.descriptor_set_key.size,
          table.target.descriptor_set_key.data,
          loom_check_emit_low_allocation_mode_name(table.allocation_mode),
          table.assignment_count, table.remark_count, table.spill_count,
          table.coalesced_copy_count, table.materialized_copy_count));

  iree_host_size_t used_register_class_count = 0;
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    if (rows[i].assignment_count != 0) {
      ++used_register_class_count;
    }
  }
  if (used_register_class_count == 0) {
    return iree_string_builder_append_cstring(&result->actual_output,
                                              "classes: none\n");
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&result->actual_output, "classes:\n"));
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    const loom_check_emit_low_allocation_summary_row_t* row = &rows[i];
    if (row->assignment_count == 0) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &table.target.descriptor_set->reg_classes[i];
    const iree_string_view_t register_class_name =
        loom_low_descriptor_set_string(table.target.descriptor_set,
                                       reg_class->name_string_ref);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &result->actual_output,
        "  %.*s: assignments=%" PRIhsz " units=%" PRIu64 " target_ids=%" PRIu64
        " physical=%" PRIu64 " spill_slots=%" PRIu64 " unassigned=%" PRIhsz
        "\n",
        (int)register_class_name.size, register_class_name.data,
        row->assignment_count, row->unit_count, row->target_id_count,
        row->physical_register_count, row->spill_slot_count,
        row->unassigned_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_low_packet_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_schedule_strategy_t strategy,
    loom_target_low_packet_diagnostic_provider_list_t
        packet_diagnostic_provider_list,
    loom_target_low_packet_diagnostic_flags_t packet_diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      module, symbol_name, test_case, filename, diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = descriptor_registry,
      .schedule_strategy = strategy,
      .emitter = emitter,
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(
      loom_low_emission_frame_build(module, low_function, &frame_options,
                                    analysis_arena, &frame, &frame_accepted));
  if (!frame_accepted) {
    return iree_ok_status();
  }
  const loom_target_low_packet_diagnostics_options_t diagnostic_options = {
      .provider_list = packet_diagnostic_provider_list,
      .diagnostic_flags = packet_diagnostic_flags,
      .emitter = emitter,
  };
  loom_target_low_packet_diagnostics_result_t diagnostic_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_low_packet_diagnostics_emit_function(
      &frame, &diagnostic_options, &diagnostic_result));
  return loom_low_packet_format_json(&frame.schedule, &frame.allocation,
                                     &result->actual_output);
}

typedef struct loom_check_emit_native_provider_consumer_t {
  // Provider receiving the projected native module.
  const loom_check_emit_provider_t* provider;
  // Common request owning the admitted public module.
  const loom_check_emit_provider_request_t* request;
} loom_check_emit_native_provider_consumer_t;

static iree_status_t loom_check_emit_invoke_native_provider(
    void* user_data, const loom_check_emit_native_module_t* native_module) {
  const loom_check_emit_native_provider_consumer_t* consumer =
      (const loom_check_emit_native_provider_consumer_t*)user_data;
  return consumer->provider->execute_native(consumer->provider,
                                            consumer->request, native_module);
}

static iree_status_t loom_check_emit_invoke_provider(
    const loom_check_emit_provider_t* provider,
    loom_check_emit_provider_request_t* request) {
  iree_arena_allocator_t case_arena;
  iree_arena_initialize(request->block_pool, &case_arena);
  request->case_arena = &case_arena;
  // Providers own comparable output by default and may explicitly suppress it
  // for diagnostics-only modes.
  request->result->has_actual_output = true;
  iree_status_t status = iree_ok_status();
  if (provider->execute != NULL) {
    status = provider->execute(provider, request);
  } else {
    const loom_check_emit_native_provider_consumer_t native_consumer = {
        .provider = provider,
        .request = request,
    };
    status = loom_check_compile_with_native_module(
        request, loom_check_emit_invoke_native_provider,
        (void*)&native_consumer);
  }
  iree_arena_deinitialize(&case_arena);
  // Successful emission owns a comparable output even when it is empty.
  // Providers must opt in when they produce complete output after a compiler
  // error; partial output remains incomparable by default.
  const bool has_error =
      loom_check_diagnostic_collector_has_error(request->diagnostic_collector);
  const bool has_comparable_error_output =
      request->result->actual_output.size != 0 &&
      iree_any_bit_set(provider->flags,
                       LOOM_CHECK_EMIT_PROVIDER_FLAG_COMPARE_ERROR_OUTPUT);
  request->result->has_actual_output =
      iree_status_is_ok(status) && request->result->has_actual_output &&
      (!has_error || has_comparable_error_output);
  return status;
}

static iree_status_t loom_check_emit_finish_provider(
    iree_status_t status, const loom_check_emit_provider_request_t* request,
    iree_host_size_t case_index, loom_check_file_report_t* report) {
  if (!iree_status_is_ok(status)) {
    return loom_check_emit_finish_status_failure(status, request->target_name,
                                                 request->result);
  }
  return loom_check_emit_finish_diagnostics_and_compare_output(
      request->diagnostic_collector, request->test_case, case_index, report,
      request->host_allocator, request->result);
}

typedef struct loom_check_emit_core_consumer_t {
  // Common request owning the admitted public module and output state.
  const loom_check_emit_provider_request_t* provider_request;
  // Parsed core emit operation.
  const loom_check_emit_request_t* emit_request;
  // Pass diagnostics emitted while producing the core output.
  iree_host_size_t diagnostic_emission_count;
} loom_check_emit_core_consumer_t;

static iree_status_t loom_check_emit_core_native_module(
    void* user_data, const loom_check_emit_native_module_t* native_module) {
  loom_check_emit_core_consumer_t* consumer =
      (loom_check_emit_core_consumer_t*)user_data;
  const loom_check_emit_provider_request_t* provider_request =
      consumer->provider_request;
  const loom_check_emit_request_t* request = consumer->emit_request;
  loom_check_diagnostic_collector_t* diagnostic_collector =
      provider_request->diagnostic_collector;
  loom_module_t* module = native_module->module;
  loom_check_result_t* result = provider_request->result;
  loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
      .diagnostic_collector = diagnostic_collector,
      .module = module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &pass_diagnostic_capture,
  };
  const iree_host_size_t actual_output_size = result->actual_output.size;
  iree_status_t status = iree_ok_status();
  if (request->format == LOOM_CHECK_EMIT_LIVENESS_JSON ||
      request->format == LOOM_CHECK_EMIT_STORAGE_INTERFERENCE) {
    status = loom_check_emit_write_function_analysis(
        module, request->analysis_symbol_name, request->format,
        provider_request->test_case, provider_request->filename,
        diagnostic_collector, emitter, diagnostic_collector->arena, result);
  } else if (request->format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON) {
    status = loom_check_emit_write_low_schedule_json(
        native_module->module, request->analysis_symbol_name,
        &provider_request->low_registry->registry, provider_request->test_case,
        provider_request->filename, diagnostic_collector,
        request->low_schedule_diagnostic_flags, request->low_schedule_strategy,
        emitter, diagnostic_collector->arena, result);
  } else if (request->format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON) {
    status = loom_check_emit_write_low_allocation_json(
        native_module->module, request->analysis_symbol_name,
        &provider_request->low_registry->registry, provider_request->test_case,
        provider_request->filename, diagnostic_collector,
        request->low_allocation_diagnostic_flags, emitter,
        diagnostic_collector->arena, result);
  } else if (request->format == LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY) {
    status = loom_check_emit_write_low_allocation_summary(
        native_module->module, request->analysis_symbol_name,
        &provider_request->low_registry->registry, provider_request->test_case,
        provider_request->filename, diagnostic_collector,
        request->low_allocation_diagnostic_flags, emitter,
        diagnostic_collector->arena, result);
  } else if (request->format == LOOM_CHECK_EMIT_LOW_COMPILE_REPORT) {
    status = loom_check_emit_low_report(
        native_module->module, request->analysis_symbol_name,
        &provider_request->low_registry->registry, provider_request->test_case,
        provider_request->filename, diagnostic_collector, emitter,
        diagnostic_collector->arena, result);
  } else if (request->format == LOOM_CHECK_EMIT_LOW_PACKET_JSON) {
    status = loom_check_emit_write_low_packet_json(
        native_module->module, request->analysis_symbol_name,
        &provider_request->low_registry->registry, provider_request->test_case,
        provider_request->filename, diagnostic_collector,
        request->low_schedule_strategy,
        loom_target_environment_low_packet_diagnostic_provider_list(
            provider_request->environment->target_environment),
        request->low_packet_diagnostic_flags, emitter,
        diagnostic_collector->arena, result);
  } else {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "core emit target '%.*s' was parsed but not handled",
        (int)request->emit_target_name.size, request->emit_target_name.data);
  }

  consumer->diagnostic_emission_count = pass_diagnostic_capture.emission_count;
  if (iree_status_is_ok(status)) {
    if (request->suppress_actual_output) {
      result->actual_output.size = actual_output_size;
      if (result->actual_output.buffer != NULL &&
          result->actual_output.capacity > actual_output_size) {
        result->actual_output.buffer[actual_output_size] = 0;
      }
    } else if (result->actual_output.size != actual_output_size) {
      result->has_actual_output = true;
    }
  }
  return status;
}

static void loom_check_emit_release_public_module(
    loom_check_emit_provider_request_t* request) {
  loomc_module_release(request->public_module);
  request->public_module = NULL;
  loom_check_compile_session_t* session = request->environment->compile_session;
  if (session->workspace != NULL) {
    loomc_workspace_trim(session->workspace);
  }
}

iree_status_t loom_check_execute_emit(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };

  iree_string_view_t provider_target_name = iree_string_view_empty();
  iree_string_view_t provider_target_options = iree_string_view_empty();
  loom_check_emit_split_target(test_case->emit_target, &provider_target_name,
                               &provider_target_options);
  const loom_check_emit_provider_t* provider =
      loom_check_environment_lookup_emit_provider(environment,
                                                  provider_target_name);
  iree_status_t status = iree_ok_status();
  if (provider == NULL &&
      !loom_check_emit_core_target_matches(provider_target_name)) {
    status = loom_check_emit_fail_unknown_target(environment,
                                                 provider_target_name, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_check_emit_request_t request;
  if (provider == NULL) {
    status = loom_check_emit_parse_request(test_case->emit_target, &request);
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
  } else {
    request = (loom_check_emit_request_t){
        .format = LOOM_CHECK_EMIT_LIVENESS_JSON,
        .emit_target_name = provider_target_name,
    };
  }
  const loom_target_low_descriptor_registry_t low_registry =
      loom_target_environment_low_descriptor_registry(
          environment->target_environment);
  loom_low_descriptor_text_print_context_initialize(
      &low_registry.registry, &diagnostic_collector.type_print_context);
  if (provider != NULL &&
      ((provider->execute == NULL) == (provider->execute_native == NULL))) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "emit provider '%.*s' must define exactly one execute callback",
        (int)provider->name.size, provider->name.data);
    status = loom_check_emit_finish_status_failure(
        status, request.emit_target_name, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  loom_check_emit_provider_request_t provider_request = {
      .emit_target = test_case->emit_target,
      .target_name = provider_target_name,
      .target_options = provider_target_options,
      .filename = filename,
      .test_case = test_case,
      .environment = environment,
      .low_registry = &low_registry,
      .diagnostic_collector = &diagnostic_collector,
      .block_pool = block_pool,
      .host_allocator = allocator,
      .result = result,
  };
  status = loom_check_compile_admit_module(
      test_case, filename, input_request, environment->compile_session,
      environment, /*max_errors=*/0, &diagnostic_collector, block_pool,
      allocator, &provider_request.public_module);
  if (!iree_status_is_ok(status)) {
    loom_check_emit_release_public_module(&provider_request);
    status = loom_check_emit_finish_status_failure(
        status, request.emit_target_name, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (provider_request.public_module == NULL ||
      loom_check_diagnostic_collector_has_error(&diagnostic_collector)) {
    loom_check_emit_release_public_module(&provider_request);
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (provider != NULL) {
    status = loom_check_emit_invoke_provider(provider, &provider_request);
    loom_check_emit_release_public_module(&provider_request);
    status = loom_check_emit_finish_provider(status, &provider_request,
                                             case_index, report);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_check_emit_core_consumer_t core_consumer = {
      .provider_request = &provider_request,
      .emit_request = &request,
  };
  status = loom_check_compile_with_native_module(
      &provider_request, loom_check_emit_core_native_module, &core_consumer);
  loom_check_emit_release_public_module(&provider_request);
  if (!iree_status_is_ok(status)) {
    status = loom_check_emit_finish_status_failure(
        status, request.emit_target_name, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (test_case->annotation_count > 0 ||
      core_consumer.diagnostic_emission_count > 0 ||
      diagnostic_collector.count > 0) {
    status = loom_check_emit_finish_diagnostics_and_compare_output(
        &diagnostic_collector, test_case, case_index, report, allocator,
        result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (request.suppress_actual_output) {
    result->raw_outcome = LOOM_CHECK_PASS;
    status = iree_ok_status();
  } else {
    status = loom_check_compare_output(test_case, allocator, result);
  }
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}
