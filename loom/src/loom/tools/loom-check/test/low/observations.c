// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete Low planning observations for authored compiler tests.

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_json.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/packet_json.h"
#include "loom/codegen/low/schedule/json.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/low_packet_diagnostics.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/format.h"
#include "loom/target/reporting/low.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/low/native_emit.h"
#include "loom/tools/loom-check/test/low/providers.h"

typedef enum loom_check_test_low_observation_e {
  LOOM_CHECK_TEST_LOW_OBSERVATION_SCHEDULE_JSON = 0,
  LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_JSON,
  LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_SUMMARY,
  LOOM_CHECK_TEST_LOW_OBSERVATION_PACKET_JSON,
  LOOM_CHECK_TEST_LOW_OBSERVATION_COMPILE_REPORT,
} loom_check_test_low_observation_t;

typedef struct loom_check_test_low_allocation_summary_row_t {
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
} loom_check_test_low_allocation_summary_row_t;

typedef struct loom_check_test_low_observation_options_t {
  // Complete observation selected by the emit target name.
  loom_check_test_low_observation_t observation;
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Low allocation diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // True once an allocation diagnostics option has been parsed.
  bool has_allocation_diagnostics_option;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t schedule_diagnostic_flags;
  // True once a scheduler diagnostics option has been parsed.
  bool has_schedule_diagnostics_option;
  // Low scheduler candidate-selection strategy requested by the RUN line.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a scheduler strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Target-low packet diagnostics requested by the RUN line.
  loom_target_low_packet_diagnostic_flags_t packet_diagnostic_flags;
  // True once a packet diagnostics option has been parsed.
  bool has_packet_diagnostics_option;
  // True when the observation should run without comparable output.
  bool suppress_actual_output;
  // True once an output option has been parsed.
  bool has_output_option;
} loom_check_test_low_observation_options_t;

static bool loom_check_test_low_observation_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("low-schedule-json")) ||
         iree_string_view_equal(target_name, IREE_SV("low-schedule")) ||
         iree_string_view_equal(target_name, IREE_SV("low-allocation-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("low-allocation-summary")) ||
         iree_string_view_equal(target_name, IREE_SV("low-allocation")) ||
         iree_string_view_equal(target_name, IREE_SV("low-packet-json")) ||
         iree_string_view_equal(target_name, IREE_SV("low-packet")) ||
         iree_string_view_equal(target_name, IREE_SV("low-compile-report"));
}

static iree_status_t loom_check_test_low_observation_parse_json_output(
    iree_string_view_t target_name, iree_string_view_t value,
    loom_check_test_low_observation_options_t* options) {
  if (options->has_output_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate %.*s option 'output'",
                            (int)target_name.size, target_name.data);
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    options->suppress_actual_output = false;
  } else if (iree_string_view_equal(value, IREE_SV("none"))) {
    options->suppress_actual_output = true;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s option 'output' expected 'json' or 'none', "
                            "got '%.*s'",
                            (int)target_name.size, target_name.data,
                            (int)value.size, value.data);
  }
  options->has_output_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_test_low_observation_parse_allocation_option(
    iree_string_view_t target_name, iree_string_view_t token,
    loom_check_test_low_observation_options_t* options) {
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
    return loom_check_test_low_observation_parse_json_output(target_name, value,
                                                             options);
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown %.*s option '%.*s'", (int)target_name.size,
                            target_name.data, (int)name.size, name.data);
  }
  if (options->has_allocation_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate %.*s option 'diagnostics'",
                            (int)target_name.size, target_name.data);
  }
  IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_allocation_diagnostics(
      value, target_name, &options->allocation_diagnostic_flags));
  options->has_allocation_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_test_low_observation_parse_schedule_option(
    iree_string_view_t token,
    loom_check_test_low_observation_options_t* options) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_test_low_observation_parse_json_output(
        IREE_SV("low-schedule-json"), value, options);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-json option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("low-schedule-json"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low-schedule-json option '%.*s'",
                            (int)name.size, name.data);
  }
  if (options->has_schedule_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate low-schedule-json option 'diagnostics'");
  }
  IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_diagnostics(
      value, IREE_SV("low-schedule-json"),
      &options->schedule_diagnostic_flags));
  options->has_schedule_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_test_low_observation_parse_packet_option(
    iree_string_view_t token,
    loom_check_test_low_observation_options_t* options) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_test_low_observation_parse_json_output(
        IREE_SV("low-packet-json"), value, options);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-packet-json option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("low-packet-json"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    if (options->has_packet_diagnostics_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-packet-json option "
                              "'diagnostics'");
    }
    if (iree_string_view_equal(value, IREE_SV("none"))) {
      options->packet_diagnostic_flags = 0;
    } else if (iree_string_view_equal(value, IREE_SV("packets"))) {
      options->packet_diagnostic_flags =
          LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_TARGET_PACKETS;
    } else if (iree_string_view_equal(value, IREE_SV("all"))) {
      options->packet_diagnostic_flags = LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_ALL;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-packet-json option 'diagnostics' expected 'none', 'packets', "
          "or 'all', got '%.*s'",
          (int)value.size, value.data);
    }
    options->has_packet_diagnostics_option = true;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown low-packet-json option '%.*s'",
                          (int)name.size, name.data);
}

typedef iree_status_t (*loom_check_test_low_parse_option_fn_t)(
    iree_string_view_t token,
    loom_check_test_low_observation_options_t* options);

static iree_status_t loom_check_test_low_observation_parse_options(
    iree_string_view_t text, loom_check_test_low_parse_option_fn_t parse_option,
    loom_check_test_low_observation_options_t* options) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(parse_option(token, options));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_low_observation_parse_symbol_and_options(
    iree_string_view_t target_name, iree_string_view_t target_options,
    iree_string_view_t* out_symbol_name, iree_string_view_t* out_option_text) {
  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_split(target_options, ' ', &symbol_name, out_option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  *out_option_text = iree_string_view_trim(*out_option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s requires a low function symbol name",
                            (int)target_name.size, target_name.data);
  }
  *out_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(*out_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low function symbol name is required");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_low_observation_parse(
    const loom_check_emit_provider_request_t* request,
    loom_check_test_low_observation_options_t* out_options) {
  *out_options = (loom_check_test_low_observation_options_t){
      .observation = LOOM_CHECK_TEST_LOW_OBSERVATION_SCHEDULE_JSON,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("low-compile-report"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t options = iree_string_view_empty();
    iree_string_view_split(request->target_options, ' ', &symbol_name,
                           &options);
    if (!iree_string_view_consume_prefix(&symbol_name, IREE_SV("@")) ||
        iree_string_view_is_empty(symbol_name) ||
        !iree_string_view_is_empty(iree_string_view_trim(options))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-compile-report requires one low function "
                              "symbol name and no options");
    }
    out_options->observation = LOOM_CHECK_TEST_LOW_OBSERVATION_COMPILE_REPORT;
    out_options->function_symbol_name = symbol_name;
    return iree_ok_status();
  }

  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_t diagnostic_target_name = request->target_name;
  if (iree_string_view_equal(request->target_name, IREE_SV("low-schedule"))) {
    diagnostic_target_name = IREE_SV("low-schedule-json");
  } else if (iree_string_view_equal(request->target_name,
                                    IREE_SV("low-packet"))) {
    diagnostic_target_name = IREE_SV("low-packet-json");
  }
  IREE_RETURN_IF_ERROR(loom_check_test_low_observation_parse_symbol_and_options(
      diagnostic_target_name, request->target_options,
      &out_options->function_symbol_name, &option_text));

  if (iree_string_view_equal(request->target_name,
                             IREE_SV("low-schedule-json")) ||
      iree_string_view_equal(request->target_name, IREE_SV("low-schedule"))) {
    out_options->observation = LOOM_CHECK_TEST_LOW_OBSERVATION_SCHEDULE_JSON;
    return loom_check_test_low_observation_parse_options(
        option_text, loom_check_test_low_observation_parse_schedule_option,
        out_options);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("low-allocation-json")) ||
      iree_string_view_equal(request->target_name,
                             IREE_SV("low-allocation-summary")) ||
      iree_string_view_equal(request->target_name, IREE_SV("low-allocation"))) {
    out_options->observation =
        iree_string_view_equal(request->target_name,
                               IREE_SV("low-allocation-json"))
            ? LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_JSON
            : LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_SUMMARY;
    option_text = iree_string_view_trim(option_text);
    while (!iree_string_view_is_empty(option_text)) {
      iree_string_view_t token = iree_string_view_empty();
      iree_string_view_t remaining = iree_string_view_empty();
      iree_string_view_split(option_text, ' ', &token, &remaining);
      token = iree_string_view_trim(token);
      if (!iree_string_view_is_empty(token)) {
        IREE_RETURN_IF_ERROR(
            loom_check_test_low_observation_parse_allocation_option(
                request->target_name, token, out_options));
      }
      option_text = iree_string_view_trim(remaining);
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("low-packet-json")) ||
      iree_string_view_equal(request->target_name, IREE_SV("low-packet"))) {
    out_options->observation = LOOM_CHECK_TEST_LOW_OBSERVATION_PACKET_JSON;
    return loom_check_test_low_observation_parse_options(
        option_text, loom_check_test_low_observation_parse_packet_option,
        out_options);
  }

  return iree_make_status(
      IREE_STATUS_INTERNAL, "unhandled Low observation target '%.*s'",
      (int)request->target_name.size, request->target_name.data);
}

static iree_status_t loom_check_test_low_observe_schedule(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module, options->function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  const loom_low_schedule_options_t schedule_options = {
      .emitter = emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS |
               LOOM_LOW_SCHEDULE_FLAG_RETAIN_PRESSURE_STEPS |
               LOOM_LOW_SCHEDULE_FLAG_RETAIN_DEPENDENCIES,
      .strategy = options->schedule_strategy,
  };
  loom_low_function_model_t model = {0};
  loom_low_schedule_table_t table = {0};
  iree_status_t status = loom_low_function_model_initialize(
      native_module->module, low_function,
      /*function_target_facts=*/NULL, &request->low_registry->registry, emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REQUIRE_SCHEDULE_MODEL, request->case_arena,
      &model);
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    status = loom_low_schedule_function(&model, &schedule_options,
                                        request->case_arena, &table);
  }
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    status =
        loom_low_schedule_format_json(&table, &request->result->actual_output);
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

static const char* loom_check_test_low_allocation_mode_name(
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

static void loom_check_test_low_allocation_summary_record_range(
    uint64_t* inout_count, const loom_low_allocation_assignment_t* assignment) {
  const uint64_t location_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (location_end > *inout_count) {
    *inout_count = location_end;
  }
}

static void loom_check_test_low_allocation_summary_record_assignment(
    const loom_low_allocation_assignment_t* assignment,
    loom_check_test_low_allocation_summary_row_t* row) {
  ++row->assignment_count;
  row->unit_count += assignment->unit_count;
  switch (assignment->location_kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
      loom_check_test_low_allocation_summary_record_range(
          &row->physical_register_count, assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
      loom_check_test_low_allocation_summary_record_range(&row->target_id_count,
                                                          assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      loom_check_test_low_allocation_summary_record_range(
          &row->spill_slot_count, assignment);
      break;
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
    default:
      ++row->unassigned_count;
      break;
  }
}

static iree_status_t loom_check_test_low_build_allocation(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter, bool* out_built,
    loom_low_allocation_table_t* out_table) {
  *out_built = false;
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module, options->function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  bool synthesis_admitted = false;
  IREE_RETURN_IF_ERROR(loom_low_diagnostic_admit_allocation_synthesis(
      native_module->module, low_function, emitter, &synthesis_admitted));
  if (!synthesis_admitted) {
    return iree_ok_status();
  }
  const loom_low_allocation_options_t allocation_options = {
      .emitter = emitter,
  };
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      native_module->module, low_function,
      /*function_target_facts=*/NULL, &request->low_registry->registry, emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, request->case_arena, &model);
  if (iree_status_is_ok(status)) {
    status = loom_low_allocate_function(&model, &allocation_options,
                                        request->case_arena, out_table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_diagnostics_emit(
        out_table, options->allocation_diagnostic_flags, emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_built = true;
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

static iree_status_t loom_check_test_low_observe_allocation_json(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_test_low_build_allocation(
      request, native_module, options, emitter, &built, &table));
  if (!built) {
    return iree_ok_status();
  }
  return loom_low_allocation_format_json(&table,
                                         &request->result->actual_output);
}

static iree_status_t loom_check_test_low_observe_allocation_summary(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_test_low_build_allocation(
      request, native_module, options, emitter, &built, &table));
  if (!built || table.error_count != 0) {
    return iree_ok_status();
  }

  const iree_host_size_t register_class_count =
      table.target.descriptor_set->reg_class_count;
  loom_check_test_low_allocation_summary_row_t* rows = NULL;
  if (register_class_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(request->case_arena, register_class_count,
                                  sizeof(*rows), (void**)&rows));
    memset(rows, 0, register_class_count * sizeof(*rows));
  }

  for (iree_host_size_t i = 0; i < table.assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &table.assignments[i];
    IREE_ASSERT(assignment->descriptor_reg_class_id < register_class_count);
    loom_check_test_low_allocation_summary_record_assignment(
        assignment, &rows[assignment->descriptor_reg_class_id]);
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(
          &request->result->actual_output,
          "low-allocation @%.*s\n"
          "target: %.*s\n"
          "descriptor_set: %.*s\n"
          "mode: %s\n"
          "assignments: %" PRIhsz "\n"
          "remarks: %" PRIhsz "\n"
          "spills: %" PRIhsz "\n"
          "copies: coalesced=%" PRIhsz " materialized=%" PRIhsz "\n",
          (int)options->function_symbol_name.size,
          options->function_symbol_name.data,
          (int)table.target.target_name.size, table.target.target_name.data,
          (int)table.target.descriptor_set_key.size,
          table.target.descriptor_set_key.data,
          loom_check_test_low_allocation_mode_name(table.allocation_mode),
          table.assignment_count, table.remark_count, table.spill_count,
          table.coalesced_copy_count, table.materialized_copy_count));

  iree_host_size_t used_register_class_count = 0;
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    if (rows[i].assignment_count != 0) {
      ++used_register_class_count;
    }
  }
  if (used_register_class_count == 0) {
    return iree_string_builder_append_cstring(&request->result->actual_output,
                                              "classes: none\n");
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &request->result->actual_output, "classes:\n"));
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    const loom_check_test_low_allocation_summary_row_t* row = &rows[i];
    if (row->assignment_count == 0) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &table.target.descriptor_set->reg_classes[i];
    const iree_string_view_t register_class_name =
        loom_low_descriptor_set_string(table.target.descriptor_set,
                                       reg_class->name_string_ref);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &request->result->actual_output,
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

static iree_status_t loom_check_test_low_observe_packet(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module, options->function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &request->low_registry->registry,
      .schedule_strategy = options->schedule_strategy,
      .emitter = emitter,
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      native_module->module, low_function, &frame_options, request->case_arena,
      &frame, &frame_accepted));
  if (!frame_accepted) {
    return iree_ok_status();
  }
  const loom_target_low_packet_diagnostics_options_t diagnostic_options = {
      .provider_list =
          loom_target_environment_low_packet_diagnostic_provider_list(
              request->environment->target_environment),
      .diagnostic_flags = options->packet_diagnostic_flags,
      .emitter = emitter,
  };
  loom_target_low_packet_diagnostics_result_t diagnostic_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_low_packet_diagnostics_emit_function(
      &frame, &diagnostic_options, &diagnostic_result));
  return loom_low_packet_format_json(&frame.schedule, &frame.allocation,
                                     &request->result->actual_output);
}

static iree_status_t loom_check_test_low_observe_compile_report(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    const loom_check_test_low_observation_options_t* options,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module, options->function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }

  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &request->low_registry->registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .emitter = emitter,
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      native_module->module, low_function, &frame_options, request->case_arena,
      &frame, &frame_accepted));
  if (!frame_accepted) {
    return iree_ok_status();
  }

  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(
      &report, iree_arena_allocator(request->case_arena));
  iree_status_t status =
      loom_target_compile_report_record_low_emission_frame(&report, &frame);
  if (iree_status_is_ok(status)) {
    loom_target_compile_report_format_options_t format_options;
    loom_target_compile_report_format_options_initialize(&format_options);
    format_options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    status = loom_target_compile_report_format_text(
        &report, &format_options, &request->result->actual_output);
  }
  loom_target_compile_report_deinitialize(&report);
  return status;
}

static iree_status_t loom_check_test_low_observation_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module) {
  (void)provider;
  loom_check_test_low_observation_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_check_test_low_observation_parse(request, &options));
  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = native_module->module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &capture,
  };
  const iree_host_size_t initial_output_size =
      request->result->actual_output.size;
  iree_status_t status = iree_ok_status();
  switch (options.observation) {
    case LOOM_CHECK_TEST_LOW_OBSERVATION_SCHEDULE_JSON:
      status = loom_check_test_low_observe_schedule(request, native_module,
                                                    &options, emitter);
      break;
    case LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_JSON:
      status = loom_check_test_low_observe_allocation_json(
          request, native_module, &options, emitter);
      break;
    case LOOM_CHECK_TEST_LOW_OBSERVATION_ALLOCATION_SUMMARY:
      status = loom_check_test_low_observe_allocation_summary(
          request, native_module, &options, emitter);
      break;
    case LOOM_CHECK_TEST_LOW_OBSERVATION_PACKET_JSON:
      status = loom_check_test_low_observe_packet(request, native_module,
                                                  &options, emitter);
      break;
    case LOOM_CHECK_TEST_LOW_OBSERVATION_COMPILE_REPORT:
      status = loom_check_test_low_observe_compile_report(
          request, native_module, &options, emitter);
      break;
  }
  if (iree_status_is_ok(status) && options.suppress_actual_output) {
    request->result->actual_output.size = initial_output_size;
    if (request->result->actual_output.buffer != NULL &&
        request->result->actual_output.capacity > initial_output_size) {
      request->result->actual_output.buffer[initial_output_size] = 0;
    }
    request->result->has_actual_output = false;
  }
  return status;
}

static iree_status_t loom_check_test_low_observation_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder,
      "low-schedule-json, low-schedule, low-allocation-json, "
      "low-allocation-summary, low-allocation, low-packet-json, low-packet, "
      "low-compile-report");
}

const loom_check_test_native_emit_provider_t
    loom_check_test_low_observation_provider = {
        .base =
            {
                .name = IREE_SVL("low-observation"),
                .flags = LOOM_CHECK_EMIT_PROVIDER_FLAG_COMPARE_ERROR_OUTPUT,
                .match = loom_check_test_low_observation_matches,
                .execute = loom_check_test_native_emit_provider_execute,
                .append_names = loom_check_test_low_observation_append_names,
            },
        .execute_native = loom_check_test_low_observation_execute,
};
