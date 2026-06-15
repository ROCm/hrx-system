// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/liveness.h"
#include "loom/analysis/liveness_json.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_json.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet_json.h"
#include "loom/codegen/low/schedule/json.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/entry_selection.h"
#include "loom/target/low_packet_diagnostics.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/low_emit.h"
#include "loom/tools/loom-check/target_low_registry_manifest.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

typedef enum loom_check_emit_format_e {
  LOOM_CHECK_EMIT_LIVENESS_JSON = 0,
  LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON = 1,
  LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST = 2,
  LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON = 3,
  LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY = 4,
  LOOM_CHECK_EMIT_LOW_PACKET_JSON = 5,
  LOOM_CHECK_EMIT_SOURCE_LOW_TEXT = 6,
} loom_check_emit_format_t;

typedef enum loom_check_emit_source_low_output_e {
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE = 0,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW = 1,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE = 2,
  LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE = 3,
} loom_check_emit_source_low_output_t;

enum {
  LOOM_CHECK_EMIT_MAX_SCHEDULE_PRESSURE_CLIFFS = 16,
};

typedef struct loom_check_emit_pressure_cliff_spec_t {
  // Stable register-class name.
  iree_string_view_t register_class;
  // Live allocation units at which this cliff is crossed.
  uint32_t cliff_units;
  // Occupancy or throughput tier before crossing the cliff.
  uint32_t tier_before;
  // Occupancy or throughput tier after crossing the cliff.
  uint32_t tier_after;
} loom_check_emit_pressure_cliff_spec_t;

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
    IREE_SVL("liveness-json"),       IREE_SVL("liveness"),
    IREE_SVL("low-schedule-json"),   IREE_SVL("low-schedule"),
    IREE_SVL("low-allocation-json"), IREE_SVL("low-allocation-summary"),
    IREE_SVL("low-allocation"),      IREE_SVL("low-packet-json"),
    IREE_SVL("low-packet"),          IREE_SVL("target-low-registry-manifest"),
    IREE_SVL("source-low"),          IREE_SVL("source-to-low"),
};

typedef struct loom_check_emit_request_t {
  // Serialized target form to produce before comparison.
  loom_check_emit_format_t format;
  // Canonical/user-facing emit target name used in diagnostics.
  iree_string_view_t emit_target_name;
  // Module-local function symbol name used by analysis dumps.
  iree_string_view_t analysis_symbol_name;
  // Low allocation budget overrides parsed from the RUN line.
  loom_low_allocation_budget_t
      low_allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |low_allocation_budgets|.
  iree_host_size_t low_allocation_budget_count;
  // Fixed low allocation requests parsed from the RUN line.
  loom_check_low_emit_fixed_value_spec_t low_allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |low_allocation_fixed_value_specs|.
  iree_host_size_t low_allocation_fixed_value_spec_count;
  // Low allocation diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t low_allocation_diagnostic_flags;
  // True once a low allocation diagnostics option has been parsed.
  bool has_low_allocation_diagnostics_option;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t low_schedule_diagnostic_flags;
  // Low scheduler pressure cliffs parsed from the RUN line.
  loom_check_emit_pressure_cliff_spec_t low_schedule_pressure_cliff_specs
      [LOOM_CHECK_EMIT_MAX_SCHEDULE_PRESSURE_CLIFFS];
  // Number of entries in |low_schedule_pressure_cliff_specs|.
  iree_host_size_t low_schedule_pressure_cliff_spec_count;
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
  // Source-low text output form.
  loom_check_emit_source_low_output_t source_low_output;
  // True once a source-low output option has been parsed.
  bool has_source_low_output_option;
  // Source-low legality diagnostics requested by the RUN line.
  loom_target_low_legality_diagnostic_flags_t source_low_diagnostic_flags;
  // True once a source-low diagnostics option has been parsed.
  bool has_source_low_diagnostics_option;
  // Source-low control-flow shape requested by the RUN line.
  loom_target_control_flow_lowering_t source_low_control_flow_lowering;
  // True once a source-low control-flow option has been parsed.
  bool has_source_low_control_flow_option;
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
    return loom_check_low_emit_parse_allocation_option(
        token, target_name, request->low_allocation_budgets,
        IREE_ARRAYSIZE(request->low_allocation_budgets),
        &request->low_allocation_budget_count,
        request->low_allocation_fixed_value_specs,
        IREE_ARRAYSIZE(request->low_allocation_fixed_value_specs),
        &request->low_allocation_fixed_value_spec_count);
  }
  if (request->has_low_allocation_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate %.*s option 'diagnostics'",
                            (int)target_name.size, target_name.data);
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->low_allocation_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("predicted-spills"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS;
  } else if (iree_string_view_equal(value, IREE_SV("copy-decisions"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("placement-decisions"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS |
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS |
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s option 'diagnostics' expected 'none' or "
        "'predicted-spills', 'copy-decisions', 'placement-decisions', or "
        "'all', got '%.*s'",
        (int)target_name.size, target_name.data, (int)value.size, value.data);
  }
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

static iree_status_t loom_check_emit_parse_source_low_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    if (request->has_source_low_diagnostics_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'diagnostics'");
    }
    if (iree_string_view_equal(value, IREE_SV("none"))) {
      request->source_low_diagnostic_flags = 0;
    } else if (iree_string_view_equal(value, IREE_SV("memory"))) {
      request->source_low_diagnostic_flags =
          LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS;
    } else if (iree_string_view_equal(value, IREE_SV("all"))) {
      request->source_low_diagnostic_flags =
          LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low option 'diagnostics' expected 'none', 'memory', or "
          "'all', got '%.*s'",
          (int)value.size, value.data);
    }
    request->has_source_low_diagnostics_option = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("control-flow"))) {
    if (request->has_source_low_control_flow_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'control-flow'");
    }
    if (iree_string_view_equal(value, IREE_SV("cfg"))) {
      request->source_low_control_flow_lowering =
          LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
    } else if (iree_string_view_equal(value, IREE_SV("structured-low"))) {
      request->source_low_control_flow_lowering =
          LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low option 'control-flow' expected 'cfg' or "
          "'structured-low', got '%.*s'",
          (int)value.size, value.data);
    }
    request->has_source_low_control_flow_option = true;
    return iree_ok_status();
  }
  if (!iree_string_view_equal(name, IREE_SV("output"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown source-low option '%.*s'", (int)name.size,
                            name.data);
  }
  if (request->has_source_low_output_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate source-low option 'output'");
  }
  if (iree_string_view_equal(value, IREE_SV("module"))) {
    request->source_low_output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE;
  } else if (iree_string_view_equal(value, IREE_SV("low"))) {
    request->source_low_output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW;
  } else if (iree_string_view_equal(value, IREE_SV("pipeline"))) {
    request->source_low_output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE;
  } else if (iree_string_view_equal(value, IREE_SV("prepared-pipeline"))) {
    request->source_low_output =
        LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low option 'output' expected 'module', 'low', 'pipeline', or "
        "'prepared-pipeline', got '%.*s'",
        (int)value.size, value.data);
  }
  request->has_source_low_output_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_source_low_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_source_low_option(token, request));
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
  if (iree_string_view_equal(name, IREE_SV("cliff"))) {
    if (request->low_schedule_pressure_cliff_spec_count >=
        IREE_ARRAYSIZE(request->low_schedule_pressure_cliff_specs)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many low-schedule-json pressure cliffs");
    }
    iree_string_view_t register_class = iree_string_view_empty();
    iree_string_view_t units_and_tiers = iree_string_view_empty();
    iree_string_view_split(value, ':', &register_class, &units_and_tiers);
    register_class = iree_string_view_trim(register_class);
    units_and_tiers = iree_string_view_trim(units_and_tiers);

    iree_string_view_t units_text = iree_string_view_empty();
    iree_string_view_t tiers_text = iree_string_view_empty();
    iree_string_view_split(units_and_tiers, ':', &units_text, &tiers_text);
    units_text = iree_string_view_trim(units_text);
    tiers_text = iree_string_view_trim(tiers_text);

    iree_string_view_t tier_before_text = iree_string_view_empty();
    iree_string_view_t tier_after_text = iree_string_view_empty();
    iree_string_view_split(tiers_text, ':', &tier_before_text,
                           &tier_after_text);
    tier_before_text = iree_string_view_trim(tier_before_text);
    tier_after_text = iree_string_view_trim(tier_after_text);
    if (iree_string_view_is_empty(register_class) ||
        iree_string_view_is_empty(units_text) ||
        iree_string_view_is_empty(tier_before_text) ||
        iree_string_view_is_empty(tier_after_text) ||
        iree_string_view_find_char(tier_after_text, ':', 0) !=
            IREE_STRING_VIEW_NPOS) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-schedule-json option 'cliff' must have the form "
          "<register-class>:<units>:<tier-before>:<tier-after>");
    }
    uint32_t cliff_units = 0;
    uint32_t tier_before = 0;
    uint32_t tier_after = 0;
    if (!iree_string_view_atoi_uint32(units_text, &cliff_units) ||
        !iree_string_view_atoi_uint32(tier_before_text, &tier_before) ||
        !iree_string_view_atoi_uint32(tier_after_text, &tier_after)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid low-schedule-json pressure cliff");
    }
    loom_check_emit_pressure_cliff_spec_t* spec =
        &request->low_schedule_pressure_cliff_specs
             [request->low_schedule_pressure_cliff_spec_count++];
    spec->register_class = register_class;
    spec->cliff_units = cliff_units;
    spec->tier_before = tier_before;
    spec->tier_after = tier_after;
    return iree_ok_status();
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
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->low_schedule_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS;
  } else if (iree_string_view_equal(value, IREE_SV("resources"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS;
  } else if (iree_string_view_equal(value, IREE_SV("hazards"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS;
  } else if (iree_string_view_equal(value, IREE_SV("candidates"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("model"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-json option 'diagnostics' expected 'none', 'pressure', "
        "'resources', 'hazards', 'candidates', 'model', or 'all', got '%.*s'",
        (int)value.size, value.data);
  }
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
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("low-packet-json"), request->low_allocation_budgets,
      IREE_ARRAYSIZE(request->low_allocation_budgets),
      &request->low_allocation_budget_count,
      request->low_allocation_fixed_value_specs,
      IREE_ARRAYSIZE(request->low_allocation_fixed_value_specs),
      &request->low_allocation_fixed_value_spec_count);
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
      .low_allocation_budget_count = 0,
      .low_allocation_fixed_value_spec_count = 0,
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
      .source_low_output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE,
      .has_source_low_output_option = false,
      .source_low_control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .has_source_low_control_flow_option = false,
  };
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t target_options = iree_string_view_empty();
  loom_check_emit_split_target(emit_target, &target_name, &target_options);
  if (!iree_string_view_is_empty(target_name)) {
    out_request->emit_target_name = target_name;
  }

  if (iree_string_view_equal(target_name, IREE_SV("liveness-json")) ||
      iree_string_view_equal(target_name, IREE_SV("liveness"))) {
    if (!iree_string_view_starts_with(target_options, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "liveness-json requires a function symbol name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(target_options, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function symbol name is required");
    }
    out_request->format = LOOM_CHECK_EMIT_LIVENESS_JSON;
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
                                    IREE_SV("target-low-registry-manifest"))) {
    if (!iree_string_view_is_empty(target_options)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target-low registry manifest does not accept target options");
    }
    out_request->format = LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name, IREE_SV("source-low")) ||
             iree_string_view_equal(target_name, IREE_SV("source-to-low"))) {
    iree_string_view_t first_token = iree_string_view_empty();
    iree_string_view_t unused_remaining = iree_string_view_empty();
    iree_string_view_split(target_options, ' ', &first_token,
                           &unused_remaining);
    first_token = iree_string_view_trim(first_token);
    if (iree_string_view_starts_with(first_token, IREE_SV("@"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low does not accept a function symbol; it runs the shared "
          "source-to-low pass pipeline for the module");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_source_low_options(target_options, out_request));
    out_request->format = LOOM_CHECK_EMIT_SOURCE_LOW_TEXT;
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

static iree_status_t loom_check_emit_compare_output(
    const loom_check_case_t* test_case, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);

  iree_status_t status =
      loom_check_strip_comments(test_case->expected, &stripped_expected);
  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  return status;
}

static iree_status_t loom_check_emit_finish_diagnostics_and_compare_output(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result));
  if (result->raw_outcome != LOOM_CHECK_PASS || !result->has_actual_output) {
    return iree_ok_status();
  }
  return loom_check_emit_compare_output(test_case, allocator, result);
}

static iree_status_t loom_check_emit_symbol_not_found(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_check_case_t* test_case, iree_string_view_t filename,
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

static bool loom_check_diagnostic_collector_has_error(
    const loom_check_diagnostic_collector_t* collector) {
  for (iree_host_size_t i = 0; i < collector->count; ++i) {
    if (collector->diagnostics[i].severity == LOOM_DIAGNOSTIC_ERROR) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_emit_find_func_like(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_check_case_t* test_case, iree_string_view_t filename,
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

static iree_status_t loom_check_emit_write_liveness_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_check_case_t* test_case, iree_string_view_t filename,
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
  loom_liveness_analysis_t analysis = {0};
  IREE_RETURN_IF_ERROR(loom_liveness_analyze_region(
      module, loom_func_like_body(function), analysis_arena, &analysis));
  return loom_liveness_format_json(&analysis, NULL, &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_schedule_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_check_emit_pressure_cliff_spec_t* pressure_cliff_specs,
    iree_host_size_t pressure_cliff_spec_count,
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
  const loom_low_schedule_pressure_cliff_t* pressure_cliffs = NULL;
  if (pressure_cliff_spec_count != 0) {
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, low_function, descriptor_registry,
        loom_target_selection_empty(), emitter, &target));
    if (!target.descriptor_set) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low-schedule-json target did not resolve descriptor set");
    }
    loom_low_schedule_pressure_cliff_t* resolved_cliffs = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        analysis_arena, pressure_cliff_spec_count, sizeof(*resolved_cliffs),
        (void**)&resolved_cliffs));
    bool pressure_cliffs_resolved = true;
    for (iree_host_size_t i = 0; i < pressure_cliff_spec_count; ++i) {
      const loom_check_emit_pressure_cliff_spec_t* spec =
          &pressure_cliff_specs[i];
      uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
      bool found_reg_class = loom_low_descriptor_set_lookup_register_class(
          target.descriptor_set, spec->register_class, &reg_class_id, NULL);
      if (!found_reg_class) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_low_diagnostic_target_key(&target)),
            loom_param_string(loom_low_diagnostic_export_name(&target)),
            loom_param_string(loom_low_diagnostic_config_key(&target)),
            loom_param_string(
                loom_low_diagnostic_function_name(module, low_function)),
            loom_param_string(spec->register_class),
            loom_param_string(target.descriptor_set_key),
        };
        const loom_diagnostic_emission_t emission = {
            .op = low_function,
            .error = LOOM_ERR_BACKEND_039,
            .params = params,
            .param_count = IREE_ARRAYSIZE(params),
        };
        IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
        pressure_cliffs_resolved = false;
        continue;
      }
      resolved_cliffs[i] = (loom_low_schedule_pressure_cliff_t){
          .descriptor_reg_class_id = reg_class_id,
          .cliff_units = spec->cliff_units,
          .tier_before = spec->tier_before,
          .tier_after = spec->tier_after,
      };
    }
    if (!pressure_cliffs_resolved) {
      return iree_ok_status();
    }
    pressure_cliffs = resolved_cliffs;
  }
  loom_low_schedule_options_t options = {
      .descriptor_registry = descriptor_registry,
      .pressure_cliffs =
          {
              .values = pressure_cliffs,
              .count = pressure_cliff_spec_count,
          },
      .emitter = emitter,
      .diagnostic_flags = diagnostic_flags,
      .strategy = strategy,
  };
  loom_low_schedule_table_t table = {0};
  IREE_RETURN_IF_ERROR(loom_low_schedule_function(
      module, low_function, &options, analysis_arena, &table));
  return loom_low_schedule_format_json(&table, &result->actual_output);
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
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count,
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
  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  bool fixed_values_resolved = false;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_resolve_fixed_value_specs(
      module, low_function, fixed_specs, fixed_spec_count, emitter,
      &fixed_values, &fixed_value_count, &fixed_values_resolved,
      analysis_arena));
  if (!fixed_values_resolved) {
    return iree_ok_status();
  }
  loom_low_allocation_options_t options = {
      .descriptor_registry = descriptor_registry,
      .budgets = budgets,
      .budget_count = budget_count,
      .fixed_values = fixed_values,
      .fixed_value_count = fixed_value_count,
      .emitter = emitter,
      .diagnostic_flags = diagnostic_flags,
  };
  IREE_RETURN_IF_ERROR(loom_low_allocate_function(
      module, low_function, &options, analysis_arena, out_table));
  *out_built = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_low_allocation_json(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_emit_build_low_allocation_table(
      module, symbol_name, descriptor_registry, test_case, filename,
      diagnostic_collector, budgets, budget_count, fixed_specs,
      fixed_spec_count, diagnostic_flags, emitter, analysis_arena, &built,
      &table));
  if (!built || table.error_count != 0) {
    return iree_ok_status();
  }
  return loom_low_allocation_format_json(&table, &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_allocation_summary(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  loom_low_allocation_table_t table = {0};
  bool built = false;
  IREE_RETURN_IF_ERROR(loom_check_emit_build_low_allocation_table(
      module, symbol_name, descriptor_registry, test_case, filename,
      diagnostic_collector, budgets, budget_count, fixed_specs,
      fixed_spec_count, diagnostic_flags, emitter, analysis_arena, &built,
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
                                       reg_class->name_string_offset);
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
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_low_schedule_strategy_t strategy,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_check_low_emit_fixed_value_spec_t* fixed_specs,
    iree_host_size_t fixed_spec_count,
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
  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  bool fixed_values_resolved = false;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_resolve_fixed_value_specs(
      module, low_function, fixed_specs, fixed_spec_count, emitter,
      &fixed_values, &fixed_value_count, &fixed_values_resolved,
      analysis_arena));
  if (!fixed_values_resolved) {
    return iree_ok_status();
  }
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = descriptor_registry,
      .schedule_strategy = strategy,
      .allocation_budgets = budgets,
      .allocation_budget_count = budget_count,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .emitter = emitter,
  };
  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      module, low_function, &frame_options, analysis_arena, &frame));
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

static void loom_check_emit_initialize_source_low_print_options(
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_string_view_t descriptor_set_key,
    loom_text_low_asm_environment_t* low_asm_environment,
    loom_text_print_options_t* print_options) {
  loom_low_descriptor_text_asm_environment_initialize(descriptor_registry,
                                                      low_asm_environment);
  *print_options = (loom_text_print_options_t){
      .flags = LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM,
      .low_asm_environment = *low_asm_environment,
      .low_asm_descriptor_set_key = descriptor_set_key,
  };
}

static iree_status_t loom_check_emit_write_source_low_artifacts(
    const loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_string_view_t descriptor_set_key, iree_string_builder_t* output) {
  if (iree_string_view_is_empty(descriptor_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low low output requires a selected descriptor-set key");
  }
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_text_print_options_t print_options = {0};
  loom_check_emit_initialize_source_low_print_options(
      descriptor_registry, descriptor_set_key, &low_asm_environment,
      &print_options);
  bool has_artifact = false;
  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_op_dialect_id(op->kind) != LOOM_DIALECT_LOW) {
      continue;
    }
    if (has_artifact) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    }
    IREE_RETURN_IF_ERROR(loom_text_print_operation_to_builder_with_options(
        module, op, output, &print_options));
    has_artifact = true;
  }
  if (!has_artifact) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-low produced no low artifacts");
  }
  return iree_ok_status();
}

static bool loom_check_emit_is_low_function_op(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op);
}

static iree_status_t loom_check_emit_select_single_low_descriptor_set_key(
    const loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter,
    iree_string_view_t* out_descriptor_set_key, bool* out_found) {
  *out_descriptor_set_key = iree_string_view_empty();
  *out_found = false;

  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!loom_check_emit_is_low_function_op(op)) {
      continue;
    }
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, op, descriptor_registry, loom_target_selection_empty(), emitter,
        &target));
    if (target.descriptor_set == NULL) {
      continue;
    }
    if (iree_string_view_is_empty(*out_descriptor_set_key)) {
      *out_descriptor_set_key = target.descriptor_set_key;
      *out_found = true;
      continue;
    }
    if (!iree_string_view_equal(*out_descriptor_set_key,
                                target.descriptor_set_key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low output=low requires all lowered funcs to use the same "
          "target-low descriptor set");
    }
  }

  return iree_ok_status();
}

static iree_string_view_t loom_check_emit_diagnostic_source_low_pipeline(
    const loom_check_emit_request_t* request) {
  const bool structured_control_flow =
      request->source_low_control_flow_lowering ==
      LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
  if (request->source_low_diagnostic_flags == 0) {
    return structured_control_flow
               ? IREE_SV(
                     "source-to-low{control-flow=structured-low,"
                     "diagnostics=none}")
               : IREE_SV("source-to-low{diagnostics=none}");
  }
  if (request->source_low_diagnostic_flags ==
      LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) {
    return structured_control_flow
               ? IREE_SV(
                     "source-to-low{control-flow=structured-low,"
                     "diagnostics=memory}")
               : IREE_SV("source-to-low{diagnostics=memory}");
  }
  return structured_control_flow
             ? IREE_SV(
                   "source-to-low{control-flow=structured-low,"
                   "diagnostics=all}")
             : IREE_SV("source-to-low{diagnostics=all}");
}

void loom_check_prepare_source_low_options_initialize(
    loom_check_prepare_source_low_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_check_prepare_source_low_options_t){
      .pipeline = IREE_SVL("default"),
      .default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW,
      .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
}

iree_status_t loom_check_prepare_source_low_module(
    loom_module_t* module,
    const loom_check_prepare_source_low_options_t* options,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_check_environment_t* environment,
    loom_source_resolver_t source_resolver,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_arena_block_pool_t* block_pool) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(low_registry);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(diagnostic_collector);
  IREE_ASSERT_ARGUMENT(block_pool);

  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = diagnostic_collector},
      .source_resolver = source_resolver,
      .max_errors = 20,
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(module, &entry_options,
                                                       20, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  if (environment->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-low emit requires a target environment");
  }

  loom_compile_pipeline_options_t compile_options = {0};
  loom_compile_pipeline_options_initialize(&compile_options);
  compile_options.pipeline = options->pipeline;
  compile_options.default_pipeline = options->default_pipeline;
  compile_options.target_pipeline_options.control_flow_lowering =
      options->control_flow_lowering;
  compile_options.target_pipeline_options
      .source_to_low_legality_diagnostic_flags =
      options->source_low_diagnostic_flags;
  compile_options.target_environment = environment->target_environment;
  compile_options.low_descriptor_registry = low_registry;
  compile_options.diagnostic_sink =
      (loom_diagnostic_sink_t){.fn = loom_check_diagnostic_collector_sink,
                               .user_data = diagnostic_collector};
  compile_options.source_resolver = source_resolver;
  compile_options.max_errors = 20;

  loom_pass_run_result_t run_result = {0};
  IREE_RETURN_IF_ERROR(loom_compile_run_pipeline(module, &compile_options,
                                                 block_pool, &run_result));
  if (run_result.error_count != 0 ||
      loom_check_diagnostic_collector_has_error(diagnostic_collector)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(module, &entry_options,
                                                       20, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_target_entry_diagnostic_emitter_t verifier_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_VERIFIER, &verifier_emitter);
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      module, low_registry, &verifier_emitter, loom_target_selection_empty(),
      20, environment->low_verify_provider_list, &low_verify_scratch,
      &low_verify_result));
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_source_low_pipeline_text(
    loom_module_t* source_module, const loom_check_emit_request_t* request,
    const loom_target_environment_t* target_environment,
    iree_arena_block_pool_t* block_pool, loom_check_result_t* result) {
  if (request->has_source_low_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low pipeline outputs cannot be combined with diagnostics");
  }
  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      source_module->context, IREE_SV("__loom_check_source_low_pipeline"),
      block_pool, NULL, source_module->allocator, &pipeline_module);
  loom_op_t* pipeline_op = NULL;
  const loom_target_pipeline_options_t target_pipeline_options = {
      .control_flow_lowering = request->source_low_control_flow_lowering,
  };
  if (iree_status_is_ok(status)) {
    if (request->source_low_output ==
        LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE) {
      status = loom_target_pipeline_build_to_prepared_low(
          pipeline_module, IREE_SV("__loom_check_prepared_low"),
          &target_pipeline_options, target_environment,
          loom_pass_environment_empty(), &pipeline_op);
    } else {
      status = loom_target_pipeline_build_to_source_low(
          pipeline_module, IREE_SV("__loom_check_source_low"),
          &target_pipeline_options, target_environment,
          loom_pass_environment_empty(), &pipeline_op);
    }
  }
  if (iree_status_is_ok(status) && pipeline_op == NULL) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "source-low pipeline builder produced no op");
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_print_module_to_builder(
        pipeline_module, &result->actual_output, LOOM_TEXT_PRINT_DEFAULT);
  }
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  if (iree_status_is_ok(status)) {
    result->has_actual_output = true;
  }
  return status;
}

static iree_status_t loom_check_emit_write_source_low_text(
    loom_module_t* module, const loom_check_emit_request_t* request,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_check_environment_t* environment,
    loom_source_resolver_t source_resolver,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_arena_block_pool_t* block_pool, loom_check_result_t* result) {
  if (environment->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-low emit requires a target environment");
  }
  if (request->source_low_output ==
          LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE ||
      request->source_low_output ==
          LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE) {
    return loom_check_emit_write_source_low_pipeline_text(
        module, request, environment->target_environment, block_pool, result);
  }

  loom_check_prepare_source_low_options_t prepare_options = {0};
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  if (request->has_source_low_diagnostics_option &&
      request->source_low_output != LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    prepare_options.pipeline =
        loom_check_emit_diagnostic_source_low_pipeline(request);
    prepare_options.default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW;
  } else {
    prepare_options.pipeline = IREE_SV("default");
    prepare_options.default_pipeline =
        request->source_low_output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW
            ? (request->has_source_low_diagnostics_option
                   ? LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_DIAGNOSTIC_ARTIFACTS
                   : LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_ARTIFACTS)
            : LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW;
  }
  prepare_options.control_flow_lowering =
      request->source_low_control_flow_lowering;
  prepare_options.source_low_diagnostic_flags =
      request->source_low_diagnostic_flags;
  IREE_RETURN_IF_ERROR(loom_check_prepare_source_low_module(
      module, &prepare_options, low_registry, environment, source_resolver,
      diagnostic_collector, block_pool));
  if (loom_check_diagnostic_collector_has_error(diagnostic_collector)) {
    return iree_ok_status();
  }

  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = diagnostic_collector},
      .source_resolver = source_resolver,
      .max_errors = 20,
  };
  loom_target_entry_diagnostic_emitter_t pass_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_PASS, &pass_emitter);
  iree_string_view_t selected_descriptor_set_key = iree_string_view_empty();
  bool has_low_artifacts = false;
  iree_status_t status = loom_check_emit_select_single_low_descriptor_set_key(
      module, &low_registry->registry, loom_target_entry_emitter(&pass_emitter),
      &selected_descriptor_set_key, &has_low_artifacts);
  IREE_RETURN_IF_ERROR(status);
  if (loom_check_diagnostic_collector_has_error(diagnostic_collector)) {
    return iree_ok_status();
  }
  if (!has_low_artifacts) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("source-to-low")),
    };
    const loom_diagnostic_emission_t emission = {
        .error = LOOM_ERR_TARGET_011,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(loom_target_entry_emitter(&pass_emitter),
                                &emission);
  }

  if (request->source_low_output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    iree_status_t status = loom_check_emit_write_source_low_artifacts(
        module, &low_registry->registry, selected_descriptor_set_key,
        &result->actual_output);
    if (iree_status_is_ok(status)) result->has_actual_output = true;
    return status;
  }
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_text_print_options_t print_options = {0};
  loom_check_emit_initialize_source_low_print_options(
      &low_registry->registry, selected_descriptor_set_key,
      &low_asm_environment, &print_options);
  status = loom_text_print_module_to_builder_with_options(
      module, &result->actual_output, &print_options);
  if (iree_status_is_ok(status)) result->has_actual_output = true;
  return status;
}

static iree_status_t loom_check_emit_verify_provider_module(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_source_resolver_t source_resolver,
    loom_low_verify_provider_list_t low_verify_provider_list,
    loom_check_diagnostic_collector_t* diagnostic_collector) {
  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = diagnostic_collector},
      .source_resolver = source_resolver,
      .max_errors = 20,
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(module, &entry_options,
                                                       20, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_target_entry_diagnostic_emitter_t verifier_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_VERIFIER, &verifier_emitter);
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      module, low_registry, &verifier_emitter, loom_target_selection_empty(),
      20, low_verify_provider_list, &low_verify_scratch, &low_verify_result));
  return iree_ok_status();
}

iree_status_t loom_check_execute_emit(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
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
  if (request.format == LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST) {
    loom_target_low_descriptor_registry_t registry = {0};
    status = loom_check_environment_initialize_low_descriptor_registry(
        environment, &registry);
    if (iree_status_is_ok(status)) {
      status = loom_check_target_low_registry_format_manifest_json(
          &registry, &result->actual_output);
    }
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    result->has_actual_output = true;
    if (test_case->annotation_count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  status = loom_check_strip_comments(test_case->input, &stripped_input);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_module_t* module = NULL;
  loom_target_low_descriptor_registry_t low_registry = {0};
  status = loom_check_environment_initialize_low_descriptor_registry(
      environment, &low_registry);
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_print_context_initialize(
        &low_registry.registry, &diagnostic_collector.type_print_context);
  }
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
      &low_registry.registry, environment->low_asm_diagnostic_provider_list,
      &low_asm_storage, &parse_options.low_asm_environment);
  iree_string_view_t stripped_view = iree_string_builder_view(&stripped_input);
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(stripped_view, filename, context, block_pool,
                             &parse_options, &module);
  }
  diagnostic_collector.module = module;
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (!module || diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    loom_module_free(module);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (provider != NULL) {
    if (provider->execute == NULL) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "emit provider '%.*s' has no execute callback",
                                (int)provider->name.size, provider->name.data);
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    loom_source_entry_t source_entry = {0};
    loom_source_table_resolver_t resolver_data = {0};
    status = loom_check_source_resolver_for_case(
        module, filename, stripped_view, &source_entry, &resolver_data);
    const loom_source_resolver_t source_resolver = {
        .fn = loom_source_table_resolve,
        .user_data = &resolver_data,
    };
    if (iree_status_is_ok(status)) {
      status = loom_check_emit_verify_provider_module(
          module, &low_registry, source_resolver,
          environment->low_verify_provider_list, &diagnostic_collector);
    }
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (diagnostic_collector.count > 0) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    const loom_check_emit_provider_request_t provider_request = {
        .emit_target = test_case->emit_target,
        .target_name = provider_target_name,
        .target_options = provider_target_options,
        .filename = filename,
        .test_case = test_case,
        .environment = environment,
        .module = module,
        .source_resolver = source_resolver,
        .low_registry = &low_registry,
        .diagnostic_collector = &diagnostic_collector,
        .case_arena = &diagnostic_arena,
        .block_pool = block_pool,
        .host_allocator = allocator,
        .result = result,
    };
    iree_host_size_t actual_output_size = result->actual_output.size;
    status = provider->execute(provider, &provider_request);
    if (iree_status_is_ok(status) &&
        result->actual_output.size != actual_output_size) {
      result->has_actual_output = true;
    }
    loom_module_free(module);
    diagnostic_collector.module = NULL;
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (test_case->annotation_count > 0 || diagnostic_collector.count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (request.format == LOOM_CHECK_EMIT_SOURCE_LOW_TEXT) {
    loom_source_entry_t source_entry = {0};
    loom_source_table_resolver_t resolver_data = {0};
    status = loom_check_source_resolver_for_case(
        module, filename, stripped_view, &source_entry, &resolver_data);
    if (iree_status_is_ok(status)) {
      status = loom_check_emit_write_source_low_text(
          module, &request, &low_registry, environment,
          (loom_source_resolver_t){.fn = loom_source_table_resolve,
                                   .user_data = &resolver_data},
          &diagnostic_collector, block_pool, result);
    }
    loom_module_free(module);
    diagnostic_collector.module = NULL;
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (test_case->annotation_count > 0 || diagnostic_collector.count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (request.format == LOOM_CHECK_EMIT_LIVENESS_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY ||
      request.format == LOOM_CHECK_EMIT_LOW_PACKET_JSON) {
    loom_source_entry_t source_entry = {0};
    loom_source_table_resolver_t resolver_data = {0};
    status = loom_check_source_resolver_for_case(
        module, filename, stripped_view, &source_entry, &resolver_data);
    loom_verify_options_t verify_options = {
        .sink = {.fn = loom_check_diagnostic_collector_sink,
                 .user_data = &diagnostic_collector},
        .max_errors = 20,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
    };
    loom_verify_result_t verify_result = {0};
    if (iree_status_is_ok(status)) {
      status = loom_verify_module(module, &verify_options, &verify_result);
    }
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (verify_result.error_count > 0 || diagnostic_collector.count > 0) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON ||
        request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON ||
        request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY ||
        request.format == LOOM_CHECK_EMIT_LOW_PACKET_JSON) {
      loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
          .diagnostic_collector = &diagnostic_collector,
          .module = module,
          .source_resolver = {.fn = loom_source_table_resolve,
                              .user_data = &resolver_data},
          .emitter = LOOM_EMITTER_VERIFIER,
      };
      loom_low_verify_options_t low_verify_options = {
          .descriptor_registry = &low_registry.registry,
          .emitter =
              {
                  .fn = loom_check_diagnostic_emitter_capture_emit,
                  .user_data = &low_diagnostic_capture,
              },
          .provider_list = environment->low_verify_provider_list,
          .max_errors = 20,
      };
      loom_low_verify_result_t low_verify_result = {0};
      loom_low_verify_scratch_t low_verify_scratch =
          loom_low_verify_scratch_for_module(module);
      status = loom_low_verify_module(module, &low_verify_options,
                                      &low_verify_scratch, &low_verify_result);
      if (iree_status_is_ok(status) && (low_verify_result.error_count > 0 ||
                                        diagnostic_collector.count > 0)) {
        status = loom_check_diagnostic_collector_finish(
            &diagnostic_collector, test_case, case_index, report, allocator,
            result);
        loom_module_free(module);
        iree_string_builder_deinitialize(&stripped_input);
        iree_arena_deinitialize(&diagnostic_arena);
        return status;
      }
    }
    loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
        .diagnostic_collector = &diagnostic_collector,
        .module = module,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
        .emitter = LOOM_EMITTER_PASS,
    };
    iree_host_size_t actual_output_size = result->actual_output.size;
    if (iree_status_is_ok(status)) {
      if (request.format == LOOM_CHECK_EMIT_LIVENESS_JSON) {
        status = loom_check_emit_write_liveness_json(
            module, request.analysis_symbol_name, test_case, filename,
            &diagnostic_collector,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, result);
      } else if (request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON) {
        status = loom_check_emit_write_low_schedule_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            test_case, filename, &diagnostic_collector,
            request.low_schedule_pressure_cliff_specs,
            request.low_schedule_pressure_cliff_spec_count,
            request.low_schedule_diagnostic_flags,
            request.low_schedule_strategy,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, result);
      } else if (request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON) {
        status = loom_check_emit_write_low_allocation_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            test_case, filename, &diagnostic_collector,
            request.low_allocation_budgets, request.low_allocation_budget_count,
            request.low_allocation_fixed_value_specs,
            request.low_allocation_fixed_value_spec_count,
            request.low_allocation_diagnostic_flags,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, result);
      } else if (request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_SUMMARY) {
        status = loom_check_emit_write_low_allocation_summary(
            module, request.analysis_symbol_name, &low_registry.registry,
            test_case, filename, &diagnostic_collector,
            request.low_allocation_budgets, request.low_allocation_budget_count,
            request.low_allocation_fixed_value_specs,
            request.low_allocation_fixed_value_spec_count,
            request.low_allocation_diagnostic_flags,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, result);
      } else {
        status = loom_check_emit_write_low_packet_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            test_case, filename, &diagnostic_collector,
            request.low_schedule_strategy, request.low_allocation_budgets,
            request.low_allocation_budget_count,
            request.low_allocation_fixed_value_specs,
            request.low_allocation_fixed_value_spec_count,
            environment->low_packet_diagnostic_provider_list,
            request.low_packet_diagnostic_flags,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, result);
      }
    }
    if (iree_status_is_ok(status)) {
      if (request.suppress_actual_output) {
        result->actual_output.size = actual_output_size;
        if (result->actual_output.buffer &&
            result->actual_output.capacity > actual_output_size) {
          result->actual_output.buffer[actual_output_size] = 0;
        }
      } else if (result->actual_output.size != actual_output_size) {
        result->has_actual_output = true;
      }
    }
    loom_module_free(module);
    diagnostic_collector.module = NULL;
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, request.emit_target_name, result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (test_case->annotation_count > 0 ||
        pass_diagnostic_capture.emission_count > 0 ||
        diagnostic_collector.count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (request.suppress_actual_output) {
      result->raw_outcome = LOOM_CHECK_PASS;
      status = iree_ok_status();
    } else {
      status = loom_check_emit_compare_output(test_case, allocator, result);
    }
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  iree_string_builder_deinitialize(&stripped_input);

  loom_module_free(module);
  diagnostic_collector.module = NULL;
  status = loom_check_emit_finish_status_failure(
      iree_make_status(IREE_STATUS_INTERNAL,
                       "core emit target '%.*s' was parsed but not handled",
                       (int)request.emit_target_name.size,
                       request.emit_target_name.data),
      request.emit_target_name, result);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}
