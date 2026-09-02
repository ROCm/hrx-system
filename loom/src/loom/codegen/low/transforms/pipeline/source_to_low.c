// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_to_low.h"

#include <string.h>

#include "loom/codegen/low/launch_config_program.h"
#include "loom/codegen/low/lower/function_boundary.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/sanitizer/options.h"
#include "loom/target/low_legality.h"
#include "loom/target/reporting/low.h"

typedef struct loom_low_source_to_low_pass_state_t {
  // Control-flow shape expected by source-to-low.
  loom_low_control_flow_lowering_t control_flow_lowering;
  // True when control-flow was explicitly provided.
  bool has_control_flow_option;
  // Maximum number of lowering diagnostics emitted before stopping.
  uint32_t max_errors;
  // True when max_errors was explicitly provided.
  bool has_max_errors_option;
  // Target-low legality diagnostics to emit while matching lowering rules.
  loom_target_low_legality_diagnostic_flags_t legality_diagnostic_flags;
  // True when diagnostics was explicitly provided.
  bool has_diagnostics_option;
  // Sanitizer assertion failure reporting behavior.
  loom_sanitizer_reporting_mode_t sanitizer_reporting_mode;
  // True when sanitizer-reporting was explicitly provided.
  bool has_sanitizer_reporting_option;
} loom_low_source_to_low_pass_state_t;

typedef struct loom_low_source_to_low_parse_context_t {
  // Mutable pass state being populated.
  loom_low_source_to_low_pass_state_t* state;
} loom_low_source_to_low_parse_context_t;

static iree_status_t loom_low_source_to_low_record_target_specialization(
    loom_target_compile_report_t* compile_report,
    const loom_low_source_selection_t* selection) {
  if (!loom_target_compile_report_wants_details(
          compile_report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    return iree_ok_status();
  }
  if (selection->target_source != LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION) {
    return iree_ok_status();
  }
  const loom_target_bundle_t* bundle =
      loom_low_source_selection_target_bundle(selection);
  const loom_target_snapshot_t* snapshot = bundle ? bundle->snapshot : NULL;
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const loom_target_compile_report_source_low_target_row_t row = {
      .function_name = selection->function_name,
      .target_source = selection->target_source,
      .target_symbol_name = selection->target_symbol_name,
      .target_bundle_name = bundle ? bundle->name : iree_string_view_empty(),
      .target_snapshot_name =
          snapshot ? snapshot->name : iree_string_view_empty(),
      .target_config_name = config ? config->name : iree_string_view_empty(),
      .target_subgroup_size = snapshot ? snapshot->subgroup_size : 0,
      .candidate_target_count = selection->candidate_target_count,
      .candidate_target_symbol_name = selection->candidate_target_symbol_name,
      .candidate_target_bundle_name = selection->candidate_target_bundle_name,
      .candidate_target_snapshot_name =
          selection->candidate_target_snapshot_name,
      .candidate_target_config_name = selection->candidate_target_config_name,
      .candidate_target_subgroup_size =
          selection->candidate_target_subgroup_size,
  };
  return loom_target_compile_report_record_source_low_target_row(compile_report,
                                                                 &row);
}

static const loom_pass_option_def_t kLowSourceToLowOptions[] = {
    {IREE_SVL("control-flow"),
     IREE_SVL("Control-flow shape expected at the source-to-low boundary: cfg "
              "or structured-low.")},
    {IREE_SVL("diagnostics"),
     IREE_SVL("Target-low legality diagnostics to emit: none, memory, "
              "operand-forms, or all.")},
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of source-to-low diagnostics to emit; zero "
              "means no limit.")},
    {IREE_SVL("sanitizer-reporting"),
     IREE_SVL("Sanitizer assertion failure reporting mode: default, trap, or "
              "report-only.")},
};

#define LOOM_LOW_SOURCE_TO_LOW_STATISTICS(V, statistics_type)                \
  V(statistics_type, errors, "errors", "Number of lowering errors emitted.") \
  V(statistics_type, functions, "functions",                                 \
    "Number of source funcs lowered.")                                       \
  V(statistics_type, remarks, "remarks",                                     \
    "Number of lowering remarks emitted.")                                   \
  V(statistics_type, declarations, "declarations",                           \
    "Number of source import declarations lowered.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_source_to_low_statistics,
                            loom_low_source_to_low_statistics_t,
                            LOOM_LOW_SOURCE_TO_LOW_STATISTICS)

static const loom_pass_info_t loom_low_source_to_low_pass_info_storage = {
    .name = IREE_SVL("source-to-low"),
    .description = IREE_SVL("Lower targeted source funcs to target-low IR."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowSourceToLowOptions,
    .option_count = IREE_ARRAYSIZE(kLowSourceToLowOptions),
    .statistic_layout = &loom_low_source_to_low_statistics_layout,
};

const loom_pass_info_t* loom_low_source_to_low_pass_info(void) {
  return &loom_low_source_to_low_pass_info_storage;
}

static iree_status_t loom_low_source_to_low_parse_control_flow(
    iree_string_view_t value, loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_control_flow_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'control-flow' for pass 'source-to-low'");
  }
  if (iree_string_view_equal(value, IREE_SV("cfg"))) {
    context->state->control_flow_lowering = LOOM_LOW_CONTROL_FLOW_LOWERING_CFG;
  } else if (iree_string_view_equal(value, IREE_SV("structured-low"))) {
    context->state->control_flow_lowering =
        LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-to-low option 'control-flow' expected 'cfg' or "
        "'structured-low', got '%.*s'",
        (int)value.size, value.data);
  }
  context->state->has_control_flow_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_source_to_low_parse_max_errors(
    uint32_t max_errors, loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_max_errors_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-errors' for pass 'source-to-low'");
  }
  context->state->max_errors = max_errors;
  context->state->has_max_errors_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_source_to_low_parse_diagnostics(
    iree_string_view_t value, loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'diagnostics' for pass 'source-to-low'");
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    context->state->legality_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("memory"))) {
    context->state->legality_diagnostic_flags =
        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS;
  } else if (iree_string_view_equal(value, IREE_SV("operand-forms"))) {
    context->state->legality_diagnostic_flags =
        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    context->state->legality_diagnostic_flags =
        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-to-low option 'diagnostics' expected 'none', 'memory', "
        "'operand-forms', or 'all', got '%.*s'",
        (int)value.size, value.data);
  }
  context->state->has_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_source_to_low_parse_sanitizer_reporting(
    iree_string_view_t value, loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_sanitizer_reporting_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'sanitizer-reporting' for pass 'source-to-low'");
  }
  IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_parse(
      value, IREE_SV("source-to-low option 'sanitizer-reporting'"),
      &context->state->sanitizer_reporting_mode));
  context->state->has_sanitizer_reporting_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_source_to_low_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_source_to_low_parse_context_t* context =
      (loom_low_source_to_low_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("control-flow"))) {
    return loom_low_source_to_low_parse_control_flow(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_low_source_to_low_parse_diagnostics(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("max-errors"))) {
    uint32_t max_errors = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("source-to-low"), name, value, &max_errors));
    return loom_low_source_to_low_parse_max_errors(max_errors, context);
  }
  if (iree_string_view_equal(name, IREE_SV("sanitizer-reporting"))) {
    return loom_low_source_to_low_parse_sanitizer_reporting(value, context);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'source-to-low'",
                          (int)name.size, name.data);
}

iree_status_t loom_low_source_to_low_create(loom_pass_t* pass,
                                            iree_string_view_t options) {
  loom_low_source_to_low_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->control_flow_lowering = LOOM_LOW_CONTROL_FLOW_LOWERING_CFG;
  state->max_errors = 20;
  state->sanitizer_reporting_mode = LOOM_SANITIZER_REPORTING_MODE_DEFAULT;

  loom_low_source_to_low_parse_context_t context = {
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("control-flow"))) {
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_control_flow(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("diagnostics"))) {
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_diagnostics(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("max-errors"))) {
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_max_errors(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("sanitizer-reporting"))) {
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_sanitizer_reporting(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'source-to-low'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_low_source_to_low_parse_option,
                                    .user_data = &context,
                                }));
  }

  pass->state = state;
  return iree_ok_status();
}

iree_status_t loom_low_source_to_low_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  const loom_low_source_to_low_pass_state_t* state =
      (const loom_low_source_to_low_pass_state_t*)pass->state;
  loom_low_source_to_low_statistics_t* statistics =
      loom_low_source_to_low_statistics(pass);
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  loom_target_compile_report_t* compile_report =
      loom_low_pass_capability_compile_report(low_capability);
  loom_kernel_launch_config_program_t* launch_config_program =
      loom_kernel_launch_config_program_from_pass(pass);
  const iree_allocator_t source_low_report_allocator =
      loom_target_compile_report_wants_details(
          compile_report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)
          ? compile_report->allocator
          : iree_allocator_null();
  const bool record_source_low_targets =
      loom_target_compile_report_wants_details(
          compile_report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS);

  iree_arena_allocator_t selection_arena;
  iree_arena_initialize(module->arena.block_pool, &selection_arena);
  loom_low_lower_module_state_t* module_state = NULL;
  loom_low_source_selection_list_t selection_list = {0};
  const loom_low_source_selection_options_t selection_options = {
      .policy_registry = policy_registry,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .lowering_kind = IREE_SV("source-to-low"),
      .function_versions =
          loom_target_pass_capability_function_versions(target_capability),
      .collect_target_candidates = record_source_low_targets,
  };
  iree_status_t status = loom_low_select_source_symbols(
      module, &selection_options, &selection_arena, &selection_list);
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status); ++i) {
    status = loom_low_source_to_low_record_target_specialization(
        compile_report, &selection_list.values[i]);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_low_lower_module_state_create(&selection_arena, &module_state);
  }
  bool emitted_error_diagnostics = false;
  uint32_t declaration_count = 0;
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !emitted_error_diagnostics;
       ++i) {
    const loom_low_source_selection_t* selection = &selection_list.values[i];
    if (selection->kind != LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL) {
      continue;
    }
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection->target_ref,
        .target_facts = selection->target_facts,
        .descriptor_registry = descriptor_registry,
        .policy = selection->policy,
        .emitter = pass->diagnostic_emitter,
        .max_errors = state ? state->max_errors : 20,
        .control_flow_lowering = state ? state->control_flow_lowering
                                       : LOOM_LOW_CONTROL_FLOW_LOWERING_CFG,
        .sanitizer_reporting_mode = state
                                        ? state->sanitizer_reporting_mode
                                        : LOOM_SANITIZER_REPORTING_MODE_DEFAULT,
        .module_state = module_state,
    };
    loom_low_lower_result_t lower_result = {0};
    status = loom_low_lower_import_declaration(module, selection->func,
                                               &lower_options, &lower_result);
    statistics->errors += (int64_t)lower_result.error_count;
    statistics->remarks += (int64_t)lower_result.remark_count;
    if (iree_status_is_ok(status) && lower_result.error_count > 0) {
      emitted_error_diagnostics = true;
      loom_low_lower_result_deinitialize(&lower_result);
      break;
    }
    if (iree_status_is_ok(status)) {
      IREE_ASSERT(lower_result.low_func_op != NULL);
      if (selection->version_handle != NULL) {
        loom_function_version_update(
            selection->version_handle,
            loom_func_like_cast(module, lower_result.low_func_op));
      }
      ++declaration_count;
    }
    loom_low_lower_result_deinitialize(&lower_result);
  }
  uint32_t function_count = 0;
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !emitted_error_diagnostics;
       ++i) {
    const loom_low_source_selection_t* selection = &selection_list.values[i];
    if (selection->kind != LOOM_LOW_SOURCE_SELECTION_FUNCTION) {
      continue;
    }
    const loom_target_low_legality_provider_list_t* legality_provider_list =
        loom_low_pass_capability_legality_provider_list(low_capability);
    loom_value_fact_table_t* fact_table = NULL;
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(selection->func,
                                                       selection->target_facts),
        &fact_table);
    if (!iree_status_is_ok(status)) break;
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection->target_ref,
        .target_facts = selection->target_facts,
        .descriptor_registry = descriptor_registry,
        .legality_provider_list =
            legality_provider_list
                ? *legality_provider_list
                : loom_target_low_legality_provider_list_empty(),
        .legality_diagnostic_flags =
            state ? state->legality_diagnostic_flags : 0,
        .policy = selection->policy,
        .fact_table = fact_table,
        .emitter = pass->diagnostic_emitter,
        .max_errors = state ? state->max_errors : 20,
        .control_flow_lowering = state ? state->control_flow_lowering
                                       : LOOM_LOW_CONTROL_FLOW_LOWERING_CFG,
        .sanitizer_reporting_mode = state
                                        ? state->sanitizer_reporting_mode
                                        : LOOM_SANITIZER_REPORTING_MODE_DEFAULT,
        .module_state = module_state,
        .report_allocator = source_low_report_allocator,
    };
    loom_low_lower_result_t lower_result = {0};
    if (launch_config_program != NULL) {
      status = loom_kernel_launch_config_program_capture(
          launch_config_program, module, selection->func,
          selection->function_name, selection->version_handle,
          selection->target_facts, fact_table);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_function(module, selection->func, &lower_options,
                                       &lower_result);
    }
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status) && compile_report != NULL) {
      status = loom_target_compile_report_record_low_lowering(compile_report,
                                                              &lower_result);
    }
    statistics->errors += (int64_t)lower_result.error_count;
    statistics->remarks += (int64_t)lower_result.remark_count;
    if (iree_status_is_ok(status) && lower_result.error_count > 0) {
      emitted_error_diagnostics = true;
      loom_low_lower_result_deinitialize(&lower_result);
      break;
    }
    if (iree_status_is_ok(status)) {
      IREE_ASSERT(lower_result.low_func_op != NULL);
      if (selection->version_handle != NULL) {
        loom_function_version_update(
            selection->version_handle,
            loom_func_like_cast(module, lower_result.low_func_op));
      }
      ++function_count;
    }
    loom_low_lower_result_deinitialize(&lower_result);
  }
  if (iree_status_is_ok(status) && !emitted_error_diagnostics) {
    status = loom_low_source_selection_finalize_policies(
        module, &selection_list, module_state, &selection_arena);
  }
  iree_arena_deinitialize(&selection_arena);
  IREE_RETURN_IF_ERROR(status);

  statistics->declarations += (int64_t)declaration_count;
  statistics->functions += (int64_t)function_count;
  if (declaration_count + function_count > 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
