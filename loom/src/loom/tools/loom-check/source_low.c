// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/source_low.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/entry_selection.h"
#include "loom/tools/loom-check/diagnostics.h"

static iree_status_t loom_check_emit_parse_source_low_option(
    iree_string_view_t token, loom_check_source_low_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("target"))) {
    if (iree_any_bit_set(request->options,
                         LOOM_CHECK_SOURCE_LOW_OPTION_TARGET)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'target'");
    }
    IREE_RETURN_IF_ERROR(
        loom_target_specification_parse(value, &request->target));
    request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_TARGET;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    if (iree_any_bit_set(request->options,
                         LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'diagnostics'");
    }
    if (iree_string_view_equal(value, IREE_SV("none"))) {
      request->diagnostic_flags = 0;
    } else if (iree_string_view_equal(value, IREE_SV("memory"))) {
      request->diagnostic_flags =
          LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS;
    } else if (iree_string_view_equal(value, IREE_SV("operand-forms"))) {
      request->diagnostic_flags =
          LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM;
    } else if (iree_string_view_equal(value, IREE_SV("all"))) {
      request->diagnostic_flags = LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low option 'diagnostics' expected 'none', 'memory', "
          "'operand-forms', or 'all', got '%.*s'",
          (int)value.size, value.data);
    }
    request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("control-flow"))) {
    if (iree_any_bit_set(request->options,
                         LOOM_CHECK_SOURCE_LOW_OPTION_CONTROL_FLOW)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'control-flow'");
    }
    if (iree_string_view_equal(value, IREE_SV("cfg"))) {
      request->control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
    } else if (iree_string_view_equal(value, IREE_SV("structured-low"))) {
      request->control_flow_lowering =
          LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source-low option 'control-flow' expected 'cfg' or "
          "'structured-low', got '%.*s'",
          (int)value.size, value.data);
    }
    request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_CONTROL_FLOW;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("sanitizer"))) {
    if (iree_any_bit_set(request->options,
                         LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate source-low option 'sanitizer'");
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
        value, IREE_SV("source-low option 'sanitizer'"),
        &request->sanitizer.checks));
    request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("sanitizer-reporting"))) {
    if (iree_any_bit_set(request->options,
                         LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER_REPORTING)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate source-low option 'sanitizer-reporting'");
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_parse(
        value, IREE_SV("source-low option 'sanitizer-reporting'"),
        &request->sanitizer.reporting_mode));
    request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_SANITIZER_REPORTING;
    return iree_ok_status();
  }
  if (!iree_string_view_equal(name, IREE_SV("output"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown source-low option '%.*s'", (int)name.size,
                            name.data);
  }
  if (iree_any_bit_set(request->options, LOOM_CHECK_SOURCE_LOW_OPTION_OUTPUT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate source-low option 'output'");
  }
  if (iree_string_view_equal(value, IREE_SV("module"))) {
    request->output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE;
  } else if (iree_string_view_equal(value, IREE_SV("low"))) {
    request->output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW;
  } else if (iree_string_view_equal(value, IREE_SV("pipeline"))) {
    request->output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE;
  } else if (iree_string_view_equal(value, IREE_SV("prepared-pipeline"))) {
    request->output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE;
  } else if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_NONE;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low option 'output' expected 'module', 'low', 'pipeline', or "
        "'prepared-pipeline', or 'none', got '%.*s'",
        (int)value.size, value.data);
  }
  request->options |= LOOM_CHECK_SOURCE_LOW_OPTION_OUTPUT;
  return iree_ok_status();
}

iree_status_t loom_check_source_low_parse(
    iree_string_view_t text, loom_check_source_low_request_t* request) {
  *request = (loom_check_source_low_request_t){
      .output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_MODULE,
      .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
  text = iree_string_view_trim(text);
  if (iree_string_view_starts_with_char(text, '@')) {
    iree_string_view_t symbol;
    iree_string_view_split(text, ' ', &symbol, &text);
    request->function_name =
        iree_string_view_substr(symbol, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(request->function_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source-low requires a nonempty function symbol");
    }
  }
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
  const bool has_target =
      iree_any_bit_set(request->options, LOOM_CHECK_SOURCE_LOW_OPTION_TARGET);
  if (has_target != !iree_string_view_is_empty(request->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source-low specialization requires both @function "
                            "and target=family:selector");
  }
  if (has_target &&
      (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE ||
       request->output ==
           LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low specialization requires module, low, or none output");
  }
  return iree_ok_status();
}

static void loom_check_emit_initialize_source_low_print_options(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* low_asm_environment,
    loom_text_print_options_t* print_options) {
  loom_low_descriptor_text_asm_environment_initialize(descriptor_registry,
                                                      low_asm_environment);
  *print_options = (loom_text_print_options_t){
      .flags = LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM,
      .low_asm_environment = *low_asm_environment,
  };
}

static iree_status_t loom_check_emit_write_source_low_artifacts(
    const loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_string_builder_t* output) {
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_text_print_options_t print_options = {0};
  loom_check_emit_initialize_source_low_print_options(
      descriptor_registry, &low_asm_environment, &print_options);
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

static bool loom_check_emit_has_low_function(const loom_module_t* module) {
  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_check_emit_is_low_function_op(op)) {
      return true;
    }
  }
  return false;
}

static iree_string_view_t loom_check_emit_diagnostic_source_low_pipeline(
    const loom_check_source_low_request_t* request) {
  const bool structured_control_flow =
      request->control_flow_lowering ==
      LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
  if (request->diagnostic_flags == 0) {
    return structured_control_flow
               ? IREE_SV(
                     "source-to-low{control-flow=structured-low,"
                     "diagnostics=none}")
               : IREE_SV("source-to-low{diagnostics=none}");
  }
  if (request->diagnostic_flags ==
      LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) {
    return structured_control_flow
               ? IREE_SV(
                     "source-to-low{control-flow=structured-low,"
                     "diagnostics=memory}")
               : IREE_SV("source-to-low{diagnostics=memory}");
  }
  if (request->diagnostic_flags ==
      LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM) {
    return structured_control_flow
               ? IREE_SV(
                     "source-to-low{control-flow=structured-low,"
                     "diagnostics=operand-forms}")
               : IREE_SV("source-to-low{diagnostics=operand-forms}");
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

  loom_compile_pipeline_result_t pipeline_result = {0};
  const loom_target_entry_options_t entry_options = {
      .function_versions = &pipeline_result.function_versions.list,
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = diagnostic_collector},
      .source_resolver = source_resolver,
      .max_errors = 20,
  };
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
  compile_options.target_pipeline_options.sanitizer = options->sanitizer;
  compile_options.target_environment = environment->target_environment;
  compile_options.target_specializations = options->target_specializations;
  compile_options.low_descriptor_registry = low_registry;
  compile_options.diagnostic_sink =
      (loom_diagnostic_sink_t){.fn = loom_check_diagnostic_collector_sink,
                               .user_data = diagnostic_collector};
  compile_options.source_resolver = source_resolver;
  compile_options.max_errors = 20;
  compile_options.report = options->report;

  iree_status_t status = loom_compile_run_pipeline(
      module, &compile_options, block_pool, &pipeline_result);
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0 &&
      !loom_check_diagnostic_collector_has_error(diagnostic_collector)) {
    loom_verify_result_t verify_result = {0};
    status = loom_target_entry_verify_module(module, &entry_options, 20,
                                             &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count == 0) {
      loom_target_entry_diagnostic_emitter_t verifier_emitter = {0};
      loom_target_entry_diagnostic_emitter_initialize(
          module, &entry_options, LOOM_EMITTER_VERIFIER, &verifier_emitter);
      loom_low_verify_result_t low_verify_result = {0};
      loom_low_verify_scratch_t low_verify_scratch =
          loom_low_verify_scratch_for_module(module);
      status = loom_target_entry_verify_low_module(
          module, low_registry, &entry_options, &verifier_emitter, 20,
          environment->low_verify_provider_list, &low_verify_scratch,
          &low_verify_result);
    }
  }
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  return status;
}

static iree_status_t loom_check_emit_write_source_low_pipeline_text(
    loom_module_t* source_module,
    const loom_check_source_low_request_t* request,
    const loom_target_environment_t* target_environment,
    iree_arena_block_pool_t* block_pool, loom_check_result_t* result) {
  if (iree_any_bit_set(request->options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)) {
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
      .control_flow_lowering = request->control_flow_lowering,
      .sanitizer = request->sanitizer,
  };
  if (iree_status_is_ok(status)) {
    if (request->output ==
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

iree_status_t loom_check_source_low_emit(
    loom_module_t* module, const loom_check_source_low_request_t* request,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_check_environment_t* environment,
    loom_source_resolver_t source_resolver,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_arena_block_pool_t* block_pool, loom_check_result_t* result) {
  if (environment->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-low emit requires a target environment");
  }
  if (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE ||
      request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE) {
    return loom_check_emit_write_source_low_pipeline_text(
        module, request, environment->target_environment, block_pool, result);
  }

  const bool sanitizer_enabled =
      loom_sanitizer_options_is_enabled(&request->sanitizer);
  if (sanitizer_enabled &&
      iree_any_bit_set(request->options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low option 'sanitizer' cannot be combined with "
        "'diagnostics' for module or low output");
  }

  loom_check_prepare_source_low_options_t prepare_options = {0};
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  if (iree_any_bit_set(request->options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS) &&
      request->output != LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    prepare_options.pipeline =
        loom_check_emit_diagnostic_source_low_pipeline(request);
    prepare_options.default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW;
  } else {
    prepare_options.pipeline = IREE_SV("default");
    prepare_options.default_pipeline =
        request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW
            ? (iree_any_bit_set(request->options,
                                LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)
                   ? LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_DIAGNOSTIC_ARTIFACTS
                   : LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_ARTIFACTS)
            : LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW;
  }
  prepare_options.control_flow_lowering = request->control_flow_lowering;
  prepare_options.source_low_diagnostic_flags = request->diagnostic_flags;
  prepare_options.sanitizer = request->sanitizer;
  loom_target_specialization_request_t specialization = {0};
  if (iree_any_bit_set(request->options, LOOM_CHECK_SOURCE_LOW_OPTION_TARGET)) {
    specialization.function_name = request->function_name;
    IREE_RETURN_IF_ERROR(loom_target_environment_select_profile(
        environment->target_environment, &request->target,
        &specialization.target_profile));
    prepare_options.target_specializations =
        (loom_target_specialization_request_list_t){&specialization, 1};
  }
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
  if (!loom_check_emit_has_low_function(module)) {
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

  if (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_NONE) {
    return iree_ok_status();
  }

  if (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    iree_status_t status = loom_check_emit_write_source_low_artifacts(
        module, &low_registry->registry, &result->actual_output);
    if (iree_status_is_ok(status)) {
      result->has_actual_output = true;
    }
    return status;
  }
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_text_print_options_t print_options = {0};
  loom_check_emit_initialize_source_low_print_options(
      &low_registry->registry, &low_asm_environment, &print_options);
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &result->actual_output, &print_options);
  if (iree_status_is_ok(status)) {
    result->has_actual_output = true;
  }
  return status;
}
