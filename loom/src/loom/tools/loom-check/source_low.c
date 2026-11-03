// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/source_low.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/iree.h"

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
    request->target = value;
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
      request->diagnostic_flags = LOOMC_SOURCE_TO_LOW_DIAGNOSTIC_MEMORY_ACCESS;
    } else if (iree_string_view_equal(value, IREE_SV("operand-forms"))) {
      request->diagnostic_flags = LOOMC_SOURCE_TO_LOW_DIAGNOSTIC_OPERAND_FORM;
    } else if (iree_string_view_equal(value, IREE_SV("all"))) {
      request->diagnostic_flags = LOOMC_SOURCE_TO_LOW_DIAGNOSTIC_ALL;
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
      request->control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
    } else if (iree_string_view_equal(value, IREE_SV("structured-low"))) {
      request->control_flow_lowering =
          LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
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
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
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
  if (!has_target && !iree_string_view_is_empty(request->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source-low @function requires "
                            "target=family:selector");
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

static iree_status_t loom_check_emit_write_source_low_pipeline_text(
    const loom_check_source_low_request_t* request,
    const loom_check_emit_provider_request_t* provider_request) {
  if (iree_any_bit_set(request->options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low pipeline outputs cannot be combined with diagnostics");
  }

  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(provider_request->host_allocator);
  loomc_result_t* verification = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_verify(
      provider_request->public_module, NULL, allocator, &verification));
  bool verified = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(
        provider_request->diagnostic_collector, verification, &verified);
  }
  loomc_result_release(verification);
  if (!iree_status_is_ok(status) || !verified) {
    return status;
  }

  const bool prepared =
      request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE;
  const loomc_sanitizer_options_t sanitizer = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer),
      .checks = (loomc_sanitizer_checks_t)request->sanitizer.checks,
      .flags = (loomc_sanitizer_flags_t)request->sanitizer.flags,
      .reporting_mode =
          (loomc_sanitizer_reporting_mode_t)request->sanitizer.reporting_mode,
  };
  const loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .next = &sanitizer,
      .identifier = prepared
                        ? loomc_make_cstring_view("__loom_check_prepared_low")
                        : loomc_make_cstring_view("__loom_check_source_low"),
      .kind = prepared ? LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW
                       : LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      .control_flow_lowering = request->control_flow_lowering,
  };
  loomc_pass_program_t* pass_program = NULL;
  loomc_result_t* result = NULL;
  status =
      iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
          provider_request->environment->compile_session->context,
          &pipeline_options, allocator, &pass_program, &result));
  bool succeeded = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(
        provider_request->diagnostic_collector, result, &succeeded);
  }
  loomc_result_release(result);
  loomc_source_t* source = NULL;
  if (iree_status_is_ok(status) && succeeded) {
    status = iree_status_from_loomc(loomc_pass_program_print_text_to_source(
        pass_program, NULL, allocator, &source));
  }
  if (iree_status_is_ok(status) && succeeded) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status = iree_string_builder_append_string(
        &provider_request->result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
    provider_request->result->has_actual_output = true;
  }
  loomc_source_release(source);
  loomc_pass_program_release(pass_program);
  return status;
}

static iree_status_t loom_check_source_low_query_has_function_kind(
    const loom_check_emit_provider_request_t* provider_request,
    loomc_module_function_kind_t kind, bool* out_has_function) {
  *out_has_function = false;
  const loomc_module_function_query_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      .structure_size = sizeof(options),
      .kind = kind,
  };
  loomc_host_size_t function_count = 0;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_query_functions(
      provider_request->public_module, &options,
      loomc_allocator_from_iree(
          iree_arena_allocator(provider_request->diagnostic_collector->arena)),
      /*function_capacity=*/0, /*out_functions=*/NULL, &function_count,
      &result));
  bool queried = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(
        provider_request->diagnostic_collector, result, &queried);
  }
  if (iree_status_is_ok(status) && queried) {
    *out_has_function = function_count != 0;
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t loom_check_source_low_query_has_function(
    const loom_check_emit_provider_request_t* provider_request,
    bool* out_has_function) {
  IREE_RETURN_IF_ERROR(loom_check_source_low_query_has_function_kind(
      provider_request, LOOMC_MODULE_FUNCTION_KIND_TARGET_FUNCTION,
      out_has_function));
  if (*out_has_function) {
    return iree_ok_status();
  }
  return loom_check_source_low_query_has_function_kind(
      provider_request, LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL,
      out_has_function);
}

static iree_status_t loom_check_emit_write_source_low_module(
    const loom_check_emit_provider_request_t* provider_request,
    const loom_check_source_low_request_t* request) {
  bool has_function = false;
  IREE_RETURN_IF_ERROR(loom_check_source_low_query_has_function(
      provider_request, &has_function));
  if (!has_function) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("source-to-low")),
    };
    return loom_check_diagnostic_collector_emit_case_source(
        provider_request->diagnostic_collector, provider_request->test_case,
        provider_request->filename, LOOM_EMITTER_PASS, LOOM_ERR_TARGET_011,
        params, IREE_ARRAYSIZE(params));
  }

  if (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_NONE) {
    provider_request->result->has_actual_output = false;
    return iree_ok_status();
  }

  const loomc_module_serialize_options_t print_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(print_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM,
      .top_level_dialect =
          request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW
              ? loomc_make_cstring_view("low")
              : loomc_string_view_empty(),
  };
  loomc_source_t* source = NULL;
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(provider_request->host_allocator);
  const loomc_status_t print_status =
      request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW
          ? loomc_module_print_text_to_source(provider_request->public_module,
                                              &print_options, allocator,
                                              &source)
          : loomc_module_serialize_text_to_source(
                provider_request->public_module, &print_options, allocator,
                &source);
  iree_status_t status = iree_status_from_loomc(print_status);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status = iree_string_builder_append_string(
        &provider_request->result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
  }
  if (iree_status_is_ok(status)) {
    provider_request->result->has_actual_output = true;
  }
  loomc_source_release(source);
  return status;
}

static bool loom_check_source_low_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("source-low")) ||
         iree_string_view_equal(target_name, IREE_SV("source-to-low"));
}

iree_status_t loom_check_source_low_execute(
    const loom_check_emit_provider_request_t* provider_request,
    const loom_check_source_low_request_t* source_request,
    loomc_target_profile_t* target_profile) {
  const loom_check_source_low_request_t request = *source_request;
  if (request.output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE ||
      request.output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE) {
    return loom_check_emit_write_source_low_pipeline_text(&request,
                                                          provider_request);
  }

  // Diagnostics observe the same source normalization and lowering regardless
  // of output scope. Only assembly printing needs extra artifact preparation.
  loomc_target_pipeline_kind_t pipeline_kind =
      LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW;
  if (request.output != LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_NONE) {
    // Both IR output scopes require assembly-ready Low, without materializing
    // the target ABI or otherwise advancing to prepared-Low.
    pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW_ARTIFACTS;
  }
  const loom_check_compile_source_low_options_t compile_options = {
      .pipeline_kind = pipeline_kind,
      .diagnostic_flags = request.diagnostic_flags,
      .control_flow_lowering = request.control_flow_lowering,
      .sanitizer = request.sanitizer,
      .function_name = request.function_name,
      .target_profile = target_profile,
  };
  bool compiled = false;
  IREE_RETURN_IF_ERROR(loom_check_compile_source_low(
      provider_request, &compile_options, /*consumer=*/NULL,
      /*user_data=*/NULL, &compiled));
  if (!compiled) {
    return iree_ok_status();
  }
  return loom_check_emit_write_source_low_module(provider_request, &request);
}

static iree_status_t loom_check_source_low_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* provider_request) {
  (void)provider;
  loom_check_source_low_request_t request;
  IREE_RETURN_IF_ERROR(
      loom_check_source_low_parse(provider_request->target_options, &request));
  loomc_target_profile_t* target_profile = NULL;
  if (!iree_string_view_is_empty(request.target)) {
    IREE_RETURN_IF_ERROR(loom_check_compile_session_select_target_profile(
        provider_request->environment->compile_session, request.target,
        &target_profile));
  }
  iree_status_t status =
      loom_check_source_low_execute(provider_request, &request, target_profile);
  loomc_target_profile_release(target_profile);
  return status;
}

static iree_status_t loom_check_source_low_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "source-low, source-to-low");
}

const loom_check_emit_provider_t loom_check_source_low_emit_provider = {
    .name = IREE_SVL("source-low"),
    .match = loom_check_source_low_emit_provider_matches,
    .execute = loom_check_source_low_emit_provider_execute,
    .append_names = loom_check_source_low_emit_provider_append_names,
};
