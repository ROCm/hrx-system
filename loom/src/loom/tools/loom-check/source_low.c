// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/source_low.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/error/source.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/entry_selection.h"
#include "loom/target/pipeline.h"
#include "loom/target/provider.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/interop.h"
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

static bool loom_check_emit_has_low_function(const loom_module_t* module) {
  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op)) {
      return true;
    }
  }
  return false;
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

  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      provider_request->context, IREE_SV("__loom_check_source_low_pipeline"),
      provider_request->block_pool, NULL, provider_request->host_allocator,
      &pipeline_module);
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
          &target_pipeline_options,
          provider_request->environment->target_environment,
          loom_pass_environment_empty(), &pipeline_op);
    } else {
      status = loom_target_pipeline_build_to_source_low(
          pipeline_module, IREE_SV("__loom_check_source_low"),
          &target_pipeline_options,
          provider_request->environment->target_environment,
          loom_pass_environment_empty(), &pipeline_op);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_print_module_to_builder(
        pipeline_module, &provider_request->result->actual_output,
        LOOM_TEXT_PRINT_DEFAULT);
  }
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  if (iree_status_is_ok(status)) {
    provider_request->result->has_actual_output = true;
  }
  return status;
}

typedef struct loom_check_source_low_consumer_t {
  // Provider invocation receiving diagnostics and comparable output.
  const loom_check_emit_provider_request_t* provider_request;
  // Parsed source-low presentation request.
  const loom_check_source_low_request_t* source_request;
} loom_check_source_low_consumer_t;

static iree_status_t loom_check_emit_write_source_low_module(
    void* user_data, const loom_check_compile_source_low_view_t* view) {
  const loom_check_source_low_consumer_t* consumer =
      (const loom_check_source_low_consumer_t*)user_data;
  const loom_check_emit_provider_request_t* provider_request =
      consumer->provider_request;
  const loom_check_source_low_request_t* request = consumer->source_request;
  const loom_module_t* module = view->module;
  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = provider_request->diagnostic_collector},
      .source_resolver = view->source_resolver,
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
    provider_request->result->has_actual_output = false;
    return iree_ok_status();
  }

  if (request->output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    return loom_check_emit_write_source_low_artifacts(
        module, &provider_request->low_registry->registry,
        &provider_request->result->actual_output);
  }

  const loomc_module_serialize_options_t serialize_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(serialize_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM,
  };
  loomc_source_t* source = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_serialize_text_to_source(
          view->public_module, &serialize_options,
          loomc_allocator_from_iree(provider_request->host_allocator),
          &source));
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status = iree_string_builder_append_string(
        &provider_request->result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
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

static iree_status_t loom_check_source_low_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* provider_request) {
  (void)provider;
  loom_check_source_low_request_t request;
  IREE_RETURN_IF_ERROR(
      loom_check_source_low_parse(provider_request->target_options, &request));
  if (request.output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PIPELINE ||
      request.output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_PREPARED_PIPELINE) {
    return loom_check_emit_write_source_low_pipeline_text(&request,
                                                          provider_request);
  }

  if (loom_sanitizer_options_is_enabled(&request.sanitizer) &&
      iree_any_bit_set(request.options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-low option 'sanitizer' cannot be combined with "
        "'diagnostics' for module or low output");
  }

  loom_check_compile_source_low_pipeline_t pipeline =
      LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DEFAULT;
  if (iree_any_bit_set(request.options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS) &&
      request.output != LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    pipeline = LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC;
  } else if (request.output == LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW) {
    pipeline = iree_any_bit_set(request.options,
                                LOOM_CHECK_SOURCE_LOW_OPTION_DIAGNOSTICS)
                   ? LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC_ARTIFACT
                   : LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_ARTIFACT;
  }
  const loom_check_compile_source_low_options_t compile_options = {
      .pipeline = pipeline,
      .diagnostic_flags = request.diagnostic_flags,
      .control_flow_lowering = request.control_flow_lowering,
      .sanitizer = request.sanitizer,
      .function_name = request.function_name,
      .target = request.target,
  };
  const loom_check_source_low_consumer_t consumer = {
      .provider_request = provider_request,
      .source_request = &request,
  };
  return loom_check_compile_source_low(provider_request, &compile_options,
                                       loom_check_emit_write_source_low_module,
                                       (void*)&consumer);
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
