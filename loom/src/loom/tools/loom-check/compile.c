// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/compile.h"

#include <inttypes.h>

#include "loom/testing/test_file.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/artifact.h"
#include "loomc/iree.h"

void loom_check_compile_session_deinitialize(
    loom_check_compile_session_t* session) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(session->artifact_pass_programs); ++i) {
    loomc_pass_program_release(session->artifact_pass_programs[i]);
  }
  loomc_compiler_release(session->compiler);
  loomc_workspace_release(session->workspace);
  loomc_context_release(session->context);
  loomc_target_environment_release(session->target_environment);
  *session = (loom_check_compile_session_t){0};
}

iree_status_t loom_check_compile_session_prepare(
    loom_check_compile_session_t* session) {
  if (session->compiler != NULL) {
    return iree_ok_status();
  }
  if (session->target_environment == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loom-check runner has no compiler target environment");
  }

  const iree_allocator_t host_allocator = session->host_allocator;
  const loomc_allocator_t allocator = loomc_allocator_from_iree(host_allocator);
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = session->target_environment,
  };
  const loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  iree_status_t status = iree_status_from_loomc(
      loomc_context_create(&context_options, allocator, &session->context));
  if (iree_status_is_ok(status)) {
    // Keep compiler scratch on the checker's small-case arena block policy.
    const loomc_workspace_options_t workspace_options = {
        .type = LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
        .structure_size = sizeof(workspace_options),
        .block_size = 32 * 1024,
    };
    status = iree_status_from_loomc(loomc_workspace_create(
        &workspace_options, allocator, &session->workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compiler_create(
        session->context, /*options=*/NULL, allocator, &session->compiler));
  }
  if (!iree_status_is_ok(status)) {
    loomc_compiler_release(session->compiler);
    loomc_workspace_release(session->workspace);
    loomc_context_release(session->context);
    session->compiler = NULL;
    session->workspace = NULL;
    session->context = NULL;
  }
  return status;
}

iree_status_t loom_check_compile_session_select_target_profile(
    loom_check_compile_session_t* session, iree_string_view_t specification,
    loomc_target_profile_t** out_target_profile) {
  *out_target_profile = NULL;
  IREE_RETURN_IF_ERROR(loom_check_compile_session_prepare(session));
  return iree_status_from_loomc(loomc_target_profile_select(
      session->target_environment, loomc_string_view_from_iree(specification),
      loomc_allocator_from_iree(session->host_allocator), out_target_profile));
}

iree_status_t loom_check_compile_admit_module(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    loom_check_compile_session_t* session,
    const loom_check_environment_t* environment, uint32_t max_errors,
    loom_check_diagnostic_collector_t* collector,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loomc_module_t** out_module) {
  *out_module = NULL;
  IREE_RETURN_IF_ERROR(loom_check_compile_session_prepare(session));

  iree_string_view_t format = input_request->format;
  iree_string_view_t input_options = iree_string_view_empty();
  if (!iree_string_view_is_empty(test_case->input_options.format)) {
    format = test_case->input_options.format;
    input_options = test_case->input_options.arguments;
  }
  const loom_input_format_t* input_format = NULL;
  IREE_RETURN_IF_ERROR(loom_input_format_select(
      environment->input_formats, format, input_request->path, &input_format,
      /*out_ordinal=*/NULL));

  loomc_source_format_t source_format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  iree_string_view_t source_identifier = input_request->path;
  iree_string_view_t source_contents = test_case->input;
  iree_string_builder_t stripped_source;
  iree_string_builder_initialize(host_allocator, &stripped_source);
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(input_format->name, loom_input_text_format.name)) {
    if (!iree_string_view_is_empty(input_options)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "Loom text input does not accept input options");
    }
    if (iree_status_is_ok(status)) {
      status = loom_test_file_strip_comments(source_contents, &stripped_source);
    }
    source_format = LOOMC_SOURCE_FORMAT_TEXT;
    source_identifier = filename;
    source_contents = iree_string_builder_view(&stripped_source);
  } else if (iree_string_view_equal(input_format->name,
                                    loom_input_bytecode_format.name)) {
    if (!iree_string_view_is_empty(input_options)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "Loom bytecode input does not accept input options");
    }
    source_format = LOOMC_SOURCE_FORMAT_BYTECODE;
    source_identifier = filename;
  }

  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_source_t* source = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = source_format,
        .identifier = loomc_string_view_from_iree(source_identifier),
        .contents = loomc_byte_span_from_iree(iree_make_const_byte_span(
            source_contents.data, source_contents.size)),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    status = iree_status_from_loomc(loomc_source_create(
        &source_options, loomc_allocator_from_iree(host_allocator), &source));
  }
  if (iree_status_is_ok(status)) {
    if (source_format != LOOMC_SOURCE_FORMAT_UNKNOWN) {
      const loomc_module_deserialize_options_t options = {
          .type = LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS,
          .structure_size = sizeof(options),
          .max_errors = max_errors,
      };
      status = iree_status_from_loomc(loomc_module_deserialize_from_source(
          session->context, session->workspace, source, &options,
          loomc_allocator_from_iree(host_allocator), &module, &result));
    } else if (session->provider != NULL && session->provider->import != NULL) {
      status = session->provider->import(
          session->provider->import_user_data, input_format->name,
          input_options, &input_request->source_path_options, session->context,
          session->workspace, source, block_pool, host_allocator, &module,
          &result);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "input format '%.*s' has no LoomC importer linked into this runner",
          (int)input_format->name.size, input_format->name.data);
    }
  }
  bool admitted = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(collector, result, &admitted);
  }
  if (iree_status_is_ok(status) && admitted) {
    *out_module = module;
    module = NULL;
  }
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_source_release(source);
  iree_string_builder_deinitialize(&stripped_source);
  return status;
}

static iree_status_t loom_check_compile_prepare_source_low_program(
    loom_check_compile_session_t* session,
    const loom_check_compile_source_low_options_t* options,
    loom_check_diagnostic_collector_t* collector,
    loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(session->host_allocator);
  const bool has_sanitizer = options->sanitizer.checks != 0 ||
                             options->sanitizer.flags != 0 ||
                             options->sanitizer.reporting_mode !=
                                 LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  const loomc_sanitizer_options_t sanitizer_options = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer_options),
      .checks = (loomc_sanitizer_checks_t)options->sanitizer.checks,
      .flags = (loomc_sanitizer_flags_t)options->sanitizer.flags,
      .reporting_mode =
          (loomc_sanitizer_reporting_mode_t)options->sanitizer.reporting_mode,
  };
  const loomc_target_pipeline_options_t public_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(public_options),
      .next = has_sanitizer ? &sanitizer_options : NULL,
      .identifier = loomc_make_cstring_view("__loom_check_source_low"),
      .kind = options->pipeline_kind,
      .control_flow_lowering = options->control_flow_lowering,
      .source_to_low_max_errors = 20,
      .source_to_low_diagnostic_flags = options->diagnostic_flags,
  };

  loomc_result_t* result = NULL;
  const loomc_status_t operation_status =
      loomc_pass_program_create_from_target_pipeline(session->context,
                                                     &public_options, allocator,
                                                     out_pass_program, &result);

  iree_status_t status = iree_status_from_loomc(operation_status);
  bool prepared = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(collector, result, &prepared);
  }
  loomc_result_release(result);
  if (!prepared) {
    loomc_pass_program_release(*out_pass_program);
    *out_pass_program = NULL;
  }
  return status;
}

iree_status_t loom_check_compile_select_source_function(
    const loomc_module_t* module, iree_string_view_t function_name,
    loom_check_diagnostic_collector_t* collector,
    iree_string_view_t* out_function_name) {
  if (!iree_string_view_is_empty(function_name)) {
    *out_function_name = function_name;
    return iree_ok_status();
  }

  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(iree_arena_allocator(collector->arena));
  loomc_host_size_t definition_count = 0;
  loomc_result_t* count_result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_query_functions(
      module, /*options=*/NULL, allocator, /*function_capacity=*/0,
      /*out_functions=*/NULL, &definition_count, &count_result));
  bool queried = false;
  if (iree_status_is_ok(status)) {
    status =
        loom_check_compile_append_result(collector, count_result, &queried);
  }
  loomc_result_release(count_result);
  if (!iree_status_is_ok(status) || !queried) {
    return status;
  }

  loomc_module_function_t* functions = NULL;
  if (definition_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(collector->arena, definition_count,
                                  sizeof(*functions), (void**)&functions));
    loomc_result_t* query_result = NULL;
    status = iree_status_from_loomc(loomc_module_query_functions(
        module, /*options=*/NULL, allocator, definition_count, functions,
        &definition_count, &query_result));
    queried = false;
    if (iree_status_is_ok(status)) {
      status =
          loom_check_compile_append_result(collector, query_result, &queried);
    }
    loomc_result_release(query_result);
    if (!iree_status_is_ok(status) || !queried) {
      return status;
    }
  }

  iree_host_size_t entry_count = 0;
  for (loomc_host_size_t i = 0; i < definition_count; ++i) {
    const loomc_module_function_t function = functions[i];
    const bool is_entry =
        iree_any_bit_set(function.flags, LOOMC_MODULE_FUNCTION_FLAG_PUBLIC) ||
        loomc_module_function_is_kernel_entry(function);
    entry_count += is_entry;
    if (i == 0 || is_entry) {
      function_name = iree_string_view_from_loomc(function.symbol_name);
    }
  }
  if (definition_count == 0 || (definition_count > 1 && entry_count != 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target compile option requires one function definition or one public "
        "function or kernel entry with private helpers; specify @function for "
        "an ambiguous module (got %" PRIhsz " definitions and %" PRIhsz
        " entries)",
        definition_count, entry_count);
  }
  *out_function_name = function_name;
  return iree_ok_status();
}

iree_status_t loom_check_compile_source_low(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_source_low_options_t* options, bool* out_compiled,
    loomc_result_t** out_result) {
  *out_compiled = false;
  if (out_result != NULL) {
    *out_result = NULL;
  }
  loom_check_compile_session_t* session = request->environment->compile_session;
  loomc_module_t* module = request->public_module;
  iree_status_t status = iree_ok_status();

  loomc_pass_program_t* pass_program = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = loom_check_compile_prepare_source_low_program(
        session, options, request->diagnostic_collector, &pass_program);
  }

  loomc_target_profile_t* target_profile = options->target_profile;
  iree_string_view_t function_name = options->function_name;
  if (iree_status_is_ok(status) && pass_program != NULL &&
      target_profile != NULL && iree_string_view_is_empty(function_name)) {
    status = loom_check_compile_select_source_function(
        module, function_name, request->diagnostic_collector, &function_name);
  }

  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_string_view_from_iree(function_name),
      .target_profile = target_profile,
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = options->report,
      .specializations = target_profile != NULL ? &specialization : NULL,
      .specialization_count = target_profile != NULL ? 1 : 0,
  };
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
  };
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status) && pass_program != NULL) {
    const loomc_allocator_t allocator =
        loomc_allocator_from_iree(request->host_allocator);
    const loomc_status_t operation_status = loomc_compile_module(
        session->compiler, session->workspace, pass_program, module,
        &compile_options, allocator, &result);
    status = iree_status_from_loomc(operation_status);
  }
  if (iree_status_is_ok(status) && result != NULL) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              result, out_compiled);
  }
  if (iree_status_is_ok(status) && out_result != NULL &&
      (*out_compiled || loomc_result_artifact_count(result) != 0)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  loomc_pass_program_release(pass_program);
  return status;
}

static iree_status_t loom_check_compile_artifact_module(
    loom_check_compile_session_t* session,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_artifact_options_t* options,
    loom_check_diagnostic_collector_t* collector, iree_allocator_t allocator,
    loomc_result_t** out_result) {
  *out_result = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_compile_artifact(
      session->compiler, session->workspace, pass_program, module, options,
      loomc_allocator_from_iree(allocator), &result));
  bool compiled = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(collector, result, &compiled);
  }
  if (iree_status_is_ok(status) &&
      (compiled || loomc_result_artifact_count(result) != 0)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t loom_check_compile_get_artifact_pass_program(
    loom_check_compile_session_t* session,
    const loom_check_compile_artifact_options_t* options,
    loom_check_diagnostic_collector_t* collector,
    const loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(session->host_allocator);
  const loomc_target_control_flow_lowering_t control_flow =
      options->control_flow_lowering;
  const iree_host_size_t pass_program_index =
      options->lower_source_to_low ? 1 + (iree_host_size_t)control_flow : 0;
  loomc_pass_program_t** pass_program =
      &session->artifact_pass_programs[pass_program_index];
  if (*pass_program == NULL && !options->lower_source_to_low) {
    IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_pass_program_create_empty(
        session->context, /*options=*/NULL, allocator, pass_program)));
  }
  if (*pass_program == NULL) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .kind = LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
        .control_flow_lowering = control_flow,
    };
    loomc_result_t* operation_result = NULL;
    iree_status_t status =
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            session->context, &pipeline_options, allocator, pass_program,
            &operation_result));
    bool prepared = false;
    if (iree_status_is_ok(status)) {
      status = loom_check_compile_append_result(collector, operation_result,
                                                &prepared);
    }
    loomc_result_release(operation_result);
    if (!prepared) {
      loomc_pass_program_release(*pass_program);
      *pass_program = NULL;
    }
    IREE_RETURN_IF_ERROR(status);
  }
  *out_pass_program = *pass_program;
  return iree_ok_status();
}

iree_status_t loom_check_compile_artifact(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_artifact_options_t* options,
    loomc_result_t** out_result) {
  *out_result = NULL;
  loom_check_compile_session_t* session = request->environment->compile_session;
  loomc_module_t* module = request->public_module;
  iree_status_t status = iree_ok_status();

  const loomc_pass_program_t* pass_program = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = loom_check_compile_get_artifact_pass_program(
        session, options, request->diagnostic_collector, &pass_program);
  }

  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = options->report,
      .artifact_format = loomc_string_view_from_iree(options->artifact_format),
  };
  const loomc_string_view_t root = loomc_string_view_from_iree(options->root);
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .roots = iree_string_view_is_empty(options->root) ? NULL : &root,
      .root_count = iree_string_view_is_empty(options->root) ? 0 : 1,
      .target_profile = options->target_profile,
      .emit_options = &emit_options,
  };
  if (iree_status_is_ok(status) && pass_program != NULL) {
    status = loom_check_compile_artifact_module(
        session, pass_program, module, &compile_options,
        request->diagnostic_collector, request->host_allocator, out_result);
  }

  return status;
}

iree_status_t loom_check_emit_artifact(
    const loom_check_emit_provider_request_t* request,
    const loomc_emit_options_t* options, loomc_result_t** out_result) {
  *out_result = NULL;
  loom_check_compile_session_t* session = request->environment->compile_session;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_emit_module(
      session->target_environment, session->workspace, request->public_module,
      options, loomc_allocator_from_iree(request->host_allocator), &result));
  bool emitted = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              result, &emitted);
  }
  if (iree_status_is_ok(status) &&
      (emitted || loomc_result_artifact_count(result) != 0)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

iree_status_t loom_check_compile_select_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    iree_string_view_t format, const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind == kind &&
        iree_string_view_equal(iree_string_view_from_loomc(artifact->format),
                               format)) {
      *out_artifact = artifact;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "compiler did not produce a '%.*s' artifact",
                          (int)format.size, format.data);
}

iree_status_t loom_check_execute_compile(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_compile_options_t* options,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };

  loomc_module_t* module = NULL;
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, options->session, environment,
      /*max_errors=*/0, &collector, block_pool, allocator, &module);

  loomc_result_t* operation_result = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    const loomc_compile_artifact_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = options->sanitizer,
        .target_profile = options->target_profile,
        .config = options->config,
    };
    status = loom_check_compile_artifact_module(
        options->session, /*pass_program=*/NULL, module, &compile_options,
        &collector, allocator, &operation_result);
  }
  loomc_result_release(operation_result);
  loomc_module_release(module);
  loomc_workspace_trim(options->session->workspace);

  if (iree_status_is_ok(status)) {
    status = loom_check_diagnostic_collector_finish(
        &collector, test_case, case_index, report, allocator, result);
    result->final_outcome = result->raw_outcome;
  }
  iree_arena_deinitialize(&arena);
  return status;
}
