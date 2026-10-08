// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/compile.h"

#include <inttypes.h>

#include "loom/ops/op_defs.h"
#include "loom/target/pipeline.h"
#include "loom/testing/test_file.h"
#include "loom/tooling/io/source_path.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/artifact.h"
#include "loomc/interop.h"
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

static iree_status_t loom_check_compile_session_prepare(
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
    status = iree_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, allocator, &session->workspace));
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

static iree_status_t loom_check_compile_append_result(
    loom_check_diagnostic_collector_t* collector, const loomc_result_t* result,
    bool* out_succeeded) {
  *out_succeeded = loomc_result_succeeded(result);
  return loom_check_compile_append_result_diagnostics(collector, result);
}

iree_status_t loom_check_compile_admit_module(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    loom_check_compile_session_t* session,
    const loom_check_environment_t* environment,
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
  const loom_input_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(loom_input_provider_select(
      environment->input_providers, format, input_request->path, &provider));

  loomc_source_format_t source_format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  iree_string_view_t source_identifier = input_request->path;
  iree_string_view_t source_contents = test_case->input;
  iree_string_builder_t stripped_source;
  iree_string_builder_initialize(host_allocator, &stripped_source);
  iree_status_t status = iree_ok_status();
  if (provider == &loom_input_text_provider) {
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
  } else if (provider == &loom_input_bytecode_provider) {
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
  char* source_identifier_storage = NULL;
  if (iree_status_is_ok(status) &&
      (provider == &loom_input_text_provider ||
       provider == &loom_input_bytecode_provider)) {
    status = loom_tooling_source_path_remap(
        source_identifier, &input_request->source_path_options, host_allocator,
        &source_identifier, &source_identifier_storage);
  }
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
    if (provider == &loom_input_text_provider ||
        provider == &loom_input_bytecode_provider) {
      status = iree_status_from_loomc(loomc_module_deserialize_from_source(
          session->context, session->workspace, source, /*options=*/NULL,
          loomc_allocator_from_iree(host_allocator), &module, &result));
    } else if (session->provider != NULL && session->provider->import != NULL) {
      status = session->provider->import(
          session->provider->import_user_data, provider->name, input_options,
          &input_request->source_path_options, session->context,
          session->workspace, source, block_pool, host_allocator, &module,
          &result);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "input format '%.*s' has no LoomC importer linked into this runner",
          (int)provider->name.size, provider->name.data);
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
  iree_allocator_free(host_allocator, source_identifier_storage);
  iree_string_builder_deinitialize(&stripped_source);
  return status;
}

static iree_status_t loom_check_compile_source_low_diagnostic_pipeline(
    const loom_check_compile_source_low_options_t* options,
    iree_string_view_t* out_pipeline) {
  bool structured = false;
  switch (options->control_flow_lowering) {
    case LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG:
      break;
    case LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW:
      structured = true;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source-to-Low control-flow mode %d",
                              (int)options->control_flow_lowering);
  }
  switch (options->diagnostic_flags) {
    case 0:
    case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS:
    case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM:
    case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source-to-Low diagnostic flags 0x%08X",
                              options->diagnostic_flags);
  }
  static const iree_string_view_t kPipelines[2][4] = {
      {
          IREE_SVL("source-to-low{diagnostics=none}"),
          IREE_SVL("source-to-low{diagnostics=memory}"),
          IREE_SVL("source-to-low{diagnostics=operand-forms}"),
          IREE_SVL("source-to-low{diagnostics=all}"),
      },
      {
          IREE_SVL(
              "source-to-low{control-flow=structured-low,diagnostics=none}"),
          IREE_SVL(
              "source-to-low{control-flow=structured-low,diagnostics=memory}"),
          IREE_SVL("source-to-low{control-flow=structured-low,"
                   "diagnostics=operand-forms}"),
          IREE_SVL(
              "source-to-low{control-flow=structured-low,diagnostics=all}"),
      },
  };
  *out_pipeline = kPipelines[structured][options->diagnostic_flags];
  return iree_ok_status();
}

static iree_status_t loom_check_compile_prepare_source_low_program(
    loom_check_compile_session_t* session,
    const loom_check_compile_source_low_options_t* options,
    loom_check_diagnostic_collector_t* collector,
    loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(session->host_allocator);
  const loom_target_pipeline_options_t native_options = {
      .control_flow_lowering = options->control_flow_lowering,
      .source_to_low_legality_diagnostic_flags = options->diagnostic_flags,
      .source_to_low_max_errors = 20,
      .sanitizer = options->sanitizer,
  };
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
      .kind = LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      .control_flow_lowering =
          (loomc_target_control_flow_lowering_t)options->control_flow_lowering,
      .source_to_low_max_errors = 20,
  };

  loomc_result_t* result = NULL;
  loomc_status_t operation_status = loomc_ok_status();
  switch (options->pipeline) {
    case LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DEFAULT:
      operation_status = loomc_pass_program_create_from_target_pipeline(
          session->context, &public_options, allocator, out_pass_program,
          &result);
      break;
    case LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_ARTIFACT:
      operation_status =
          loomc_pass_program_create_from_native_source_low_pipeline(
              session->context,
              loom_target_pipeline_build_to_source_low_artifacts,
              public_options.identifier, &native_options, allocator,
              out_pass_program, &result);
      break;
    case LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC_ARTIFACT:
      operation_status =
          loomc_pass_program_create_from_native_source_low_pipeline(
              session->context,
              loom_target_pipeline_build_to_source_low_diagnostic_artifacts,
              public_options.identifier, &native_options, allocator,
              out_pass_program, &result);
      break;
    case LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC: {
      iree_string_view_t pipeline = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_check_compile_source_low_diagnostic_pipeline(
          options, &pipeline));
      operation_status = loomc_pass_program_create_from_pipeline_text(
          session->context, loomc_string_view_from_iree(pipeline), NULL,
          allocator, out_pass_program, &result);
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source-Low pipeline kind %d",
                              (int)options->pipeline);
  }

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

static iree_status_t loom_check_compile_select_source_function(
    const loom_module_t* module, iree_string_view_t function_name,
    iree_string_view_t* out_function_name) {
  if (!iree_string_view_is_empty(function_name)) {
    *out_function_name = function_name;
    return iree_ok_status();
  }

  iree_host_size_t definition_count = 0;
  iree_host_size_t public_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_func_like_t function =
        loom_func_like_const_cast(module, symbol->defining_op);
    if (loom_func_like_body(function) == NULL) {
      continue;
    }
    ++definition_count;
    const bool is_public =
        iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC);
    public_count += is_public;
    if (definition_count == 1 || is_public) {
      function_name = loom_string_table_get(&module->strings, symbol->name_id);
    }
  }
  if (definition_count == 0 || (definition_count > 1 && public_count != 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target compile option requires one function definition or one public "
        "entry with private helpers; specify @function for an ambiguous "
        "module (got %" PRIhsz " definitions and %" PRIhsz " public entries)",
        definition_count, public_count);
  }
  *out_function_name = function_name;
  return iree_ok_status();
}

static iree_status_t loom_check_compile_project_module(
    loomc_module_t* module, loom_check_diagnostic_collector_t* collector,
    iree_allocator_t allocator, loomc_module_interop_view_t* out_view,
    bool* out_succeeded) {
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_get_interop_view(
      module, loomc_allocator_from_iree(allocator), out_view, &result));
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(collector, result, out_succeeded);
  }
  loomc_result_release(result);
  return status;
}

iree_status_t loom_check_compile_with_native_module(
    const loom_check_emit_provider_request_t* request,
    loom_check_compile_native_module_consumer_fn_t consumer, void* user_data) {
  loomc_module_interop_view_t verified_view = {0};
  bool projected = false;
  iree_status_t status = loom_check_compile_project_module(
      request->public_module, request->diagnostic_collector,
      request->host_allocator, &verified_view, &projected);
  if (!iree_status_is_ok(status) || !projected) {
    return status;
  }

  const loomc_module_mutable_interop_view_t mutable_view =
      loomc_module_get_mutable_interop_view(request->public_module);
  const loom_check_emit_native_module_t native_module = {
      .module = mutable_view.module,
      .source_resolver =
          {
              .fn = loom_source_table_resolve,
              .user_data = (void*)mutable_view.source_table,
          },
  };
  request->diagnostic_collector->module = native_module.module;
  status = consumer(user_data, &native_module);
  request->diagnostic_collector->module = NULL;

  if (iree_status_is_ok(status) && !loom_check_diagnostic_collector_has_error(
                                       request->diagnostic_collector)) {
    verified_view = (loomc_module_interop_view_t){0};
    projected = false;
    status = loom_check_compile_project_module(
        request->public_module, request->diagnostic_collector,
        request->host_allocator, &verified_view, &projected);
  }
  return status;
}

iree_status_t loom_check_compile_source_low(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_source_low_options_t* options,
    loom_check_compile_source_low_consumer_fn_t consumer, void* user_data) {
  loom_check_compile_session_t* session = request->environment->compile_session;
  loomc_module_t* module = request->public_module;
  iree_status_t status = iree_ok_status();

  loomc_pass_program_t* pass_program = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = loom_check_compile_prepare_source_low_program(
        session, options, request->diagnostic_collector, &pass_program);
  }

  loomc_target_profile_t* target_profile = NULL;
  iree_string_view_t function_name = options->function_name;
  if (iree_status_is_ok(status) && pass_program != NULL &&
      !iree_string_view_is_empty(options->target)) {
    loomc_module_interop_view_t native_view = {0};
    bool projected = false;
    if (iree_string_view_is_empty(function_name)) {
      status = loom_check_compile_project_module(
          module, request->diagnostic_collector, request->host_allocator,
          &native_view, &projected);
    }
    if (iree_status_is_ok(status) && iree_string_view_is_empty(function_name) &&
        projected) {
      status = loom_check_compile_select_source_function(
          native_view.module, function_name, &function_name);
    }
    if (iree_status_is_ok(status) && iree_string_view_is_empty(function_name) &&
        !projected) {
      loomc_pass_program_release(pass_program);
      pass_program = NULL;
    }
    if (iree_status_is_ok(status) && pass_program != NULL) {
      status = loom_check_compile_session_select_target_profile(
          session, options->target, &target_profile);
    }
  }

  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_string_view_from_iree(function_name),
      .target_profile = target_profile,
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .specializations = target_profile != NULL ? &specialization : NULL,
      .specialization_count = target_profile != NULL ? 1 : 0,
  };
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = target_profile != NULL ? &target_options : NULL,
  };
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status) && pass_program != NULL) {
    const loomc_allocator_t allocator =
        loomc_allocator_from_iree(request->host_allocator);
    const loomc_status_t operation_status =
        options->report != NULL
            ? loomc_compile_module_with_native_report(
                  session->compiler, session->workspace, pass_program, module,
                  &compile_options, options->report, allocator, &result)
            : loomc_compile_module(session->compiler, session->workspace,
                                   pass_program, module, &compile_options,
                                   allocator, &result);
    status = iree_status_from_loomc(operation_status);
  }
  bool compiled = false;
  if (iree_status_is_ok(status) && result != NULL) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              result, &compiled);
  }

  loomc_module_interop_view_t native_view = {0};
  bool projected = false;
  if (iree_status_is_ok(status) && compiled) {
    status = loom_check_compile_project_module(
        module, request->diagnostic_collector, request->host_allocator,
        &native_view, &projected);
  }
  if (iree_status_is_ok(status) && projected && consumer != NULL) {
    request->diagnostic_collector->module = native_view.module;
    const loom_check_compile_source_low_view_t view = {
        .public_module = module,
        .module = native_view.module,
        .source_resolver =
            {
                .fn = loom_source_table_resolve,
                .user_data = (void*)native_view.source_table,
            },
    };
    status = consumer(user_data, &view);
    request->diagnostic_collector->module = NULL;
  }

  loomc_result_release(result);
  loomc_target_profile_release(target_profile);
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
  if (iree_status_is_ok(status) && compiled) {
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
    loomc_source_t** out_artifact_source) {
  *out_artifact_source = NULL;
  loom_check_compile_session_t* session = request->environment->compile_session;
  loomc_module_t* module = request->public_module;
  iree_status_t status = iree_ok_status();

  const loomc_pass_program_t* pass_program = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = loom_check_compile_get_artifact_pass_program(
        session, options, request->diagnostic_collector, &pass_program);
  }

  loomc_target_profile_t* target_profile = NULL;
  if (iree_status_is_ok(status) && pass_program != NULL &&
      !iree_string_view_is_empty(options->target)) {
    status = loom_check_compile_session_select_target_profile(
        session, options->target, &target_profile);
  }

  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .artifact_format = loomc_string_view_from_iree(options->artifact_format),
  };
  const loomc_string_view_t root = loomc_string_view_from_iree(options->root);
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .roots = iree_string_view_is_empty(options->root) ? NULL : &root,
      .root_count = iree_string_view_is_empty(options->root) ? 0 : 1,
      .target_profile = target_profile,
      .emit_options = &emit_options,
  };
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status) && pass_program != NULL) {
    status = loom_check_compile_artifact_module(
        session, pass_program, module, &compile_options,
        request->diagnostic_collector, request->host_allocator, &result);
  }
  if (iree_status_is_ok(status) && result != NULL) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, 0);
    status = iree_status_from_loomc(loomc_artifact_create_source(
        artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
        loomc_allocator_from_iree(request->host_allocator),
        out_artifact_source));
  }

  loomc_result_release(result);
  loomc_target_profile_release(target_profile);
  return status;
}

iree_status_t loom_check_execute_compile(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_input_request_t* input_request,
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
      &collector, block_pool, allocator, &module);

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
