// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/compile.h"

#include <string.h>

#include "config.h"
#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/atomics.h"
#include "loom/pass/environment.h"
#include "loom/pass/interpreter.h"
#include "loom/target/predicate.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "module.h"
#include "pass_program.h"
#include "result.h"
#include "source.h"
#include "target.h"
#include "workspace.h"

struct loomc_compiler_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for compiler-owned storage.
  loomc_allocator_t allocator;

  // Context retained by the prepared compiler.
  loomc_context_t* context;
};

typedef struct loomc_compile_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
} loomc_compile_diagnostic_capture_t;

static loomc_status_t loomc_compile_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_validate_compiler_options(
    const loomc_compiler_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compiler options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compiler options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "compiler option extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_validate_options(
    const loomc_compile_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile options structure_size is too small");
  }
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_target_selection_options_resolve(options->next, &target_selection));
  LOOMC_RETURN_IF_ERROR(
      loomc_compile_validate_string_view(options->module_name));
  const loomc_compile_artifact_flags_t known_artifact_flags =
      LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
      LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
      LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON;
  if ((options->artifact_flags & ~known_artifact_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile options contain unknown artifact flags");
  }
  return loomc_config_validate_options(&options->config);
}

static iree_status_t loomc_compile_capture_diagnostic_emission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_compile_diagnostic_capture_t* capture =
      (loomc_compile_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static loomc_status_t loomc_compile_run_pass_program(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loom_module_t* internal_module,
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref,
    loomc_result_t* result) {
  loomc_compile_diagnostic_capture_t capture = {
      .result = result,
  };
  loom_low_pass_environment_storage_t low_environment_storage = {0};
  loom_pass_environment_t pass_environment = loom_pass_environment_empty();
  loom_target_pass_predicate_provider_storage_t predicate_storage = {0};
  loom_pass_predicate_provider_t predicate_provider = {0};
  const loomc_target_pass_environment_t* target_environment =
      loomc_context_target_pass_environment(compiler->context);
  if (target_environment != NULL) {
    pass_environment = loomc_target_pass_environment_make_loom_pass_environment(
        target_environment, target_selection, target_ref,
        &low_environment_storage);
    loom_target_pass_predicate_provider_storage_initialize(
        loomc_workspace_block_pool(workspace), &predicate_storage);
    predicate_provider =
        loom_target_pass_predicate_provider(&predicate_storage);
  }
  loom_pass_interpreter_options_t interpreter_options = {
      .block_pool = loomc_workspace_block_pool(workspace),
      .predicate_provider = predicate_provider,
      .diagnostic_emitter =
          {
              .fn = loomc_compile_capture_diagnostic_emission,
              .user_data = &capture,
          },
      .environment = pass_environment,
  };
  loom_pass_run_result_t run_result = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_pass_interpreter_run_module(
      loomc_pass_program_loom_pass_program(pass_program), internal_module,
      &interpreter_options, &run_result)));
  if (run_result.error_count != 0) {
    return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_make_artifact_identifier(
    const loomc_compile_options_t* options, loomc_string_view_t file_extension,
    loomc_string_view_t fallback_identifier, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier) {
  *out_identifier = loomc_string_view_empty();
  loomc_string_view_t module_name =
      options ? options->module_name : loomc_string_view_empty();
  if (loomc_string_view_is_empty(module_name)) {
    return loomc_string_view_clone(fallback_identifier, allocator,
                                   out_identifier);
  }

  const loomc_host_size_t identifier_length =
      module_name.size + file_extension.size;
  char* identifier = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, identifier_length, (void**)&identifier));
  memcpy(identifier, module_name.data, module_name.size);
  memcpy(identifier + module_name.size, file_extension.data,
         file_extension.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_result_take_source_artifact(
    loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_source_t* source) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  loomc_allocator_t allocator = loomc_result_allocator(result);
  loomc_status_t status = loomc_source_take_contents(source, &contents);
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, kind, format, loomc_source_identifier(source), contents);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, (void*)contents.data);
  }
  return status;
}

static loomc_status_t loomc_compile_add_module_artifact(
    loomc_result_t* result, const loomc_compile_options_t* options,
    const loomc_module_t* module, loomc_source_format_t source_format,
    loomc_string_view_t artifact_format, loomc_string_view_t file_extension,
    loomc_string_view_t fallback_identifier) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  loomc_string_view_t identifier = loomc_string_view_empty();
  loomc_status_t status = loomc_compile_make_artifact_identifier(
      options, file_extension, fallback_identifier, allocator, &identifier);

  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    loomc_module_serialize_options_t serialize_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        .structure_size = sizeof(serialize_options),
        .format = source_format,
        .identifier = identifier,
    };
    status = loomc_module_serialize_to_source(module, &serialize_options,
                                              allocator, &source);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_result_take_source_artifact(
        result, LOOMC_ARTIFACT_KIND_MODULE, artifact_format, source);
  }

  loomc_source_release(source);
  loomc_allocator_free(allocator, (void*)identifier.data);
  return status;
}

static iree_status_t loomc_compile_write_json_string_field(
    loom_output_stream_t* stream, const char* field_name,
    loomc_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\":"));
  return loom_json_write_escaped_string(stream,
                                        iree_string_view_from_loomc(value));
}

static iree_status_t loomc_compile_write_report_json(
    const loomc_compile_options_t* options, const loomc_result_t* result,
    loomc_host_size_t artifact_count, loom_output_stream_t* stream) {
  const loomc_config_options_t empty_config_options = {0};
  const loomc_config_options_t* config_options =
      options ? &options->config : &empty_config_options;
  const loomc_string_view_t module_name =
      options ? options->module_name : loomc_string_view_empty();

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(loomc_compile_write_json_string_field(
      stream, "kind", loomc_make_cstring_view("loomc.compile")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loomc_compile_write_json_string_field(
      stream, "state",
      loomc_result_succeeded(result) ? loomc_make_cstring_view("succeeded")
                                     : loomc_make_cstring_view("failed")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_count\":%zu",
      (size_t)loomc_result_diagnostic_count(result)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"artifact_count\":%zu", (size_t)artifact_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loomc_compile_write_json_string_field(
      stream, "module_name", module_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, ",\"config_binding_count\":%zu",
                                      (size_t)config_options->binding_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"has_config_json\":%s",
      loomc_string_view_is_empty(config_options->json_object) ? "false"
                                                              : "true"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"config_flags\":%u", (unsigned)config_options->flags));
  return loom_output_stream_write_cstring(stream, "}\n");
}

static loomc_status_t loomc_compile_add_report_json_artifact(
    loomc_result_t* result, const loomc_compile_options_t* options,
    loomc_host_size_t artifact_count) {
  loomc_allocator_t allocator = loomc_result_allocator(result);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loomc_status_t status =
      loomc_status_from_iree(loomc_compile_write_report_json(
          options, result, artifact_count, &stream));

  char* report_storage = NULL;
  iree_host_size_t report_length = 0;
  if (loomc_status_is_ok(status)) {
    report_length = iree_string_builder_size(&builder);
    report_storage = iree_string_builder_take_storage(&builder);
  }

  loomc_string_view_t identifier = loomc_string_view_empty();
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_make_artifact_identifier(
        options, loomc_make_cstring_view(".compile-report.json"),
        loomc_make_cstring_view("compile-report.json"), allocator, &identifier);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_JSON), identifier,
        loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) {
    report_storage = NULL;
  }

  loomc_allocator_free(allocator, (void*)identifier.data);
  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_compile_emit_requested_artifacts(
    loomc_result_t* result, const loomc_compile_options_t* options,
    const loomc_module_t* module) {
  const loomc_compile_artifact_flags_t artifact_flags =
      options ? options->artifact_flags : 0;
  if (artifact_flags == 0) {
    return loomc_ok_status();
  }

  loomc_status_t status = loomc_ok_status();
  if (loomc_result_succeeded(result) &&
      iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT)) {
    status = loomc_compile_add_module_artifact(
        result, options, module, LOOMC_SOURCE_FORMAT_TEXT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT),
        loomc_make_cstring_view(".loom"),
        loomc_make_cstring_view("module.loom"));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE)) {
    status = loomc_compile_add_module_artifact(
        result, options, module, LOOMC_SOURCE_FORMAT_BYTECODE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
        loomc_make_cstring_view(".loombc"),
        loomc_make_cstring_view("module.loombc"));
  }
  if (loomc_status_is_ok(status) &&
      iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON)) {
    status = loomc_compile_add_report_json_artifact(
        result, options, loomc_result_artifact_count(result));
  }
  return status;
}

loomc_status_t loomc_compiler_create(loomc_context_t* context,
                                     const loomc_compiler_options_t* options,
                                     loomc_allocator_t allocator,
                                     loomc_compiler_t** out_compiler) {
  if (out_compiler == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_compiler must not be NULL");
  }
  *out_compiler = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_compile_validate_compiler_options(options));

  loomc_compiler_t* compiler = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*compiler), (void**)&compiler));
  memset(compiler, 0, sizeof(*compiler));
  iree_atomic_ref_count_init(&compiler->ref_count);
  compiler->allocator = allocator;
  compiler->context = context;
  loomc_context_retain(context);
  *out_compiler = compiler;
  return loomc_ok_status();
}

loomc_status_t loomc_compile_module(loomc_compiler_t* compiler,
                                    loomc_workspace_t* workspace,
                                    const loomc_pass_program_t* pass_program,
                                    loomc_module_t* module,
                                    const loomc_compile_options_t* options,
                                    loomc_allocator_t allocator,
                                    loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (compiler == NULL || workspace == NULL || pass_program == NULL ||
      module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, pass_program, and module must not be NULL");
  }
  if (loomc_module_context(module) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module was created with another context");
  }
  if (loomc_pass_program_context(pass_program) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass program was created with another context");
  }
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  LOOMC_RETURN_IF_ERROR(loomc_compile_validate_options(options));
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_target_selection_options_resolve(
      options ? options->next : NULL, &target_selection));
  const loomc_target_environment_t* context_target_environment =
      loomc_context_target_environment(compiler->context);
  LOOMC_RETURN_IF_ERROR(loomc_target_selection_validate_environment(
      target_selection, context_target_environment));
  const loom_target_selection_t internal_target_selection =
      loomc_target_selection_loom_target_selection(target_selection);
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  const loom_target_environment_t* internal_target_environment =
      loomc_target_environment_loom_target_environment(
          context_target_environment);
  if (internal_target_environment != NULL &&
      !loom_target_selection_is_empty(internal_target_selection)) {
    LOOMC_RETURN_IF_ERROR(
        loomc_status_from_iree(loom_target_environment_materialize_selection(
            internal_target_environment, internal_module,
            internal_target_selection, &target_ref)));
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_status_t status =
      loomc_result_verify_loom_module(internal_module, /*source=*/NULL, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_config_options_t empty_config_options = {0};
    const loomc_config_options_t* config_options =
        options ? &options->config : &empty_config_options;
    loomc_config_apply_to_module_options_t config_apply_options = {
        .config = config_options,
        .module = internal_module,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = allocator,
    };
    status = loomc_config_apply_to_module(&config_apply_options);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_run_pass_program(
        compiler, workspace, pass_program, internal_module,
        internal_target_selection, target_ref, result);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_emit_requested_artifacts(result, options, module);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

void loomc_compiler_retain(loomc_compiler_t* compiler) {
  if (compiler == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&compiler->ref_count);
}

void loomc_compiler_release(loomc_compiler_t* compiler) {
  if (compiler == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&compiler->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = compiler->allocator;
  loomc_context_release(compiler->context);
  loomc_allocator_free(allocator, compiler);
}
