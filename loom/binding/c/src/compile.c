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
#include "emit.h"
#include "iree/base/internal/atomics.h"
#include "loom/codegen/low/launch_config_program.h"
#include "loom/compile/request.h"
#include "loom/pass/environment.h"
#include "loom/pass/interpreter.h"
#include "loom/target/predicate.h"
#include "loom/target/specialization.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "module.h"
#include "module_bytecode.h"
#include "option_chain.h"
#include "pass_program.h"
#include "pass_trace.h"
#include "product.h"
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

typedef struct loomc_compiled_module_product_t {
  // Generic immutable product interface exposed to callers.
  loomc_product_t base;

  // Allocator used for product-owned metadata.
  loomc_allocator_t allocator;

  // Retained result owning artifact strings and byte sequences.
  loomc_result_t* result;
} loomc_compiled_module_product_t;

static void loomc_compiled_module_product_destroy(
    loomc_product_t* base_product) {
  loomc_compiled_module_product_t* product =
      (loomc_compiled_module_product_t*)base_product;
  const loomc_allocator_t allocator = product->allocator;
  loomc_result_release(product->result);
  loomc_allocator_free(allocator, product);
}

static const loomc_product_descriptor_t
    loomc_compiled_module_product_descriptor_ = {
        .destroy = loomc_compiled_module_product_destroy,
};

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
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, &target_specialization));
  LOOMC_RETURN_IF_ERROR(
      loomc_compile_validate_string_view(options->module_name));
  const loomc_compile_artifact_flags_t known_artifact_flags =
      LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
      LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
      LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON |
      LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG;
  if ((options->artifact_flags & ~known_artifact_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile options contain unknown artifact flags");
  }
  return loomc_config_validate_policy_flags(options->config_flags);
}

static loomc_status_t loomc_compile_validate_artifact_options(
    const loomc_compiler_t* compiler,
    const loomc_compile_artifact_options_t* options,
    loomc_option_chain_t* out_option_chain) {
  *out_option_chain = (loomc_option_chain_t){0};
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile artifact options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile artifact options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options->next,
      LOOMC_OPTION_CHAIN_ALLOW_SANITIZER | LOOMC_OPTION_CHAIN_ALLOW_PASS_TRACE,
      out_option_chain));
  if (options->root_count != 0 && options->roots == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile artifact root_count is non-zero but roots is NULL");
  }
  if (options->excluded_root_count != 0 && options->excluded_roots == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile artifact excluded_root_count is non-zero but excluded_roots "
        "is NULL");
  }
  if (options->root_count != 0 && options->excluded_root_count != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile artifact roots and excluded_roots are mutually exclusive");
  }
  for (loomc_host_size_t i = 0; i < options->root_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_compile_validate_string_view(options->roots[i]));
  }
  for (loomc_host_size_t i = 0; i < options->excluded_root_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_compile_validate_string_view(options->excluded_roots[i]));
  }
  LOOMC_RETURN_IF_ERROR(loomc_config_validate_text_options(options->config));
  if (options->target_profile != NULL) {
    LOOMC_RETURN_IF_ERROR(loomc_target_profile_validate_environment(
        options->target_profile,
        loomc_context_target_environment(compiler->context)));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_make_string_list(
    const loomc_string_view_t* values, loomc_host_size_t count,
    iree_arena_allocator_t* arena, iree_string_view_list_t* out_list) {
  *out_list = (iree_string_view_list_t){0};
  if (count == 0) {
    return loomc_ok_status();
  }
  iree_string_view_t* internal_values = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_arena_allocate_array(
      arena, count, sizeof(*internal_values), (void**)&internal_values)));
  for (loomc_host_size_t i = 0; i < count; ++i) {
    internal_values[i] = iree_string_view_from_loomc(values[i]);
  }
  *out_list = (iree_string_view_list_t){
      .values = internal_values,
      .count = count,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_validate_config_module(
    const loomc_compiler_t* compiler, const loomc_module_t* program_module,
    const loomc_compile_options_t* options) {
  const loomc_module_t* config_module = options ? options->config_module : NULL;
  if (config_module == NULL) {
    return loomc_ok_status();
  }
  if (config_module == program_module) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "config module must be distinct from the program module");
  }
  if (loomc_module_context(config_module) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "config module was created with another context");
  }
  if (loomc_module_const_loom_module(config_module) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "config module does not contain internal IR");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_fail_result_from_status(
    loomc_result_t* result, loomc_string_view_t code, loomc_status_t status) {
  if (!loomc_status_is_result_diagnostic(status)) {
    return status;
  }
  return loomc_result_fail_status_diagnostic_consume(
      result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status);
}

static loomc_status_t loomc_compile_run_pass_program(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loom_module_t* internal_module,
    loom_source_resolver_t source_resolver,
    loom_function_version_owner_t* function_version_owner,
    loom_kernel_launch_config_program_t* launch_config_program,
    loom_target_compile_report_t* compile_report,
    const loomc_pass_trace_options_t* pass_trace_options,
    loomc_result_t* result) {
  const loomc_target_pass_environment_t* target_pass_environment =
      loomc_context_target_pass_environment(compiler->context);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      result, /*source=*/NULL, internal_module, source_resolver,
      LOOM_EMITTER_PASS,
      target_pass_environment
          ? &target_pass_environment->diagnostic_type_print_options
          : NULL,
      &capture);
  loom_codegen_pass_environment_storage_t codegen_environment_storage = {0};
  const loom_pass_environment_t pass_environment =
      loomc_codegen_pass_environment_storage_initialize(
          target_pass_environment,
          loomc_context_cleanup_pattern_registry(compiler->context),
          function_version_owner,
          launch_config_program != NULL
              ? loom_kernel_launch_config_program_capability(
                    launch_config_program)
              : NULL,
          compile_report, &codegen_environment_storage);
  loom_target_pass_predicate_provider_storage_t predicate_storage = {0};
  loom_pass_predicate_provider_t predicate_provider = {0};
  if (loomc_context_target_pass_environment(compiler->context) != NULL) {
    loom_target_pass_predicate_provider_storage_initialize(
        loomc_workspace_block_pool(workspace), &predicate_storage);
    predicate_provider =
        loom_target_pass_predicate_provider(&predicate_storage);
  }
  loomc_pass_trace_state_t pass_trace_state = {0};
  loom_pass_trace_t* pass_trace = NULL;
  if (pass_trace_options != NULL) {
    loomc_pass_trace_state_initialize(
        pass_trace_options, loomc_pass_program_trace_stage(pass_program),
        loomc_context_target_pass_environment(compiler->context),
        &function_version_owner->list, loomc_workspace_block_pool(workspace),
        &pass_trace_state);
    pass_trace = &pass_trace_state.trace;
  }
  const loom_pass_interpreter_options_t interpreter_options = {
      .block_pool = loomc_workspace_block_pool(workspace),
      .predicate_provider = predicate_provider,
      .diagnostic_emitter =
          {
              .fn = loomc_diagnostic_capture_emission,
              .user_data = &capture,
          },
      .environment = pass_environment,
      .function_versions = &function_version_owner->list,
      .trace = pass_trace,
  };
  loom_pass_run_result_t run_result = {0};
  loomc_status_t status =
      loomc_status_from_iree(loom_pass_interpreter_run_program(
          loomc_pass_program_loom_pass_program(pass_program), internal_module,
          &interpreter_options, &run_result));
  if (!loomc_status_is_ok(status)) {
    if (pass_trace_state.callback_failed) {
      return status;
    }
    if (!loomc_status_is_result_diagnostic(status)) {
      return status;
    }
    if (run_result.error_count == 0) {
      return loomc_result_fail_status_diagnostic_consume(
          result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
          loomc_make_cstring_view("PASS_PROGRAM/EXECUTION"), status);
    }
    loomc_status_free(status);
  }
  return run_result.error_count != 0
             ? loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED)
             : loomc_ok_status();
}

static loomc_status_t loomc_compile_specialize_functions(
    const loomc_target_environment_t* target_environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings, loom_module_t* module,
    loom_source_resolver_t source_resolver, loomc_result_t* result,
    loom_function_version_owner_t* function_versions) {
  if (requests.count == 0 && bindings.count == 0) {
    return loomc_ok_status();
  }

  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      result, /*source=*/NULL, module, source_resolver, LOOM_EMITTER_PASS,
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL,
      &capture);
  uint32_t error_count = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_target_specialize_functions(
      loomc_target_environment_loom_target_environment(target_environment),
      module, requests, bindings,
      (iree_diagnostic_emitter_t){
          .fn = loomc_diagnostic_capture_emission,
          .user_data = &capture,
      },
      function_versions, &error_count)));
  if (error_count != 0) {
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

static loomc_status_t loomc_compile_add_launch_config_artifact(
    loomc_result_t* result, const loomc_compile_options_t* options,
    const loomc_module_t* module, const loom_module_t* launch_config_module) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  loomc_string_view_t identifier = loomc_string_view_empty();
  loomc_status_t status = loomc_compile_make_artifact_identifier(
      options, loomc_make_cstring_view(".launch-config.loombc"),
      loomc_make_cstring_view("launch-config.loombc"), allocator, &identifier);

  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_module_serialize_internal_bytecode_to_source(
        loomc_module_context(module), launch_config_module, identifier,
        /*projection=*/NULL, allocator, &source);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_result_take_source_artifact(
        result, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE), source);
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
    const loomc_target_specialization_options_t* target_options,
    const loomc_config_application_result_t* config_application,
    loomc_host_size_t artifact_count, loom_output_stream_t* stream) {
  const loomc_string_view_t module_name =
      options ? options->module_name : loomc_string_view_empty();
  const loomc_config_policy_flags_t config_flags =
      options ? options->config_flags : 0;
  const loomc_host_size_t config_materialized_count =
      config_application ? config_application->materialized_count : 0;
  const loomc_host_size_t config_ignored_count =
      config_application ? config_application->ignored_count : 0;

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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"has_config_module\":%s",
      options != NULL && options->config_module != NULL ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"config_definition_count\":%zu",
      (size_t)(config_materialized_count + config_ignored_count)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"config_materialized_count\":%zu",
      (size_t)config_materialized_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"config_ignored_count\":%zu", (size_t)config_ignored_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"config_flags\":%u", (unsigned)config_flags));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"target_specialization_count\":%zu",
      target_options ? (size_t)target_options->specialization_count : 0));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"target_binding_count\":%zu",
      target_options ? (size_t)target_options->target_binding_count : 0));
  return loom_output_stream_write_cstring(stream, "}\n");
}

static loomc_status_t loomc_compile_add_report_json_artifact(
    loomc_result_t* result, const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t* target_options,
    const loomc_config_application_result_t* config_application,
    loomc_host_size_t artifact_count) {
  loomc_allocator_t allocator = loomc_result_allocator(result);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loomc_status_t status =
      loomc_status_from_iree(loomc_compile_write_report_json(
          options, result, target_options, config_application, artifact_count,
          &stream));

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
    const loomc_target_specialization_options_t* target_options,
    const loomc_config_application_result_t* config_application,
    const loomc_module_t* module, const loom_module_t* launch_config_module) {
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
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG)) {
    status = loomc_compile_add_launch_config_artifact(result, options, module,
                                                      launch_config_module);
  }
  if (loomc_status_is_ok(status) &&
      iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON)) {
    status = loomc_compile_add_report_json_artifact(
        result, options, target_options, config_application,
        loomc_result_artifact_count(result));
  }
  return status;
}

static loomc_status_t loomc_compiled_module_product_create(
    loomc_result_t* result, loomc_host_size_t export_count,
    loomc_allocator_t allocator, loomc_product_t** out_product) {
  *out_product = NULL;
  const loomc_host_size_t artifact_count = loomc_result_artifact_count(result);
  loomc_host_size_t artifact_storage_size = 0;
  loomc_host_size_t allocation_size = sizeof(loomc_compiled_module_product_t);
  if (!iree_host_size_checked_mul(artifact_count, sizeof(loomc_artifact_t),
                                  &artifact_storage_size) ||
      !iree_host_size_checked_add(allocation_size, artifact_storage_size,
                                  &allocation_size)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "compiled product metadata is too large");
  }

  loomc_compiled_module_product_t* product = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&product));
  memset(product, 0, sizeof(*product));
  product->allocator = allocator;
  product->result = result;
  loomc_result_retain(result);

  loomc_artifact_t* artifacts = (loomc_artifact_t*)(product + 1);
  for (loomc_host_size_t i = 0; i < artifact_count; ++i) {
    artifacts[i] = *loomc_result_artifact_at(result, i);
  }
  loomc_product_initialize(&loomc_compiled_module_product_descriptor_,
                           artifacts, artifact_count, export_count,
                           /*requirement_count=*/0, &product->base);
  *out_product = &product->base;
  return loomc_ok_status();
}

static loomc_status_t loomc_compile_resolve_target_specialization(
    const loomc_compiler_t* compiler, const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_compile_validate_options(options));
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options ? options->next : NULL, out_target_specialization));
  return loomc_target_specialization_options_validate_environment(
      *out_target_specialization,
      loomc_context_target_environment(compiler->context));
}

// Runs one prepared program over an already-configured module. Internal target
// specialization intent is consumed directly so root materialization need not
// translate back through the public row representation.
static loomc_status_t loomc_compile_prepared_module_into_result(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t* target_specialization,
    loom_target_specialization_request_list_t target_specializations,
    loom_target_declaration_binding_list_t target_bindings,
    const loomc_config_application_result_t* config_application,
    loom_target_compile_report_t* compile_report,
    const loomc_pass_trace_options_t* pass_trace_options,
    loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(compiler);
  IREE_ASSERT_ARGUMENT(workspace);
  IREE_ASSERT_ARGUMENT(pass_program);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(result);
  loom_module_t* internal_module = loomc_module_loom_module(module);
  IREE_ASSERT_ARGUMENT(internal_module);

  const loomc_target_environment_t* context_target_environment =
      loomc_context_target_environment(compiler->context);
  loom_function_version_owner_t* function_versions =
      loomc_module_function_version_owner(module);
  loom_kernel_launch_config_program_t launch_config_program = {0};
  bool launch_config_program_initialized = false;
  const bool launch_config_requested =
      options != NULL &&
      iree_any_bit_set(options->artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
  const loom_module_t* launch_config_module = NULL;
  loomc_status_t status = loomc_ok_status();
  if (loomc_result_succeeded(result)) {
    status = loomc_module_verify(module, context_target_environment, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_specialize_functions(
        context_target_environment, target_specializations, target_bindings,
        internal_module, loomc_module_source_resolver(module), result,
        function_versions);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_requested) {
    status =
        loomc_status_from_iree(loom_kernel_launch_config_program_initialize(
            internal_module->context, loomc_workspace_block_pool(workspace),
            iree_allocator_from_loomc(loomc_result_allocator(result)),
            &launch_config_program));
    launch_config_program_initialized = loomc_status_is_ok(status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_run_pass_program(
        compiler, workspace, pass_program, internal_module,
        loomc_module_source_resolver(module), function_versions,
        launch_config_requested ? &launch_config_program : NULL, compile_report,
        pass_trace_options, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_requested) {
    status = loomc_status_from_iree(loom_kernel_launch_config_program_finalize(
        &launch_config_program, internal_module,
        loomc_workspace_block_pool(workspace), &launch_config_module));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_module != NULL) {
    status = loomc_result_verify_loom_module(
        launch_config_module, (loom_source_resolver_t){0}, result);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_emit_requested_artifacts(
        result, options, target_specialization, config_application, module,
        launch_config_module);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
    loomc_module_invalidate_compilation(module);
  }
  if (launch_config_program_initialized) {
    loom_kernel_launch_config_program_deinitialize(&launch_config_program);
  }
  return status;
}

// Compiles a validated mutable module into an existing succeeded result.
//
// Deserialization-backed callers reuse their parse result so diagnostics keep
// one stable operation order and no merge or second result allocation is
// required.
static loomc_status_t loomc_compile_module_into_result(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(compiler);
  IREE_ASSERT_ARGUMENT(workspace);
  IREE_ASSERT_ARGUMENT(pass_program);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(result);
  IREE_ASSERT(loomc_result_succeeded(result));
  loom_module_t* internal_module = loomc_module_loom_module(module);
  IREE_ASSERT_ARGUMENT(internal_module);

  const loomc_target_environment_t* context_target_environment =
      loomc_context_target_environment(compiler->context);
  loom_function_version_owner_t* function_versions =
      loomc_module_function_version_owner(module);
  loomc_config_application_result_t config_application = {0};
  loom_target_specialization_request_list_t target_specializations = {0};
  loom_target_declaration_binding_list_t target_bindings = {0};

  loomc_status_t status =
      loomc_module_verify(module, context_target_environment, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_module_t* config_module =
        options ? options->config_module : NULL;
    const loomc_config_apply_module_options_t config_apply_options = {
        .config_module = loomc_module_const_loom_module(config_module),
        .config_source_resolver = loomc_module_source_resolver(config_module),
        .target_module = internal_module,
        .target_source_resolver = loomc_module_source_resolver(module),
        .binding_sink = loomc_module_config_binding_sink(module),
        .policy_flags = options ? options->config_flags : 0,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
    };
    status =
        loomc_config_apply_module(&config_apply_options, &config_application);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      target_specialization != NULL) {
    status = loomc_target_specialization_options_make_lists(
        target_specialization, function_versions->arena,
        &target_specializations, &target_bindings);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_prepared_module_into_result(
        compiler, workspace, pass_program, module, options,
        target_specialization, target_specializations, target_bindings,
        &config_application, /*compile_report=*/NULL,
        /*pass_trace_options=*/NULL, result);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
    loomc_module_invalidate_compilation(module);
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
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_compile_resolve_target_specialization(
      compiler, options, &target_specialization));
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_compile_validate_config_module(compiler, module, options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_RESULT_STATE_SUCCEEDED,
      loomc_context_source_retention(compiler->context), allocator, &result));
  loomc_status_t status = loomc_compile_module_into_result(
      compiler, workspace, pass_program, module, options, target_specialization,
      result);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_compile_artifact(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_artifact_options_t* options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (compiler == NULL || workspace == NULL || module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, and module must not be NULL");
  }
  if (loomc_module_context(module) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module was created with another context");
  }
  if (pass_program != NULL &&
      loomc_pass_program_context(pass_program) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass program was created with another context");
  }
  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(compiler->context);
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "compiler context has no target environment");
  }
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_compile_validate_artifact_options(
      compiler, options, &option_chain));
  if (pass_program != NULL && option_chain.has_sanitizer) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "sanitizer options require the emitter default pass program");
  }
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  const loomc_config_options_t* config = options ? options->config : NULL;
  const bool require_resolved_config =
      config && iree_any_bit_set(config->flags,
                                 LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED);

  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_result_create(
      LOOMC_RESULT_STATE_SUCCEEDED,
      loomc_context_source_retention(compiler->context), allocator, &result);
  loomc_emit_transaction_t emit_transaction = {0};
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_transaction_initialize(
        options ? options->emit_options : NULL, result, &emit_transaction);
  }
  if (loomc_status_is_ok(status)) {
    loomc_emit_transaction_set_diagnostic_context(
        &emit_transaction, internal_module, target_environment);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_verify(module, target_environment, result);
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_module_invalidate_compilation(module);
    loomc_config_options_t materialization_config =
        config ? *config : (loomc_config_options_t){0};
    // Root materialization may prune unresolved config declarations that are
    // unreachable from the selected program. Validate the survivors below.
    materialization_config.flags &= ~LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    const loomc_config_apply_text_to_module_options_t config_apply_options = {
        .config = config ? &materialization_config : NULL,
        .module = internal_module,
        .source_resolver = loomc_module_source_resolver(module),
        .binding_sink = loomc_module_config_binding_sink(module),
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = allocator,
    };
    status = loomc_config_apply_text_to_module(&config_apply_options);
  }

  loom_function_version_owner_t* function_versions =
      loomc_module_function_version_owner(module);
  iree_string_view_list_t roots = {0};
  iree_string_view_list_t excluded_roots = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_make_string_list(options ? options->roots : NULL,
                                            options ? options->root_count : 0,
                                            function_versions->arena, &roots);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_make_string_list(
        options ? options->excluded_roots : NULL,
        options ? options->excluded_root_count : 0, function_versions->arena,
        &excluded_roots);
  }

  loom_compile_request_t request = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loom_compile_request_options_t request_options = {
        .roots = roots,
        .format = iree_string_view_from_loomc(
            loomc_emit_transaction_artifact_format(&emit_transaction)),
        .target_profile = options && options->target_profile
                              ? loomc_target_profile_loom_target_profile(
                                    options->target_profile)
                              : NULL,
        .excluded_roots = excluded_roots,
    };
    status = loomc_status_from_iree(loom_compile_request_resolve(
        internal_module, &request_options,
        loomc_target_environment_loom_target_environment(target_environment),
        function_versions->arena, &request));
    if (!loomc_status_is_ok(status)) {
      status = loomc_compile_fail_result_from_status(
          result, loomc_make_cstring_view("COMPILE/REQUEST"), status);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    if (emit_transaction.options.artifact_manifest_mode !=
            LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
        request.selection.kind != LOOM_COMPILE_ENTRY_KIND_KERNEL) {
      status = loomc_compile_fail_result_from_status(
          result, loomc_make_cstring_view("COMPILE/REQUEST"),
          loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "artifact manifests are only valid for loadable kernel "
              "formats"));
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_emit_transaction_bind_emitter(&emit_transaction,
                                        request.target_emitter);
  }

  loom_target_specialization_request_list_t target_specializations = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loom_source_table_projection_t sources = {
        .table = *loomc_module_source_table(module),
        .arena = function_versions->arena,
    };
    const loom_source_resolver_t source_resolver = {
        .fn = loom_source_table_resolve,
        .user_data = &sources.table,
    };
    const loomc_target_pass_environment_t* pass_environment =
        loomc_target_environment_pass_environment(target_environment);
    loomc_diagnostic_capture_t capture;
    loomc_diagnostic_capture_initialize(
        result, /*source=*/NULL, internal_module, source_resolver,
        LOOM_EMITTER_PASS,
        pass_environment ? &pass_environment->diagnostic_type_print_options
                         : NULL,
        &capture);
    const loom_target_entry_options_t entry_options = {
        .diagnostic_sink =
            {
                .fn = loomc_diagnostic_capture,
                .user_data = &capture,
            },
        .source_resolver = source_resolver,
    };
    uint32_t error_count = 0;
    status = loomc_status_from_iree(loom_compile_request_materialize(
        &request,
        loomc_target_environment_loom_target_environment(target_environment),
        &entry_options, internal_module,
        LOOM_COMPILE_REQUEST_SOURCE_TRANSFERRED, &sources,
        function_versions->arena, loomc_module_block_pool(module),
        &internal_module, &target_specializations, &error_count));
    loomc_status_t source_status = loomc_module_replace_source_table(
        module, internal_module, &sources.table);
    if (!loomc_status_is_ok(source_status)) {
      loomc_module_clear_sources(module, internal_module);
    }
    status = loomc_status_join(status, source_status);
    loomc_module_adopt_loom_module_replacement(module, internal_module,
                                               LOOMC_MODULE_INPUT_UNVERIFIED);
    loomc_emit_transaction_set_diagnostic_context(
        &emit_transaction, internal_module, target_environment);
    if (!loomc_status_is_ok(status)) {
      status = loomc_compile_fail_result_from_status(
          result, loomc_make_cstring_view("COMPILE/MATERIALIZE"), status);
    }
    if (loomc_status_is_ok(status) && error_count != 0) {
      status = loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      require_resolved_config) {
    status = loomc_status_from_iree(loom_tooling_config_require_resolved_module(
        loomc_module_loom_module(module), NULL));
    if (!loomc_status_is_ok(status)) {
      status = loomc_compile_fail_result_from_status(
          result, loomc_make_cstring_view("CONFIG/INVALID"), status);
    }
  }

  loomc_pass_program_t* default_pass_program = NULL;
  const loomc_pass_program_t* selected_pass_program = pass_program;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      selected_pass_program == NULL) {
    loom_target_pipeline_options_t default_pipeline_options =
        request.target_emitter->default_pipeline_options;
    if (option_chain.has_sanitizer) {
      default_pipeline_options.sanitizer = option_chain.sanitizer;
    }
    status = loomc_pass_program_create_from_internal_target_pipeline(
        compiler->context, &default_pipeline_options, allocator, result,
        &default_pass_program);
    selected_pass_program = default_pass_program;
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_prepared_module_into_result(
        compiler, workspace, selected_pass_program, module,
        /*options=*/NULL, /*target_specialization=*/NULL,
        target_specializations, (loom_target_declaration_binding_list_t){0},
        /*config_application=*/NULL,
        loomc_emit_transaction_compile_report(&emit_transaction),
        option_chain.pass_trace, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_emit_transaction_emit(&emit_transaction, target_environment,
                                         workspace, module);
  }
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
    loomc_emit_transaction_record_status(&emit_transaction,
                                         IREE_STATUS_FAILED_PRECONDITION);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_transaction_finish(&emit_transaction);
  }

  loomc_pass_program_release(default_pass_program);
  loomc_emit_transaction_deinitialize(&emit_transaction);
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
    loomc_module_invalidate_compilation(module);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

static loomc_status_t loomc_compile_validate_request_roots(
    const loomc_module_t* module, const loomc_request_t* request) {
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  const loomc_request_root_t* roots = loomc_request_roots(request);
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i].module_ordinal != 0) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "request root addresses an unavailable module");
    }
    if (roots[i].symbol_ordinal >= internal_module->symbols.count) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "request root addresses an unavailable symbol");
    }
  }
  return loomc_ok_status();
}

loomc_status_t loomc_compile_request(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, const loomc_request_t* request,
    const loomc_compile_options_t* options, loomc_allocator_t allocator,
    loomc_product_t** out_product, loomc_result_t** out_result) {
  if (out_product == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_product and out_result must not be NULL");
  }
  *out_product = NULL;
  *out_result = NULL;
  if (compiler == NULL || workspace == NULL || pass_program == NULL ||
      request == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, pass_program, and request must not be NULL");
  }
  if (loomc_request_product_descriptor(request) !=
      loomc_compiled_module_product_descriptor()) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request does not require a compiled module product");
  }
  if (loomc_pass_program_context(pass_program) != compiler->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass program was created with another context");
  }
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_compile_resolve_target_specialization(
      compiler, options, &target_specialization));
  LOOMC_RETURN_IF_ERROR(
      loomc_compile_validate_config_module(compiler, NULL, options));

  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_product_t* product = NULL;
  loomc_status_t status = loomc_module_deserialize_from_source(
      compiler->context, workspace, loomc_request_source(request),
      /*options=*/NULL, allocator, &module, &result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_validate_request_roots(module, request);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_module_into_result(compiler, workspace, pass_program,
                                              module, options,
                                              target_specialization, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compiled_module_product_create(
        result, loomc_request_root_count(request), allocator, &product);
  }
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_product = product;
      product = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loomc_product_release(product);
  loomc_result_release(result);
  loomc_module_release(module);
  return status;
}

const loomc_product_descriptor_t* loomc_compiled_module_product_descriptor(
    void) {
  return &loomc_compiled_module_product_descriptor_;
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
