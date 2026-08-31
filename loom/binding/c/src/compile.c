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
#include "iree/base/byte_sequence.h"
#include "iree/base/internal/atomics.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/environment.h"
#include "loom/pass/interpreter.h"
#include "loom/target/predicate.h"
#include "loom/target/specialization.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "module.h"
#include "module_bytecode.h"
#include "pass_program.h"
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

typedef struct loomc_compile_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;

  // Diagnostic subsystem attributed to captured emissions.
  loom_emitter_t emitter;

  // Number of error diagnostics captured.
  uint32_t error_count;
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

static iree_status_t loomc_compile_capture_diagnostic_emission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_compile_diagnostic_capture_t* capture =
      (loomc_compile_diagnostic_capture_t*)user_data;
  if (emission != NULL && emission->error != NULL &&
      emission->error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++capture->error_count;
  }
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, capture->emitter, emission));
}

static loomc_status_t loomc_compile_run_pass_program(
    loomc_workspace_t* workspace, const loomc_pass_program_t* pass_program,
    loom_module_t* internal_module,
    loom_function_version_owner_t* function_version_owner,
    const loom_pass_environment_capability_t* additional_capability,
    loomc_result_t* result) {
  loomc_compile_diagnostic_capture_t capture = {
      .result = result,
      .emitter = LOOM_EMITTER_PASS,
  };
  loom_low_pass_environment_storage_t low_environment_storage = {0};
  loom_pass_environment_t pass_environment = loom_pass_environment_empty();
  loom_target_pass_predicate_provider_storage_t predicate_storage = {0};
  loom_pass_predicate_provider_t predicate_provider = {0};
  const loomc_target_pass_environment_t* target_environment =
      loomc_context_target_pass_environment(
          loomc_pass_program_context(pass_program));
  if (target_environment != NULL) {
    pass_environment = loomc_target_pass_environment_make_loom_pass_environment(
        target_environment, function_version_owner, &low_environment_storage);
    loom_target_pass_predicate_provider_storage_initialize(
        loomc_workspace_block_pool(workspace), &predicate_storage);
    predicate_provider =
        loom_target_pass_predicate_provider(&predicate_storage);
  }
  const loom_pass_environment_capability_t* extended_capabilities
      [IREE_ARRAYSIZE(low_environment_storage.capabilities) + 1];
  if (additional_capability != NULL) {
    for (iree_host_size_t i = 0; i < pass_environment.capability_count; ++i) {
      extended_capabilities[i] = pass_environment.capabilities[i];
    }
    extended_capabilities[pass_environment.capability_count] =
        additional_capability;
    pass_environment = loom_pass_environment_make(
        extended_capabilities, pass_environment.capability_count + 1);
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
      .function_versions = &function_version_owner->list,
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

static loomc_status_t loomc_compile_emit_launch_config_program(
    const loomc_target_pass_environment_t* pass_environment,
    const loom_target_launch_config_compiler_t* launch_config_compiler,
    const loom_pass_environment_capability_t* launch_config_capability,
    const loom_function_version_list_t* function_versions,
    loom_module_t* module, loomc_result_t* result,
    iree_arena_allocator_t* scratch_arena,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  loomc_compile_diagnostic_capture_t capture = {
      .result = result,
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  const loom_target_emit_request_t request = {
      .target_environment = pass_environment->target_environment,
      .low_descriptor_registry =
          &pass_environment->low_descriptor_registry.registry,
      .module = module,
      .function_versions = function_versions,
      .identifier = launch_config_compiler->default_identifier,
      .diagnostic_emitter =
          {
              .fn = loomc_compile_capture_diagnostic_emission,
              .user_data = &capture,
          },
      .scratch_arena = scratch_arena,
      .allocator = iree_allocator_from_loomc(loomc_result_allocator(result)),
  };
  iree_status_t emit_status = launch_config_compiler->emit(
      launch_config_capability, &request, out_artifact);
  loomc_status_t status = loomc_status_from_iree(emit_status);
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_result_fail_status_diagnostic_consume(
        result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        loomc_make_cstring_view("COMPILE/LAUNCH_CONFIG"), status);
  }
  if (loomc_status_is_ok(status) && capture.error_count != 0) {
    status = loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      out_artifact->contents == NULL) {
    status = loomc_make_status(
        LOOMC_STATUS_INTERNAL,
        "launch-config compiler returned no artifact contents");
  }
  return status;
}

static loomc_status_t loomc_compile_specialize_functions(
    const loomc_target_environment_t* target_environment,
    const loomc_target_specialization_options_t* options, loom_module_t* module,
    loomc_result_t* result, iree_arena_allocator_t* arena,
    loom_function_version_owner_t* out_function_versions) {
  loom_function_version_owner_initialize(arena, out_function_versions);
  if (options == NULL || (options->specialization_count == 0 &&
                          options->target_binding_count == 0)) {
    return loomc_ok_status();
  }

  loom_target_specialization_request_list_t requests = {0};
  loom_target_declaration_binding_list_t bindings = {0};
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_make_lists(
      options, arena, &requests, &bindings));

  loomc_compile_diagnostic_capture_t capture = {
      .result = result,
      .emitter = LOOM_EMITTER_PASS,
  };
  loom_target_specialization_result_t specialization_result = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_target_specialize_functions(
      loomc_target_environment_loom_target_environment(target_environment),
      module, requests, bindings,
      (iree_diagnostic_emitter_t){
          .fn = loomc_compile_capture_diagnostic_emission,
          .user_data = &capture,
      },
      arena, &specialization_result)));
  *out_function_versions = specialization_result.function_versions;
  if (specialization_result.error_count != 0) {
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

  loomc_host_size_t identifier_length = 0;
  if (!iree_host_size_checked_add(module_name.size, file_extension.size,
                                  &identifier_length)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "artifact identifier length exceeds the host size domain");
  }
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
    const loom_target_launch_config_compiler_t* launch_config_compiler,
    const loom_target_emit_artifact_t* target_artifact) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  loomc_string_view_t identifier = loomc_string_view_empty();
  loomc_status_t status = loomc_compile_make_artifact_identifier(
      options,
      loomc_string_view_from_iree(launch_config_compiler->file_extension),
      loomc_string_view_from_iree(launch_config_compiler->default_identifier),
      allocator, &identifier);

  if (loomc_status_is_ok(status)) {
    const loomc_artifact_t artifact = {
        .kind = LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        .format = loomc_string_view_from_iree(
            launch_config_compiler->public_artifact_format),
        .identifier = identifier,
        .contents = loomc_byte_sequence_from_iree(target_artifact->contents),
    };
    status = loomc_result_add_artifact(result, &artifact);
  }

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
    const loomc_module_t* module,
    const loom_target_launch_config_compiler_t* launch_config_compiler,
    const loom_target_emit_artifact_t* launch_config_artifact) {
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
    status = loomc_compile_add_launch_config_artifact(
        result, options, launch_config_compiler, launch_config_artifact);
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

static loomc_status_t loomc_compiled_module_product_allocate(
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
  const loomc_target_pass_environment_t* target_pass_environment =
      loomc_context_target_pass_environment(compiler->context);
  iree_arena_allocator_t* function_version_arena =
      loomc_module_prepare_function_versions(module);
  loom_function_version_owner_t function_versions = {0};
  const bool launch_config_requested =
      options != NULL &&
      iree_any_bit_set(options->artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
  const loom_target_launch_config_compiler_t* launch_config_compiler = NULL;
  const loom_pass_environment_capability_t* launch_config_capability = NULL;
  iree_arena_allocator_t launch_config_arena;
  bool launch_config_arena_initialized = false;
  loom_target_emit_artifact_t launch_config_artifact = {0};
  loomc_config_application_result_t config_application = {0};

  loomc_status_t status =
      loomc_result_verify_loom_module(internal_module, /*source=*/NULL, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_module_t* config_module =
        options ? options->config_module : NULL;
    loomc_config_apply_module_options_t config_apply_options = {
        .config_module = loomc_module_const_loom_module(config_module),
        .target_module = internal_module,
        .policy_flags = options ? options->config_flags : 0,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
    };
    status =
        loomc_config_apply_module(&config_apply_options, &config_application);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_specialize_functions(
        context_target_environment, target_specialization, internal_module,
        result, function_version_arena, &function_versions);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_requested) {
    if (target_pass_environment == NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "launch configuration compilation requires a target environment");
    } else {
      launch_config_compiler = loom_target_environment_launch_config_compiler(
          target_pass_environment->target_environment);
      if (launch_config_compiler == NULL) {
        status = loomc_make_status(
            LOOMC_STATUS_FAILED_PRECONDITION,
            "target environment cannot compile launch configurations");
      }
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_requested) {
    iree_arena_initialize(loomc_workspace_block_pool(workspace),
                          &launch_config_arena);
    launch_config_arena_initialized = true;
    status = loomc_status_from_iree(launch_config_compiler->prepare(
        &launch_config_arena, &launch_config_capability));
    if (loomc_status_is_ok(status) && launch_config_capability == NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "launch-config compiler returned no pass capability");
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_run_pass_program(workspace, pass_program,
                                            internal_module, &function_versions,
                                            launch_config_capability, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      launch_config_requested) {
    status = loomc_compile_emit_launch_config_program(
        target_pass_environment, launch_config_compiler,
        launch_config_capability, &function_versions.list, internal_module,
        result, &launch_config_arena, &launch_config_artifact);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_module_publish_function_versions(module, function_versions);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_emit_requested_artifacts(
        result, options, target_specialization, &config_application, module,
        launch_config_compiler, &launch_config_artifact);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_prepare_function_versions(module);
  }
  loom_target_emit_artifact_release(&launch_config_artifact);
  if (launch_config_arena_initialized) {
    iree_arena_deinitialize(&launch_config_arena);
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
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
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

static loomc_status_t loomc_compile_validate_request_goals(
    const loomc_request_t* request) {
  const loomc_request_root_t* roots = loomc_request_roots(request);
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i].goal != LOOMC_REQUEST_ROOT_GOAL_DEFAULT) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "compiled module request root has an unsupported goal");
    }
  }
  return loomc_ok_status();
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
  LOOMC_RETURN_IF_ERROR(loomc_compile_validate_request_goals(request));
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
    status = loomc_compiled_module_product_allocate(
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
