// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/export/cxx/loom_check.h"

#include <limits.h>
#include <stdlib.h>

#include "loom/target/tool/process.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/requirements.h"
#include "loomc/iree.h"
#include "loomc/module.h"

typedef struct loom_cxx_export_options_t {
  // Optional callable root forwarded to loom-export-cxx.
  iree_string_view_t root;
  // Whether to import and verify the rendered C++ before comparison.
  bool roundtrip;
} loom_cxx_export_options_t;

static iree_status_t loom_cxx_export_options_parse(
    iree_string_view_t source, loom_cxx_export_options_t* out_options) {
  *out_options = (loom_cxx_export_options_t){0};
  iree_string_view_t remaining = iree_string_view_trim(source);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t option = iree_string_view_empty();
    iree_string_view_split(remaining, ' ', &option, &remaining);
    option = iree_string_view_trim(option);
    remaining = iree_string_view_trim(remaining);
    if (iree_string_view_equal(option, IREE_SV("roundtrip"))) {
      if (out_options->roundtrip) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "cxx export option 'roundtrip' is repeated");
      }
      out_options->roundtrip = true;
      continue;
    }
    if (iree_string_view_starts_with(option, IREE_SV("@"))) {
      if (!iree_string_view_is_empty(out_options->root)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "cxx export accepts at most one callable root");
      }
      if (option.size == 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "cxx export callable root cannot be empty");
      }
      out_options->root = option;
      continue;
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unknown cxx export option '%.*s'; expected 'roundtrip' or '@symbol'",
        (int)option.size, option.data);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_cxx_export_tool_path(bool* out_search_path) {
  const char* configured_path = getenv("LOOM_EXPORT_CXX");
  if (configured_path != NULL && configured_path[0] != '\0') {
    *out_search_path = false;
    return iree_make_cstring_view(configured_path);
  }
  *out_search_path = true;
  return IREE_SV("loom-export-cxx" IREE_PLATFORM_EXECUTABLE_SUFFIX);
}

static iree_status_t loom_cxx_export_run_tool(
    const iree_string_view_t* arguments, iree_host_size_t argument_count,
    iree_allocator_t allocator, loom_tool_process_result_t* out_result) {
  bool search_path = false;
  const iree_string_view_t executable_path =
      loom_cxx_export_tool_path(&search_path);
  return loom_tool_process_run(executable_path, search_path, arguments,
                               argument_count, allocator, out_result);
}

static iree_status_t loom_cxx_export_tool_failure_status(
    const loom_tool_process_result_t* result, iree_string_view_t action) {
  const int stdout_length =
      (int)iree_min(result->stdout_bytes.length, (iree_host_size_t)INT_MAX);
  const int stderr_length =
      (int)iree_min(result->stderr_bytes.length, (iree_host_size_t)INT_MAX);
  const int action_length =
      (int)iree_min(action.size, (iree_host_size_t)INT_MAX);
  const char* stdout_data =
      result->stdout_bytes.data ? result->stdout_bytes.data : "";
  const char* stderr_data =
      result->stderr_bytes.data ? result->stderr_bytes.data : "";
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "loom-export-cxx failed while %.*s with exit code %d\nstdout:\n%.*s\n"
      "stderr:\n%.*s",
      action_length, action.data, result->exit_code, stdout_length, stdout_data,
      stderr_length, stderr_data);
}

static bool loom_cxx_export_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("cxx"));
}

static iree_status_t loom_cxx_export_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_test_case_t* test_case, iree_string_view_t target_options,
    loom_check_result_t* result, bool* out_continue_execution) {
  (void)provider;
  IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
      test_case, IREE_SV("loom-export-cxx"), result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }
  loom_cxx_export_options_t options;
  return loom_cxx_export_options_parse(target_options, &options);
}

static iree_status_t loom_cxx_export_validate_roundtrip(
    const loom_check_emit_provider_request_t* request,
    const loom_tool_output_t* output) {
  loom_check_compile_session_t* session = request->environment->compile_session;
  if (session == NULL || session->provider == NULL ||
      session->provider->import == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cxx export roundtrip requires a configured C++ importer");
  }
  IREE_RETURN_IF_ERROR(loom_check_compile_session_prepare(session));

  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(request->host_allocator);
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
      .identifier = loomc_make_cstring_view("<loom-export-cxx>.cxx"),
      .contents = loomc_make_byte_span(output->data, output->length),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = NULL;
  iree_status_t status = iree_status_from_loomc(
      loomc_source_create(&source_options, allocator, &source));

  loomc_module_t* module = NULL;
  loomc_result_t* import_result = NULL;
  if (iree_status_is_ok(status)) {
    const loom_tooling_source_path_options_t source_path_options = {0};
    status = session->provider->import(
        session->provider->import_user_data, IREE_SV("cxx"),
        iree_string_view_empty(), &source_path_options, session->context,
        session->workspace, source, request->block_pool,
        request->host_allocator, &module, &import_result);
  }
  bool imported = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              import_result, &imported);
  }
  loomc_result_release(import_result);
  loomc_source_release(source);

  loomc_result_t* verification_result = NULL;
  if (iree_status_is_ok(status) && imported) {
    status = iree_status_from_loomc(loomc_module_verify(
        module, /*options=*/NULL, allocator, &verification_result));
  }
  bool verified = false;
  if (iree_status_is_ok(status) && imported) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              verification_result, &verified);
  }
  loomc_result_release(verification_result);
  loomc_module_release(module);
  return status;
}

static iree_status_t loom_cxx_export_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_cxx_export_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_cxx_export_options_parse(request->target_options, &options));
  loom_tool_temp_file_t input_file = {0};
  iree_status_t status =
      loom_tool_temp_file_initialize(IREE_SV("loom-export-cxx"), &input_file);

  if (iree_status_is_ok(status)) {
    const loomc_module_serialize_options_t serialize_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        .structure_size = sizeof(serialize_options),
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
    };
    status = iree_status_from_loomc(loomc_module_serialize_to_path(
        request->public_module, &serialize_options,
        loomc_string_view_from_iree(loom_tool_temp_file_path(&input_file)),
        loomc_allocator_from_iree(request->host_allocator)));
  }

  loom_tool_process_result_t result = {0};
  if (iree_status_is_ok(status)) {
    iree_string_view_t arguments[4] = {loom_tool_temp_file_path(&input_file),
                                       IREE_SV("--output=-")};
    iree_host_size_t argument_count = 2;
    if (!iree_string_view_is_empty(options.root)) {
      arguments[argument_count++] = IREE_SV("--root");
      arguments[argument_count++] = options.root;
    }
    status = loom_cxx_export_run_tool(arguments, argument_count,
                                      request->host_allocator, &result);
  }
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_cxx_export_tool_failure_status(
        &result, IREE_SV("exporting the admitted module"));
  }
  if (iree_status_is_ok(status)) {
    loom_tool_output_normalize_newlines(&result.stdout_bytes);
    if (options.roundtrip) {
      status =
          loom_cxx_export_validate_roundtrip(request, &result.stdout_bytes);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &request->result->actual_output,
        iree_make_string_view(result.stdout_bytes.data,
                              result.stdout_bytes.length));
  }

  loom_tool_process_result_deinitialize(&result, request->host_allocator);
  status =
      iree_status_join(status, loom_tool_temp_file_deinitialize(&input_file));
  return status;
}

static iree_status_t loom_cxx_export_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "cxx");
}

static bool loom_cxx_export_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement, IREE_SV("loom-export-cxx"));
}

static iree_status_t loom_cxx_export_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  (void)provider;
  (void)environment;
  if (!iree_string_view_equal(requirement, IREE_SV("loom-export-cxx"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown C++ export loom-check requirement '%.*s'",
                            (int)requirement.size, requirement.data);
  }
  const iree_string_view_t arguments[] = {IREE_SV("--help")};
  loom_tool_process_result_t result = {0};
  iree_status_t status = loom_cxx_export_run_tool(
      arguments, IREE_ARRAYSIZE(arguments), allocator, &result);
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_cxx_export_tool_failure_status(
        &result, IREE_SV("querying availability"));
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

static iree_status_t loom_cxx_export_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "loom-export-cxx");
}

static const loom_check_emit_provider_t kLoomCxxExportEmitProvider = {
    .name = IREE_SVL("cxx-export"),
    .match = loom_cxx_export_emit_provider_matches,
    .check_requirements = loom_cxx_export_emit_provider_check_requirements,
    .execute = loom_cxx_export_emit_provider_execute,
    .append_names = loom_cxx_export_emit_provider_append_names,
};

static const loom_check_requirement_provider_t
    kLoomCxxExportRequirementProvider = {
        .name = IREE_SVL("cxx-export"),
        .match = loom_cxx_export_requirement_provider_matches,
        .query = loom_cxx_export_requirement_provider_query,
        .append_names = loom_cxx_export_requirement_provider_append_names,
};

static const loom_check_emit_provider_t* const kLoomCxxExportEmitProviders[] = {
    &kLoomCxxExportEmitProvider,
};

static const loom_check_requirement_provider_t* const
    kLoomCxxExportRequirementProviders[] = {
        &kLoomCxxExportRequirementProvider,
};

const loom_check_provider_t loom_cxx_export_check_provider = {
    .name = IREE_SVL("cxx-export"),
    .emit_providers = kLoomCxxExportEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomCxxExportEmitProviders),
    .requirement_providers = kLoomCxxExportRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomCxxExportRequirementProviders),
};
