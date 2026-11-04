// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"
#include "loom/target/provider.h"
#include "loom/tools/loom-check/comparison.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"

static void loom_check_emit_split_target(
    iree_string_view_t emit_target, iree_string_view_t* out_target_name,
    iree_string_view_t* out_target_options) {
  emit_target = iree_string_view_trim(emit_target);
  iree_string_view_split(emit_target, ' ', out_target_name, out_target_options);
  *out_target_name = iree_string_view_trim(*out_target_name);
  *out_target_options = iree_string_view_trim(*out_target_options);
}

static iree_status_t loom_check_emit_append_supported_target_names(
    const loom_check_environment_t* environment,
    iree_string_builder_t* output) {
  if (environment == NULL || environment->emit_providers.providers == NULL) {
    return iree_ok_status();
  }
  bool has_names = false;
  for (iree_host_size_t i = 0; i < environment->emit_providers.provider_count;
       ++i) {
    const loom_check_emit_provider_t* provider =
        environment->emit_providers.providers[i];
    if (provider == NULL || provider->append_names == NULL) {
      continue;
    }
    if (has_names) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(provider->append_names(provider, output));
    has_names = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_fail_unknown_target(
    const loom_check_environment_t* environment, iree_string_view_t target_name,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &result->detail,
      "unknown emit target '%.*s'; supported emit targets are ",
      (int)target_name.size, target_name.data));
  IREE_RETURN_IF_ERROR(loom_check_emit_append_supported_target_names(
      environment, &result->detail));
  return iree_string_builder_append_cstring(&result->detail, "\n");
}

static iree_status_t loom_check_emit_finish_status_failure(
    iree_status_t failure_status, iree_string_view_t emit_target_name,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  iree_status_t status = iree_string_builder_append_format(
      &result->detail,
      "emit target '%.*s' failed: ", (int)emit_target_name.size,
      emit_target_name.data);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_status(&result->detail, failure_status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, "\n");
  }
  iree_status_free(failure_status);
  return status;
}

static iree_status_t loom_check_emit_finish_diagnostics_and_compare_output(
    loom_check_diagnostic_collector_t* collector,
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result));
  if (result->raw_outcome != LOOM_CHECK_PASS || !result->has_actual_output) {
    return iree_ok_status();
  }
  return loom_check_compare_output(test_case, allocator, result);
}

typedef struct loom_check_emit_native_provider_consumer_t {
  // Provider receiving the projected native module.
  const loom_check_emit_provider_t* provider;
  // Common request owning the admitted public module.
  const loom_check_emit_provider_request_t* request;
} loom_check_emit_native_provider_consumer_t;

static iree_status_t loom_check_emit_invoke_native_provider(
    void* user_data, const loom_check_emit_native_module_t* native_module) {
  const loom_check_emit_native_provider_consumer_t* consumer =
      (const loom_check_emit_native_provider_consumer_t*)user_data;
  return consumer->provider->execute_native(consumer->provider,
                                            consumer->request, native_module);
}

static iree_status_t loom_check_emit_invoke_provider(
    const loom_check_emit_provider_t* provider,
    loom_check_emit_provider_request_t* request) {
  iree_arena_allocator_t case_arena;
  iree_arena_initialize(request->block_pool, &case_arena);
  request->case_arena = &case_arena;
  // Providers own comparable output by default and may explicitly suppress it
  // for diagnostics-only modes.
  request->result->has_actual_output = true;
  iree_status_t status = iree_ok_status();
  if (provider->execute != NULL) {
    status = provider->execute(provider, request);
  } else {
    const loom_check_emit_native_provider_consumer_t native_consumer = {
        .provider = provider,
        .request = request,
    };
    status = loom_check_compile_with_native_module(
        request, loom_check_emit_invoke_native_provider,
        (void*)&native_consumer);
  }
  iree_arena_deinitialize(&case_arena);
  // Successful emission owns a comparable output even when it is empty.
  // Providers must opt in when they produce complete output after a compiler
  // error; partial output remains incomparable by default.
  const bool has_error =
      loom_check_diagnostic_collector_has_error(request->diagnostic_collector);
  const bool has_comparable_error_output =
      request->result->actual_output.size != 0 &&
      iree_any_bit_set(provider->flags,
                       LOOM_CHECK_EMIT_PROVIDER_FLAG_COMPARE_ERROR_OUTPUT);
  request->result->has_actual_output =
      iree_status_is_ok(status) && request->result->has_actual_output &&
      (!has_error || has_comparable_error_output);
  return status;
}

static iree_status_t loom_check_emit_finish_provider(
    iree_status_t status, const loom_check_emit_provider_request_t* request,
    iree_host_size_t case_index, loom_check_file_report_t* report) {
  if (!iree_status_is_ok(status)) {
    return loom_check_emit_finish_status_failure(status, request->target_name,
                                                 request->result);
  }
  return loom_check_emit_finish_diagnostics_and_compare_output(
      request->diagnostic_collector, request->test_case, case_index, report,
      request->host_allocator, request->result);
}

static void loom_check_emit_release_public_module(
    loom_check_emit_provider_request_t* request) {
  loomc_module_release(request->public_module);
  request->public_module = NULL;
  loom_check_compile_session_t* session = request->environment->compile_session;
  if (session->workspace != NULL) {
    loomc_workspace_trim(session->workspace);
  }
}

iree_status_t loom_check_execute_emit(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };

  iree_string_view_t provider_target_name = iree_string_view_empty();
  iree_string_view_t provider_target_options = iree_string_view_empty();
  loom_check_emit_split_target(test_case->emit_target, &provider_target_name,
                               &provider_target_options);
  const loom_check_emit_provider_t* provider =
      loom_check_environment_lookup_emit_provider(environment,
                                                  provider_target_name);
  iree_status_t status = iree_ok_status();
  if (provider == NULL) {
    status = loom_check_emit_fail_unknown_target(environment,
                                                 provider_target_name, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  const loom_target_low_descriptor_registry_t low_registry =
      loom_target_environment_low_descriptor_registry(
          environment->target_environment);
  loom_low_descriptor_text_print_context_initialize(
      &low_registry.registry, &diagnostic_collector.type_print_context);
  if ((provider->execute == NULL) == (provider->execute_native == NULL)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "emit provider '%.*s' must define exactly one execute callback",
        (int)provider->name.size, provider->name.data);
    status = loom_check_emit_finish_status_failure(status, provider_target_name,
                                                   result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  loom_check_emit_provider_request_t provider_request = {
      .emit_target = test_case->emit_target,
      .target_name = provider_target_name,
      .target_options = provider_target_options,
      .filename = filename,
      .test_case = test_case,
      .environment = environment,
      .low_registry = &low_registry,
      .diagnostic_collector = &diagnostic_collector,
      .block_pool = block_pool,
      .host_allocator = allocator,
      .result = result,
  };
  status = loom_check_compile_admit_module(
      test_case, filename, input_request, environment->compile_session,
      environment, /*max_errors=*/0, &diagnostic_collector, block_pool,
      allocator, &provider_request.public_module);
  if (!iree_status_is_ok(status)) {
    loom_check_emit_release_public_module(&provider_request);
    status = loom_check_emit_finish_status_failure(status, provider_target_name,
                                                   result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (provider_request.public_module == NULL ||
      loom_check_diagnostic_collector_has_error(&diagnostic_collector)) {
    loom_check_emit_release_public_module(&provider_request);
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  status = loom_check_emit_invoke_provider(provider, &provider_request);
  loom_check_emit_release_public_module(&provider_request);
  status = loom_check_emit_finish_provider(status, &provider_request,
                                           case_index, report);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}
