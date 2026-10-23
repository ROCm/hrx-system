// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/module_format.h"
#include "loom/tools/loom-check/comparison.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/iree.h"

static iree_status_t loom_check_roundtrip_append_diagnostics(
    const loom_check_diagnostic_collector_t* collector,
    loom_check_result_t* result) {
  for (iree_host_size_t i = 0; i < collector->count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &result->detail, collector->diagnostics[i].formatted_diagnostic));
  }
  return iree_ok_status();
}

iree_status_t loom_check_validate_printed_ir(
    iree_string_view_t source, loom_check_compile_session_t* session,
    iree_arena_block_pool_t* block_pool,
    const loomc_module_serialize_options_t* print_options,
    loom_check_result_t* result, bool* out_valid) {
  *out_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_compile_session_prepare(session));
  const iree_allocator_t allocator = result->detail.allocator;
  const loomc_allocator_t public_allocator =
      loomc_allocator_from_iree(allocator);
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = IREE_SV("<printed IR>"),
      .result = result,
  };
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("<printed IR>"),
      .contents = loomc_make_byte_span(source.data, source.size),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* input = NULL;
  iree_status_t status = iree_status_from_loomc(
      loomc_source_create(&source_options, public_allocator, &input));
  loomc_module_t* module = NULL;
  loomc_result_t* decoding = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        session->context, session->workspace, input, /*options=*/NULL,
        public_allocator, &module, &decoding));
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result_diagnostics(&collector, decoding);
  }
  loomc_result_release(decoding);
  loomc_source_release(input);
  if (iree_status_is_ok(status) && module == NULL) {
    status = iree_string_builder_append_cstring(&result->detail,
                                                "printed IR failed to parse\n");
  }

  loomc_source_t* reprinted = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = iree_status_from_loomc(loomc_module_serialize_text_to_source(
        module, print_options, public_allocator, &reprinted));
  }
  loomc_module_release(module);
  if (iree_status_is_ok(status) && reprinted != NULL) {
    const loomc_byte_span_t contents = loomc_source_contents(reprinted);
    const iree_string_view_t canonical =
        iree_make_string_view((const char*)contents.data, contents.data_length);
    *out_valid = iree_string_view_equal(source, canonical);
    if (!*out_valid) {
      status = iree_string_builder_append_cstring(
          &result->detail, "printed IR changed after reparsing\n");
      if (iree_status_is_ok(status)) {
        status =
            loom_check_result_record_diff(source, canonical, allocator, result);
      }
    }
  }
  loomc_source_release(reprinted);
  const iree_status_t diagnostic_status =
      loom_check_roundtrip_append_diagnostics(&collector, result);
  iree_arena_deinitialize(&diagnostic_arena);
  return iree_status_join(status, diagnostic_status);
}

iree_status_t loom_check_execute_roundtrip(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };
  loomc_module_t* module = NULL;
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, environment->compile_session,
      environment, /*max_errors=*/20, &collector, block_pool, allocator,
      &module);
  const loomc_module_serialize_options_t print_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(print_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .print_locations = iree_all_bits_set(test_case->output_flags,
                                           LOOM_TEST_OUTPUT_LOCATIONS),
  };
  loomc_source_t* text = NULL;
  if (iree_status_is_ok(status) && module != NULL) {
    status = iree_status_from_loomc(loomc_module_serialize_text_to_source(
        module, &print_options, loomc_allocator_from_iree(allocator), &text));
  }
  loomc_module_release(module);
  if (iree_status_is_ok(status) && text != NULL) {
    const loomc_byte_span_t contents = loomc_source_contents(text);
    status = iree_string_builder_append_string(
        &result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
  }
  const bool printed = text != NULL;
  loomc_source_release(text);
  const iree_status_t diagnostic_status =
      loom_check_roundtrip_append_diagnostics(&collector, result);
  iree_arena_deinitialize(&diagnostic_arena);
  IREE_RETURN_IF_ERROR(iree_status_join(status, diagnostic_status));
  if (!printed) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }
  bool valid_output = false;
  IREE_RETURN_IF_ERROR(loom_check_validate_printed_ir(
      iree_string_builder_view(&result->actual_output),
      environment->compile_session, block_pool, &print_options, result,
      &valid_output));
  if (!valid_output) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }
  result->has_actual_output = true;

  return loom_check_compare_output(test_case, allocator, result);
}

static iree_status_t loom_check_format_verify_module(
    loomc_module_t* module, loom_check_diagnostic_collector_t* collector,
    uint32_t max_errors, iree_allocator_t allocator) {
  const loomc_module_verify_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
      .structure_size = sizeof(options),
      .max_errors = max_errors,
      .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
  };
  loomc_result_t* verification = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_verify(
      module, &options, loomc_allocator_from_iree(allocator), &verification));
  if (iree_status_is_ok(status)) {
    status =
        loom_check_compile_append_result_diagnostics(collector, verification);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(verification)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "format module failed verification");
  }
  loomc_result_release(verification);
  return status;
}

iree_status_t loom_check_execute_format(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_AUTO;
  IREE_RETURN_IF_ERROR(loom_module_format_parse(test_case->format_target,
                                                /*allow_auto=*/false, &format));
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };
  loom_check_compile_session_t* session = environment->compile_session;
  loomc_module_t* module = NULL;
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, session, environment,
      /*max_errors=*/0, &collector, block_pool, allocator, &module);
  if (iree_status_is_ok(status) && module == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input failed to produce a module");
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_format_verify_module(module, &collector,
                                             /*max_errors=*/0, allocator);
  }

  const loomc_allocator_t public_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .format = format == LOOM_MODULE_FORMAT_TEXT
                    ? LOOMC_SOURCE_FORMAT_TEXT
                    : LOOMC_SOURCE_FORMAT_BYTECODE,
      .identifier = loomc_string_view_from_iree(filename),
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_PRESERVE,
  };
  loomc_source_t* serialized = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &options, public_allocator, &serialized));
  }
  loomc_module_release(module);
  module = NULL;

  loomc_result_t* decoding = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        session->context, session->workspace, serialized, /*options=*/NULL,
        public_allocator, &module, &decoding));
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result_diagnostics(&collector, decoding);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(decoding)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "formatted output failed to produce a module");
  }
  loomc_result_release(decoding);
  loomc_source_release(serialized);
  if (iree_status_is_ok(status)) {
    status = loom_check_format_verify_module(module, &collector,
                                             /*max_errors=*/100, allocator);
  }

  loomc_source_t* text = NULL;
  if (iree_status_is_ok(status)) {
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &options, public_allocator, &text));
  }
  loomc_module_release(module);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(text);
    status = iree_string_builder_append_string(
        &result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
  }
  loomc_source_release(text);

  const iree_status_t diagnostic_status =
      loom_check_roundtrip_append_diagnostics(&collector, result);
  iree_arena_deinitialize(&diagnostic_arena);
  IREE_RETURN_IF_ERROR(iree_status_join(status, diagnostic_status));
  result->has_actual_output = true;
  return loom_check_compare_output(test_case, allocator, result);
}
