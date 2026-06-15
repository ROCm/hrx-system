// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/tools/loom-check/execute.h"

iree_status_t loom_check_execute_roundtrip(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  // Strip standalone comment lines from input. Comments become blank
  // lines to preserve line count for diagnostic source locations.
  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  IREE_RETURN_IF_ERROR(
      loom_check_strip_comments(test_case->input, &stripped_input));

  // Parse the stripped input.
  loom_module_t* module = NULL;
  loom_check_diagnostic_capture_t diagnostic_capture = {
      .detail = &result->detail,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .max_errors = 20,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  iree_status_t registry_status =
      loom_check_environment_initialize_low_descriptor_registry(environment,
                                                                &low_registry);
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  if (iree_status_is_ok(registry_status)) {
    loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
        &low_registry.registry, environment->low_asm_diagnostic_provider_list,
        &low_asm_storage, &parse_options.low_asm_environment);
  }
  iree_status_t parse_status =
      iree_status_is_ok(registry_status)
          ? loom_text_parse(iree_string_builder_view(&stripped_input), filename,
                            context, block_pool, &parse_options, &module)
          : registry_status;
  iree_string_builder_deinitialize(&stripped_input);
  IREE_RETURN_IF_ERROR(parse_status);
  if (!module) {
    // Parse errors are content failures, not infrastructure failures.
    // Diagnostics are already in result->detail from the sink.
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }

  // Print the parsed module to canonical text (directly into the result's
  // actual_output so --update can use it) and free the module.
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_initialize(&low_registry.registry,
                                                      &low_asm_environment);
  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = low_asm_environment,
  };
  iree_status_t print_status = loom_text_print_module_to_builder_with_options(
      module, &result->actual_output, &print_options);
  loom_module_free(module);
  IREE_RETURN_IF_ERROR(print_status);
  result->has_actual_output = true;

  // Strip comments from the expected section for comparison. When no
  // // ---- separator is present, expected == input and stripping
  // normalizes both sides identically.
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  IREE_RETURN_IF_ERROR(
      loom_check_strip_comments(test_case->expected, &stripped_expected));

  // Compare printed output against expected (trimmed to ignore trailing
  // whitespace differences).
  iree_string_view_t actual_trimmed =
      iree_string_view_trim(iree_string_builder_view(&result->actual_output));
  iree_string_view_t expected_trimmed =
      iree_string_view_trim(iree_string_builder_view(&stripped_expected));

  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
    result->raw_outcome = LOOM_CHECK_PASS;
  } else {
    result->raw_outcome = LOOM_CHECK_FAIL;
    status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                           allocator, result);
  }

  iree_string_builder_deinitialize(&stripped_expected);
  return status;
}
