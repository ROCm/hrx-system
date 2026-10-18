// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_file_format.h"

#include "loom/testing/test_diagnostic_loomc.h"
#include "loom/testing/test_file.h"
#include "loomc/iree.h"

static bool loom_test_file_format_case_preserves_source_spelling(
    const loom_test_case_t* test_case) {
  return test_case->has_expected_section &&
         (test_case->mode == LOOM_TEST_MODE_ROUNDTRIP ||
          test_case->mode == LOOM_TEST_MODE_FORMAT);
}

// Only verification cases establish the generic verifier as the operation
// under test. Passes and emitters may intentionally accept input that they
// legalize or diagnose themselves, while roundtrip only promises parse/print.
static bool loom_test_file_format_case_verifies_input(
    const loom_test_case_t* test_case) {
  return test_case->mode == LOOM_TEST_MODE_VERIFY;
}

typedef struct loom_test_file_format_input_slices_t {
  // Input through the last non-comment line.
  iree_string_view_t module_source;
  // Unattached comments and blank lines preserved by the container.
  iree_string_view_t trailing_trivia;
} loom_test_file_format_input_slices_t;

// Splits unattached trailing comments and blank lines from the module text.
// They belong to the test container rather than the module printer and must
// remain byte-for-byte stable around // ---- and // ==== separators.
static loom_test_file_format_input_slices_t loom_test_file_format_split_input(
    iree_string_view_t input) {
  iree_host_size_t offset = 0;
  iree_host_size_t module_end = 0;
  bool has_module_line = false;
  while (offset < input.size) {
    const iree_string_view_t remaining =
        iree_string_view_substr(input, offset, IREE_HOST_SIZE_MAX);
    const iree_host_size_t newline =
        iree_string_view_find_char(remaining, '\n', 0);
    const iree_host_size_t line_size =
        newline == IREE_STRING_VIEW_NPOS ? remaining.size : newline;
    const iree_string_view_t line =
        iree_string_view_substr(remaining, 0, line_size);
    const iree_string_view_t trimmed = iree_string_view_trim(line);
    const iree_host_size_t next_offset =
        offset + line_size + (newline == IREE_STRING_VIEW_NPOS ? 0 : 1);
    if (!iree_string_view_is_empty(trimmed) &&
        !iree_string_view_starts_with(trimmed, IREE_SV("//"))) {
      has_module_line = true;
      module_end = next_offset;
    }
    offset = next_offset;
  }
  if (!has_module_line) {
    return (loom_test_file_format_input_slices_t){
        .trailing_trivia = input,
    };
  }
  return (loom_test_file_format_input_slices_t){
      .module_source = iree_string_view_substr(input, 0, module_end),
      .trailing_trivia =
          iree_string_view_substr(input, module_end, IREE_HOST_SIZE_MAX),
  };
}

static iree_status_t loom_test_file_format_annotate_failure(
    iree_status_t status, iree_string_view_t filename,
    iree_host_size_t case_index, const loom_test_diagnostic_t* diagnostics,
    iree_host_size_t count, iree_allocator_t allocator) {
  iree_string_builder_t detail;
  iree_string_builder_initialize(allocator, &detail);
  iree_status_t detail_status = iree_string_builder_append_format(
      &detail, "while formatting case %zu in '%.*s'", case_index + 1,
      (int)filename.size, filename.data);
  for (iree_host_size_t i = 0; iree_status_is_ok(detail_status) && i < count;
       ++i) {
    if (diagnostics[i].matched) {
      continue;
    }
    detail_status = iree_string_builder_append_cstring(&detail, "\n");
    if (iree_status_is_ok(detail_status)) {
      detail_status = iree_string_builder_append_string(
          &detail, diagnostics[i].formatted_diagnostic);
    }
  }
  if (iree_status_is_ok(detail_status)) {
    const iree_string_view_t detail_view = iree_string_builder_view(&detail);
    status = iree_status_annotate_f(status, "%.*s", (int)detail_view.size,
                                    detail_view.data);
  } else {
    status = iree_status_join(status, detail_status);
  }
  iree_string_builder_deinitialize(&detail);
  return status;
}

// Only diagnosed input failures may be accepted by annotations. Allocation,
// output, and API failures retain their status even if earlier diagnostics
// match.
static iree_status_t loom_test_file_format_match_failure(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    iree_string_view_t filename, const loomc_result_t* parse_result,
    const loomc_result_t* verify_result, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, iree_string_builder_t* output) {
  const loomc_result_t* results[] = {parse_result, verify_result};
  const iree_host_size_t count = loomc_result_diagnostic_count(parse_result) +
                                 loomc_result_diagnostic_count(verify_result);
  loom_test_diagnostic_t* diagnostics = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*diagnostics), (void**)&diagnostics));
  iree_status_t status = iree_ok_status();
  iree_host_size_t ordinal = 0;
  for (iree_host_size_t result_index = 0;
       iree_status_is_ok(status) && result_index < IREE_ARRAYSIZE(results);
       ++result_index) {
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) &&
         i < loomc_result_diagnostic_count(results[result_index]);
         ++i) {
      status = loom_test_diagnostic_materialize_loomc(
          loomc_result_diagnostic_at(results[result_index], i), arena,
          &diagnostics[ordinal++]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  iree_host_size_t* annotation_to_diagnostic = NULL;
  IREE_RETURN_IF_ERROR(loom_test_diagnostics_match_annotations(
      diagnostics, count, test_case->annotations, test_case->annotation_count,
      filename, arena, &annotation_to_diagnostic));
  bool has_error = false;
  bool all_diagnostics_matched = count > 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    has_error |= diagnostics[i].severity == LOOM_DIAGNOSTIC_ERROR;
    all_diagnostics_matched &= diagnostics[i].matched;
  }
  if (has_error && all_diagnostics_matched) {
    return iree_string_builder_append_string(output, test_case->input);
  }
  return loom_test_file_format_annotate_failure(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid test input"),
      filename, case_index, diagnostics, count, allocator);
}

static iree_status_t loom_test_file_format_case(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    iree_string_view_t filename, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, iree_string_builder_t* output) {
  if (loom_test_file_format_case_preserves_source_spelling(test_case)) {
    return iree_string_builder_append_string(output, test_case->input);
  }
  const loom_test_file_format_input_slices_t input_slices =
      loom_test_file_format_split_input(test_case->input);
  if (iree_string_view_is_empty(input_slices.module_source)) {
    return iree_string_builder_append_string(output, test_case->input);
  }
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_string_view_from_iree(filename),
      .contents =
          loomc_make_byte_span(test_case->input.data, test_case->input.size),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_source_create(&source_options, compiler_allocator, &source)));
  loomc_module_t* module = NULL;
  loomc_result_t* parse_result = NULL;
  loomc_result_t* verify_result = NULL;
  // Parse the full case so EOF diagnostics retain their original locations.
  // The parser leaves final trivia unattached for verbatim output.
  iree_status_t status =
      iree_status_from_loomc(loomc_module_deserialize_from_source(
          context, workspace, source, NULL, compiler_allocator, &module,
          &parse_result));
  loomc_source_release(source);
  if (iree_status_is_ok(status) && loomc_result_succeeded(parse_result) &&
      loom_test_file_format_case_verifies_input(test_case)) {
    const loomc_module_verify_options_t verify_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
        .structure_size = sizeof(verify_options),
        .max_errors = 100,
        .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
    };
    status = iree_status_from_loomc(loomc_module_verify(
        module, &verify_options, compiler_allocator, &verify_result));
  }

  if (iree_status_is_ok(status) &&
      (!loomc_result_succeeded(parse_result) ||
       (verify_result && !loomc_result_succeeded(verify_result)))) {
    status = loom_test_file_format_match_failure(
        test_case, case_index, filename, parse_result, verify_result, arena,
        allocator, output);
  } else if (iree_status_is_ok(status)) {
    const loomc_module_serialize_options_t print_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        .structure_size = sizeof(print_options),
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .print_locations = iree_all_bits_set(test_case->output_flags,
                                             LOOM_TEST_OUTPUT_LOCATIONS),
    };
    loomc_source_t* formatted = NULL;
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &print_options, compiler_allocator, &formatted));
    if (iree_status_is_ok(status)) {
      const loomc_byte_span_t contents = loomc_source_contents(formatted);
      status = iree_string_builder_append_string(
          output, iree_make_string_view((const char*)contents.data,
                                        contents.data_length));
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(output,
                                                 input_slices.trailing_trivia);
    }
    loomc_source_release(formatted);
  }

  loomc_result_release(verify_result);
  loomc_result_release(parse_result);
  loomc_module_release(module);
  return status;
}

iree_status_t loom_test_file_format(iree_string_view_t source,
                                    iree_string_view_t filename,
                                    loomc_context_t* context,
                                    loomc_workspace_t* workspace,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_allocator_t allocator,
                                    iree_string_builder_t* out_source) {
  iree_string_builder_reset(out_source);

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_test_file_t test_file = {0};
  iree_status_t status = loom_test_file_parse(source, &arena, &test_file);

  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < test_file.case_count; ++i) {
    const loom_test_case_t* test_case = &test_file.cases[i];
    status = iree_string_builder_append_string(
        out_source,
        iree_string_view_substr(source, cursor,
                                test_case->input_range.start_byte - cursor));
    if (iree_status_is_ok(status)) {
      status =
          loom_test_file_format_case(test_case, i, filename, context, workspace,
                                     &arena, allocator, out_source);
    }
    cursor = test_case->input_range.end_byte;
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        out_source,
        iree_string_view_substr(source, cursor, IREE_HOST_SIZE_MAX));
  }

  iree_arena_deinitialize(&arena);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_reset(out_source);
  }
  return status;
}
