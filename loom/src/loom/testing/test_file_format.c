// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_file_format.h"

#include <string.h>

#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/testing/test_diagnostic.h"
#include "loom/testing/test_file.h"
#include "loom/verify/verify.h"

typedef struct loom_test_file_format_diagnostic_collector_t {
  loom_test_diagnostic_t* diagnostics;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_arena_allocator_t* arena;
  iree_allocator_t host_allocator;
  loom_test_diagnostic_format_options_t format_options;
} loom_test_file_format_diagnostic_collector_t;

static iree_status_t loom_test_file_format_diagnostic_collector_grow(
    loom_test_file_format_diagnostic_collector_t* collector) {
  const iree_host_size_t new_capacity =
      collector->capacity == 0 ? 16 : collector->capacity * 2;
  loom_test_diagnostic_t* new_diagnostics = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(collector->arena, new_capacity,
                                                 sizeof(*new_diagnostics),
                                                 (void**)&new_diagnostics));
  if (collector->count > 0) {
    memcpy(new_diagnostics, collector->diagnostics,
           collector->count * sizeof(*new_diagnostics));
  }
  collector->diagnostics = new_diagnostics;
  collector->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_test_file_format_diagnostic_collector_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_test_file_format_diagnostic_collector_t* collector =
      (loom_test_file_format_diagnostic_collector_t*)user_data;
  if (collector->count >= collector->capacity) {
    IREE_RETURN_IF_ERROR(
        loom_test_file_format_diagnostic_collector_grow(collector));
  }
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_materialize(
      diagnostic, &collector->format_options, collector->arena,
      collector->host_allocator, &collector->diagnostics[collector->count]));
  ++collector->count;
  return iree_ok_status();
}

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

static loom_text_print_flags_t loom_test_file_format_case_print_flags(
    const loom_test_case_t* test_case) {
  loom_text_print_flags_t flags =
      LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_PREFER_LOW_ASM;
  if (iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_LOCATIONS)) {
    flags |= LOOM_TEXT_PRINT_LOCATIONS;
  }
  return flags;
}

typedef struct loom_test_file_format_input_slices_t {
  iree_string_view_t module_source;
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
    iree_host_size_t case_index,
    const loom_test_file_format_diagnostic_collector_t* collector,
    iree_allocator_t allocator) {
  iree_string_builder_t detail;
  iree_string_builder_initialize(allocator, &detail);
  iree_status_t detail_status = iree_string_builder_append_format(
      &detail, "while formatting case %zu in '%.*s'", case_index + 1,
      (int)filename.size, filename.data);
  for (iree_host_size_t i = 0;
       iree_status_is_ok(detail_status) && i < collector->count; ++i) {
    if (collector->diagnostics[i].matched) continue;
    detail_status = iree_string_builder_append_cstring(&detail, "\n");
    if (iree_status_is_ok(detail_status)) {
      detail_status = iree_string_builder_append_string(
          &detail, collector->diagnostics[i].formatted_diagnostic);
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

static iree_status_t loom_test_file_format_parse_module(
    iree_string_view_t source, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_text_low_asm_environment_t low_asm_environment,
    loom_module_t** out_module) {
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = diagnostic_sink,
      .low_asm_environment = low_asm_environment,
  };
  IREE_RETURN_IF_ERROR(loom_text_parse(source, filename, context, block_pool,
                                       &parse_options, out_module));
  if (*out_module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse test input '%.*s'",
                            (int)filename.size, filename.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_test_file_format_verify_module(
    iree_string_view_t source, iree_string_view_t filename,
    const loom_module_t* module, loom_diagnostic_sink_t diagnostic_sink) {
  const loom_source_entry_t source_entry = {
      .source_id = 0,
      .source = source,
      .filename = filename,
  };
  const loom_source_table_resolver_t source_table = {
      .entries = &source_entry,
      .count = 1,
  };
  const loom_verify_options_t verify_options = {
      .sink = diagnostic_sink,
      .source_resolver =
          {
              .fn = loom_source_table_resolve,
              .user_data = (void*)&source_table,
          },
      .max_errors = 100,
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  if (verify_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test input verification failed with %u error%s",
                            verify_result.error_count,
                            verify_result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static iree_status_t loom_test_file_format_case(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    iree_string_view_t filename, loom_context_t* context,
    iree_arena_block_pool_t* block_pool,
    loom_text_low_asm_environment_t low_asm_environment,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    iree_string_builder_t* output) {
  if (loom_test_file_format_case_preserves_source_spelling(test_case)) {
    return iree_string_builder_append_string(output, test_case->input);
  }
  const loom_test_file_format_input_slices_t input_slices =
      loom_test_file_format_split_input(test_case->input);
  if (iree_string_view_is_empty(input_slices.module_source)) {
    return iree_string_builder_append_string(output, test_case->input);
  }
  const loom_text_print_flags_t print_flags =
      loom_test_file_format_case_print_flags(test_case);
  loom_test_file_format_diagnostic_collector_t collector = {
      .arena = arena,
      .host_allocator = allocator,
      .format_options =
          {
              .text_print_options =
                  {
                      .flags = print_flags,
                      .low_asm_environment = low_asm_environment,
                  },
          },
  };
  const loom_diagnostic_sink_t diagnostic_sink = {
      .fn = loom_test_file_format_diagnostic_collector_sink,
      .user_data = &collector,
  };
  loom_module_t* module = NULL;
  iree_status_t status = loom_test_file_format_parse_module(
      input_slices.module_source, filename, context, block_pool,
      diagnostic_sink, low_asm_environment, &module);
  if (iree_status_is_ok(status) &&
      loom_test_file_format_case_verifies_input(test_case)) {
    collector.format_options.module = module;
    status = loom_test_file_format_verify_module(
        input_slices.module_source, filename, module, diagnostic_sink);
  }

  if (!iree_status_is_ok(status)) {
    iree_host_size_t* annotation_to_diagnostic = NULL;
    iree_status_t match_status = loom_test_diagnostics_match_annotations(
        collector.diagnostics, collector.count, test_case->annotations,
        test_case->annotation_count, arena, &annotation_to_diagnostic);
    bool has_error = false;
    bool all_diagnostics_matched = collector.count > 0;
    for (iree_host_size_t i = 0; i < collector.count; ++i) {
      has_error |= collector.diagnostics[i].severity == LOOM_DIAGNOSTIC_ERROR;
      all_diagnostics_matched &= collector.diagnostics[i].matched;
    }
    if (!iree_status_is_ok(match_status)) {
      status = iree_status_join(status, match_status);
    } else if (has_error && all_diagnostics_matched) {
      iree_status_free(status);
      status = iree_string_builder_append_string(output, test_case->input);
    } else {
      status = loom_test_file_format_annotate_failure(
          status, filename, case_index, &collector, allocator);
    }
  } else {
    iree_string_builder_t formatted;
    iree_string_builder_initialize(allocator, &formatted);
    const loom_text_print_options_t print_options = {
        .flags = print_flags,
        .low_asm_environment = low_asm_environment,
    };
    status = loom_text_print_module_to_builder_with_options(module, &formatted,
                                                            &print_options);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          output, iree_string_builder_view(&formatted));
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(output,
                                                 input_slices.trailing_trivia);
    }
    iree_string_builder_deinitialize(&formatted);
  }

  loom_module_free(module);
  return status;
}

iree_status_t loom_test_file_format(
    iree_string_view_t source, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_text_low_asm_environment_t low_asm_environment,
    iree_allocator_t allocator, iree_string_builder_t* out_source) {
  iree_string_builder_reset(out_source);

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_test_file_t test_file = {0};
  iree_status_t status = loom_test_file_parse(source, &arena, &test_file);

  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < test_file.case_count; ++i) {
    const loom_test_case_t* test_case = &test_file.cases[i];
    if (test_case->input_range.start_byte < cursor ||
        test_case->input_range.end_byte < test_case->input_range.start_byte ||
        test_case->input_range.end_byte > source.size) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "test input range is outside the source");
      break;
    }
    status = iree_string_builder_append_string(
        out_source,
        iree_string_view_substr(source, cursor,
                                test_case->input_range.start_byte - cursor));
    if (iree_status_is_ok(status)) {
      status = loom_test_file_format_case(test_case, i, filename, context,
                                          block_pool, low_asm_environment,
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
