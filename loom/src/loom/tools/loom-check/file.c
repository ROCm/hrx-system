// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/file.h"

#include <stdio.h>

#include "iree/io/file_contents.h"
#include "loom/testing/test_file.h"
#include "loom/tooling/input/format.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-check/json_output.h"
#include "loom/tools/loom-check/output.h"
#include "loom/tools/loom-check/update.h"
#include "loom/util/stream.h"

static iree_status_t loom_check_write_source(iree_string_view_t path,
                                             iree_string_view_t source,
                                             iree_allocator_t allocator) {
  return loom_tooling_write_output_file(path, source, allocator);
}

static iree_status_t loom_check_write_updates(
    iree_string_view_t path, iree_string_view_t original_source,
    const loom_test_file_t* file, const loom_check_case_update_t* updates,
    iree_allocator_t allocator) {
  iree_string_builder_t new_source;
  iree_string_builder_initialize(allocator, &new_source);

  iree_host_size_t update_count = 0;
  iree_status_t status = loom_check_apply_updates(
      original_source, file, updates, &new_source, &update_count);

  if (iree_status_is_ok(status) && update_count > 0) {
    status = loom_check_write_source(
        path, iree_string_builder_view(&new_source), allocator);
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "updated %zu case%s in %.*s\n", update_count,
              update_count == 1 ? "" : "s", (int)path.size, path.data);
    }
  }

  iree_string_builder_deinitialize(&new_source);
  return status;
}

static iree_status_t loom_check_process_file(
    iree_string_view_t path, iree_string_view_t filename,
    iree_string_view_t source, bool is_stdin,
    const loom_check_process_options_t* options,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, iree_host_size_t* pass_count,
    iree_host_size_t* fail_count, iree_host_size_t* skip_count) {
  loom_test_file_t file = {0};
  const bool compile = options->compile.target_profile != NULL;
  const bool update = options->mode == LOOM_CHECK_PROCESS_UPDATE;
  // Binary modules cannot contain case directives. Textual provider selection
  // happens after splitting so each case's INPUT can override the filename.
  const bool bytecode =
      iree_string_view_is_empty(options->input_format)
          ? iree_string_view_ends_with(path, IREE_SV(".loombc"))
          : iree_string_view_equal(options->input_format,
                                   loom_input_bytecode_format.name);
  iree_status_t status = iree_ok_status();
  loom_test_case_t bytecode_case = {.input = source};
  if (bytecode) {
    if (!compile) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "bytecode qualification requires --target");
    } else {
      file.cases = &bytecode_case;
      file.case_count = 1;
    }
  } else {
    status = loom_test_file_parse(source, arena, &file);
  }

  if (iree_status_is_ok(status) && update && is_stdin) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--update cannot be used with stdin");
  }
  if (iree_status_is_ok(status) && compile && update) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--target cannot be combined with --update");
  }

  if (iree_status_is_ok(status) && compile) {
    if (file.case_count == 0) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "compiler qualification requires at least one case");
    }
    file.default_mode = LOOM_TEST_MODE_COMPILE;
    for (iree_host_size_t i = 0; i < file.case_count; ++i) {
      file.cases[i].mode = LOOM_TEST_MODE_COMPILE;
      file.cases[i].xfail = false;
    }
  }

  loom_check_file_report_t report = {0};
  if (iree_status_is_ok(status)) {
    status = loom_check_file_report_initialize(&file, arena, &report);
  }

  loom_check_case_update_t* updates = NULL;
  if (iree_status_is_ok(status) && update && file.case_count > 0) {
    status = iree_allocator_malloc_array(allocator, file.case_count,
                                         sizeof(*updates), (void**)&updates);
  }

  loom_check_result_t* results = NULL;
  if (iree_status_is_ok(status) && file.case_count > 0) {
    status = iree_allocator_malloc_array(allocator, file.case_count,
                                         sizeof(*results), (void**)&results);
  }

  iree_host_size_t initialized_result_count = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < file.case_count;
       ++i) {
    const loom_test_case_t* test_case = &file.cases[i];

    loom_check_result_initialize(allocator, &results[i]);
    ++initialized_result_count;
    const loom_check_input_t input_request = {
        .path = is_stdin ? IREE_SV("<stdin>") : path,
        .format = options->input_format,
        .source_path_options = options->source_path_options,
    };
    if (compile) {
      status = loom_check_execute_compile(
          test_case, i, &report, filename, &input_request, &options->compile,
          environment, block_pool, allocator, &results[i]);
    } else {
      status = loom_check_execute_case(test_case, i, &report, filename,
                                       &input_request, environment, block_pool,
                                       allocator, &results[i]);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }

    if (options->verbose || results[i].final_outcome == LOOM_CHECK_FAIL) {
      loom_check_print_case_header(filename, i, test_case, &results[i]);
    }
    if ((results[i].final_outcome == LOOM_CHECK_FAIL ||
         (options->verbose && results[i].final_outcome == LOOM_CHECK_SKIP)) &&
        results[i].detail.size > 0) {
      fprintf(stderr, "%.*s", (int)results[i].detail.size,
              results[i].detail.buffer);
    }

    if (results[i].final_outcome == LOOM_CHECK_PASS) {
      ++(*pass_count);
    } else if (results[i].final_outcome == LOOM_CHECK_SKIP) {
      ++(*skip_count);
    } else {
      ++(*fail_count);
    }

    const bool wants_json_case =
        options->json_enabled &&
        (options->json_output_mode == LOOM_CHECK_JSON_OUTPUT_ALL ||
         (options->json_output_mode == LOOM_CHECK_JSON_OUTPUT_FAILURES &&
          results[i].final_outcome == LOOM_CHECK_FAIL));
    if ((wants_json_case || updates) && results[i].has_actual_output &&
        !iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_CHECKS)) {
      iree_string_view_t stripped_expected_trimmed =
          iree_string_view_trim(test_case->expected);
      iree_string_view_t actual_output =
          iree_string_builder_view(&results[i].actual_output);
      iree_string_view_t actual_trimmed = iree_string_view_trim(actual_output);
      if (!iree_string_view_equal(stripped_expected_trimmed, actual_trimmed)) {
        if (wants_json_case) {
          status = loom_check_build_update_edit(
              source, test_case, actual_output, &results[i].update_edit.text,
              &results[i].update_edit.value);
          if (iree_status_is_ok(status)) {
            results[i].update_edit.present = true;
          }
        }
        if (iree_status_is_ok(status) && updates) {
          updates[i].needs_update = true;
          updates[i].actual_output = actual_output;
          updates[i].input_end = test_case->input.data + test_case->input.size;
          if (test_case->has_expected_section) {
            updates[i].expected_start = test_case->expected.data;
            updates[i].expected_end =
                test_case->expected.data + test_case->expected.size;
          }
        }
      }
    }
  }

  if (iree_status_is_ok(status) && updates) {
    bool any_updates = false;
    for (iree_host_size_t i = 0; i < file.case_count; ++i) {
      if (updates[i].needs_update) {
        any_updates = true;
        break;
      }
    }
    if (any_updates) {
      status =
          loom_check_write_updates(path, source, &file, updates, allocator);
    }
  }

  if (iree_status_is_ok(status) && options->json_enabled) {
    loom_output_stream_t stdout_stream;
    loom_output_stream_for_file(stdout, &stdout_stream);
    status = loom_check_json_write_file_result(
        filename, &file, &report, results, *pass_count, *fail_count,
        *skip_count, options->json_output_mode, &stdout_stream);
  }

  if (results) {
    for (iree_host_size_t i = 0; i < initialized_result_count; ++i) {
      loom_check_result_deinitialize(&results[i]);
    }
    iree_allocator_free(allocator, results);
  }
  iree_allocator_free(allocator, updates);
  return status;
}

iree_status_t loom_check_read_and_process(
    iree_string_view_t path, const loom_check_process_options_t* options,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_host_size_t* pass_count, iree_host_size_t* fail_count,
    iree_host_size_t* skip_count) {
  const bool is_stdin = loom_tooling_file_path_is_stdio(path);

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(path, host_allocator, &contents);

  if (iree_status_is_ok(status)) {
    iree_string_view_t source =
        loom_tooling_file_contents_string_view(contents);
    iree_string_view_t filename = is_stdin ? IREE_SV("<stdin>") : path;
    char* filename_storage = NULL;
    status = loom_tooling_source_path_remap(
        filename, &options->source_path_options, host_allocator, &filename,
        &filename_storage);
    if (iree_status_is_ok(status)) {
      iree_arena_allocator_t arena;
      iree_arena_initialize(block_pool, &arena);
      status = loom_check_process_file(
          path, filename, source, is_stdin, options, environment, block_pool,
          &arena, host_allocator, pass_count, fail_count, skip_count);
      iree_arena_deinitialize(&arena);
    }
    iree_allocator_free(host_allocator, filename_storage);
  }

  iree_io_file_contents_free(contents);
  return status;
}
