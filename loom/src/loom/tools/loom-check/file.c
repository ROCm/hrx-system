// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/file.h"

#include <stdio.h>
#include <stdlib.h>

#include "iree/io/file_contents.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/json_output.h"
#include "loom/tools/loom-check/output.h"
#include "loom/tools/loom-check/template_sync.h"
#include "loom/tools/loom-check/update.h"
#include "loom/util/stream.h"

static iree_status_t loom_check_write_source(iree_string_view_t path,
                                             iree_string_view_t source,
                                             iree_allocator_t allocator) {
  return loom_tooling_write_output_file(path, source, allocator);
}

static iree_status_t loom_check_write_updates(
    iree_string_view_t path, iree_string_view_t original_source,
    const loom_check_file_t* file, const loom_check_case_update_t* updates,
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
    iree_string_view_t filename, iree_string_view_t source, bool is_stdin,
    const loom_check_process_options_t* options,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    iree_host_size_t* pass_count, iree_host_size_t* fail_count,
    iree_host_size_t* skip_count) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  loom_check_file_t file = {0};
  iree_status_t status = loom_check_parse(source, &arena, &file);

  iree_string_builder_t template_synced_source;
  iree_string_builder_initialize(allocator, &template_synced_source);
  bool template_sync_changed = false;
  if (iree_status_is_ok(status) && options->update &&
      file.has_template_directive) {
    if (is_stdin) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--update cannot be used with stdin");
    } else {
      iree_io_file_contents_t* template_contents = NULL;
      status = iree_io_file_contents_read(file.template_path, allocator,
                                          &template_contents);
      if (iree_status_is_ok(status)) {
        status = loom_check_template_sync_build_source(
            source, &file, filename,
            loom_tooling_file_contents_string_view(template_contents),
            file.template_path, context, block_pool, &arena, allocator,
            &template_synced_source, &template_sync_changed);
      }
      iree_io_file_contents_free(template_contents);
      if (iree_status_is_ok(status) && template_sync_changed) {
        iree_arena_deinitialize(&arena);
        iree_arena_initialize(block_pool, &arena);
        source = iree_string_builder_view(&template_synced_source);
        file = (loom_check_file_t){0};
        status = loom_check_parse(source, &arena, &file);
      }
    }
  }

  loom_check_file_report_t report = {0};
  if (iree_status_is_ok(status)) {
    status = loom_check_file_report_initialize(&file, &arena, &report);
  }

  loom_check_case_update_t* updates = NULL;
  if (iree_status_is_ok(status) && options->update) {
    if (is_stdin) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--update cannot be used with stdin");
    } else {
      updates =
          (loom_check_case_update_t*)calloc(file.case_count, sizeof(*updates));
      if (!updates) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to allocate update tracking");
      }
    }
  }

  loom_check_result_t* results = NULL;
  if (iree_status_is_ok(status) && file.case_count > 0) {
    results = (loom_check_result_t*)calloc(file.case_count, sizeof(*results));
    if (!results) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to allocate result array");
    }
  }

  iree_host_size_t initialized_result_count = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < file.case_count;
       ++i) {
    const loom_check_case_t* test_case = &file.cases[i];

    loom_check_result_initialize(allocator, &results[i]);
    ++initialized_result_count;
    status =
        loom_check_execute_case(test_case, i, &report, filename, environment,
                                context, block_pool, allocator, &results[i]);
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
    if ((wants_json_case || updates) && results[i].has_actual_output) {
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
          loom_check_write_updates(filename, source, &file, updates, allocator);
    } else if (template_sync_changed) {
      status = loom_check_write_source(filename, source, allocator);
      if (iree_status_is_ok(status)) {
        fprintf(stderr, "synchronized template cases in %.*s\n",
                (int)filename.size, filename.data);
      }
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
    free(results);
  }
  free(updates);
  iree_string_builder_deinitialize(&template_synced_source);
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_check_read_and_process(
    iree_string_view_t path, const loom_check_process_options_t* options,
    const loom_check_environment_t* environment, loom_context_t* context,
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
    status = loom_check_process_file(
        filename, source, is_stdin, options, environment, context, block_pool,
        host_allocator, pass_count, fail_count, skip_count);
  }

  iree_io_file_contents_free(contents);
  return status;
}
