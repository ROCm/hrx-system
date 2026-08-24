// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-test-loom/library_linker.h"

#include "loom/link/linker.h"
#include "loom/tooling/io/file.h"

static iree_status_t iree_test_loom_add_library(loom_run_session_t* session,
                                                iree_string_view_t library_path,
                                                loom_linker_t* linker) {
  if (loom_tooling_file_path_is_stdio(library_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--library requires a filesystem path");
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_module_t library_module = {0};
  iree_status_t status = loom_tooling_read_input_file(
      library_path, session->host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = library_path;
    parse_options.source = loom_tooling_file_contents_string_view(contents);
    status = loom_run_module_parse(session, &parse_options, &library_module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_add_module(linker, library_module.module,
                                    /*options=*/NULL);
  }
  loom_run_module_deinitialize(&library_module);
  iree_io_file_contents_free(contents);
  return status;
}

iree_status_t iree_test_loom_link_libraries(
    loom_run_session_t* session, loom_run_module_t* run_module,
    iree_string_view_list_t library_paths) {
  if (library_paths.count == 0) {
    return iree_ok_status();
  }
  if (library_paths.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "library path count is non-zero but values is NULL");
  }

  loom_linker_t* linker = NULL;
  loom_module_t* linked_module = NULL;
  const loom_linker_options_t linker_options = {
      .module_name = IREE_SV("linked"),
  };
  iree_status_t status = loom_linker_allocate(
      loom_run_session_context(session), &linker_options,
      loom_run_session_block_pool(session), session->host_allocator, &linker);
  if (iree_status_is_ok(status)) {
    status = loom_linker_add_module(linker, run_module->module,
                                    /*options=*/NULL);
  }
  for (iree_host_size_t i = 0;
       i < library_paths.count && iree_status_is_ok(status); ++i) {
    status =
        iree_test_loom_add_library(session, library_paths.values[i], linker);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, &linked_module);
  }
  if (iree_status_is_ok(status)) {
    loom_module_free(run_module->module);
    run_module->module = linked_module;
    linked_module = NULL;
  }
  loom_module_free(linked_module);
  loom_linker_free(linker);
  return status;
}
