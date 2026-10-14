// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/config/input.h"

#include "loom/tooling/io/file.h"

iree_status_t loom_tooling_config_text_binding_set_append_assignment(
    loom_config_text_binding_set_t* binding_set,
    iree_string_view_t assignment) {
  IREE_ASSERT_ARGUMENT(binding_set);
  iree_string_view_t key = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  if (iree_string_view_split(assignment, '=', &key, &value) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config assignment must use key=value syntax, got '%.*s'",
        (int)assignment.size, assignment.data);
  }
  return loom_config_text_binding_set_append(binding_set, key, value);
}

iree_status_t loom_tooling_config_text_binding_set_append_json_file(
    loom_config_text_binding_set_t* binding_set, iree_string_view_t path,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(binding_set);
  path = iree_string_view_trim(path);
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config JSON file requires a filesystem path, not stdin");
  }

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(path, host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = loom_config_text_binding_set_append_json_object(
        binding_set, loom_tooling_file_contents_string_view(contents));
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "config JSON file '%.*s'",
                                    (int)path.size, path.data);
  }
  iree_io_file_contents_free(contents);
  return status;
}
