// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// File input and output helpers shared by Loom command-line tools.

#ifndef LOOM_TOOLING_IO_FILE_H_
#define LOOM_TOOLING_IO_FILE_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/io/file_contents.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |path| denotes stdin or stdout for command-line tools.
bool loom_tooling_file_path_is_stdio(iree_string_view_t path);

// Returns true when |path| ends in a platform path separator.
bool loom_tooling_file_path_has_trailing_separator(iree_string_view_t path);

// Returns whether |path| exists and is a directory.
iree_status_t loom_tooling_file_path_is_directory(iree_string_view_t path,
                                                  iree_allocator_t allocator,
                                                  bool* out_is_directory);

// Joins |lhs| and |rhs| into an allocator-owned NUL-terminated path.
iree_status_t loom_tooling_file_path_join(iree_string_view_t lhs,
                                          iree_string_view_t rhs,
                                          iree_allocator_t allocator,
                                          char** out_path);

// Creates |path| and any missing parent directories.
iree_status_t loom_tooling_create_directory_if_needed(
    iree_string_view_t path, iree_allocator_t allocator);

// Reads |path| into |out_contents|, treating empty and "-" paths as stdin.
iree_status_t loom_tooling_read_input_file(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents);

// Returns the contents buffer as a borrowed string view.
iree_string_view_t loom_tooling_file_contents_string_view(
    const iree_io_file_contents_t* contents);

// Writes |contents| to |path|, treating empty and "-" paths as stdout.
iree_status_t loom_tooling_write_output_file(iree_string_view_t path,
                                             iree_string_view_t contents,
                                             iree_allocator_t allocator);

// Writes |contents| to stdout and flushes the stream.
iree_status_t loom_tooling_write_stdout(iree_string_view_t contents);

typedef struct loom_tooling_output_stream_t {
  // Stream adapter passed to formatters while this output is open.
  loom_output_stream_t stream;
  // Open FILE* backing stream.
  FILE* file;
  // True when file must be closed instead of flushed.
  bool close_file;
  // Borrowed destination path used for close diagnostics.
  iree_string_view_t path;
} loom_tooling_output_stream_t;

// Opens an output stream to stderr, stdout/"-", or a file path.
iree_status_t loom_tooling_output_stream_open(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_tooling_output_stream_t* out_output);

// Flushes or closes an open output stream.
iree_status_t loom_tooling_output_stream_close(
    loom_tooling_output_stream_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_IO_FILE_H_
