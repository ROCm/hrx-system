// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/file.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#if defined(IREE_PLATFORM_WINDOWS)
#include <direct.h>
#else
#include <sys/types.h>
#endif  // defined(IREE_PLATFORM_WINDOWS)

#include "iree/base/internal/path.h"

static iree_status_t loom_tooling_file_path_dup(iree_string_view_t path,
                                                iree_allocator_t allocator,
                                                char** out_path) {
  *out_path = NULL;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, path.size + 1, (void**)&storage));
  iree_string_view_to_cstring(path, storage, path.size + 1);
  *out_path = storage;
  return iree_ok_status();
}

bool loom_tooling_file_path_is_stdio(iree_string_view_t path) {
  return iree_string_view_is_empty(path) ||
         iree_string_view_equal(path, IREE_SV("-"));
}

bool loom_tooling_file_path_has_trailing_separator(iree_string_view_t path) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return false;
  }
  const char last_char = path.data[path.size - 1];
  return last_char == '/' || last_char == '\\';
}

iree_status_t loom_tooling_file_path_is_directory(iree_string_view_t path,
                                                  iree_allocator_t allocator,
                                                  bool* out_is_directory) {
  *out_is_directory = false;
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_file_path_dup(path, allocator, &storage));
  struct stat file_stat;
  const int result = stat(storage, &file_stat);
  const int stat_error = result == 0 ? 0 : errno;
  iree_allocator_free(allocator, storage);
  if (result != 0) {
    if (stat_error == ENOENT || stat_error == ENOTDIR) {
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(stat_error),
                            "failed to stat path '%.*s' (%d)", (int)path.size,
                            path.data, stat_error);
  }
#if defined(IREE_PLATFORM_WINDOWS)
  *out_is_directory = (file_stat.st_mode & _S_IFMT) == _S_IFDIR;
#else
  *out_is_directory = S_ISDIR(file_stat.st_mode);
#endif  // defined(IREE_PLATFORM_WINDOWS)
  return iree_ok_status();
}

iree_status_t loom_tooling_file_path_join(iree_string_view_t lhs,
                                          iree_string_view_t rhs,
                                          iree_allocator_t allocator,
                                          char** out_path) {
  return iree_file_path_join(lhs, rhs, allocator, out_path);
}

iree_status_t loom_tooling_create_directory_if_needed(
    iree_string_view_t path, iree_allocator_t allocator) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_file_path_dup(path, allocator, &storage));
  iree_host_size_t length = strlen(storage);
  while (length > 1 &&
         (storage[length - 1] == '/' || storage[length - 1] == '\\')) {
    storage[--length] = '\0';
  }
  if (length == 1 && (storage[0] == '/' || storage[0] == '\\')) {
    iree_allocator_free(allocator, storage);
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 1; i < length && iree_status_is_ok(status); ++i) {
    if (storage[i] != '/' && storage[i] != '\\') {
      continue;
    }
    const char separator = storage[i];
    storage[i] = '\0';
    if (!(i == 2 && storage[1] == ':') && strlen(storage) != 0) {
#if defined(IREE_PLATFORM_WINDOWS)
      const int result = _mkdir(storage);
#else
      const int result = mkdir(storage, 0700);
#endif  // defined(IREE_PLATFORM_WINDOWS)
      if (result != 0 && errno != EEXIST) {
        const int create_error = errno;
        status = iree_make_status(iree_status_code_from_errno(create_error),
                                  "failed to create directory '%s' (%d)",
                                  storage, create_error);
      }
    }
    storage[i] = separator;
  }
  if (iree_status_is_ok(status)) {
#if defined(IREE_PLATFORM_WINDOWS)
    const int result = _mkdir(storage);
#else
    const int result = mkdir(storage, 0700);
#endif  // defined(IREE_PLATFORM_WINDOWS)
    if (result != 0 && errno != EEXIST) {
      const int create_error = errno;
      status = iree_make_status(iree_status_code_from_errno(create_error),
                                "failed to create directory '%s' (%d)", storage,
                                create_error);
    }
  }
  iree_allocator_free(allocator, storage);
  return status;
}

iree_status_t loom_tooling_read_input_file(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents) {
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_io_file_contents_read_stdin(allocator, out_contents);
  }
  return iree_io_file_contents_read(path, allocator, out_contents);
}

iree_string_view_t loom_tooling_file_contents_string_view(
    const iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

iree_status_t loom_tooling_write_output_file(iree_string_view_t path,
                                             iree_string_view_t contents,
                                             iree_allocator_t allocator) {
  iree_const_byte_span_t bytes =
      iree_make_const_byte_span(contents.data, contents.size);
  if (!loom_tooling_file_path_is_stdio(path)) {
    return iree_io_file_contents_write(path, bytes, allocator);
  }

  if (bytes.data_length > 0 &&
      fwrite(bytes.data, bytes.data_length, 1, stdout) != 1) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write %" PRIhsz " bytes to stdout",
                            bytes.data_length);
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

iree_status_t loom_tooling_write_stdout(iree_string_view_t contents) {
  if (iree_string_view_is_empty(contents)) {
    return iree_ok_status();
  }
  return loom_tooling_write_output_file(IREE_SV("-"), contents,
                                        iree_allocator_null());
}

iree_status_t loom_tooling_output_stream_open(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_tooling_output_stream_t* out_output) {
  *out_output = (loom_tooling_output_stream_t){
      .path = path,
  };
  if (iree_string_view_equal(path, IREE_SV("stderr"))) {
    out_output->file = stderr;
    loom_output_stream_for_file(stderr, &out_output->stream);
    return iree_ok_status();
  }
  if (loom_tooling_file_path_is_stdio(path) ||
      iree_string_view_equal(path, IREE_SV("stdout"))) {
    out_output->file = stdout;
    loom_output_stream_for_file(stdout, &out_output->stream);
    return iree_ok_status();
  }

  char* path_cstring = NULL;
  IREE_RETURN_IF_ERROR(
      loom_tooling_file_path_dup(path, allocator, &path_cstring));
  FILE* file = fopen(path_cstring, "wb");
  const int open_error = file == NULL ? errno : 0;
  iree_allocator_free(allocator, path_cstring);
  if (file == NULL) {
    return iree_make_status(iree_status_code_from_errno(open_error),
                            "failed to open output stream '%.*s' (%d)",
                            (int)path.size, path.data, open_error);
  }
  out_output->file = file;
  out_output->close_file = true;
  loom_output_stream_for_file(file, &out_output->stream);
  return iree_ok_status();
}

iree_status_t loom_tooling_output_stream_close(
    loom_tooling_output_stream_t* output) {
  if (!output || !output->file) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  if (output->close_file) {
    if (fclose(output->file) != 0) {
      const int close_error = errno;
      status = iree_make_status(iree_status_code_from_errno(close_error),
                                "failed to close output stream '%.*s' (%d)",
                                (int)output->path.size, output->path.data,
                                close_error);
    }
  } else if (fflush(output->file) != 0) {
    const int flush_error = errno;
    status = iree_make_status(iree_status_code_from_errno(flush_error),
                              "failed to flush output stream '%.*s' (%d)",
                              (int)output->path.size, output->path.data,
                              flush_error);
  }
  memset(output, 0, sizeof(*output));
  return status;
}
