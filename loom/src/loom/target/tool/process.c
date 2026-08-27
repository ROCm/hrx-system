// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process.h"

#include <string.h>

#include "loom/target/tool/process_platform.h"

//===----------------------------------------------------------------------===//
// String and argv utilities
//===----------------------------------------------------------------------===//

static bool loom_tool_string_view_contains_nul(iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] == '\0') {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_tool_dup_cstring(iree_string_view_t value,
                                           iree_allocator_t allocator,
                                           char** out_string) {
  *out_string = NULL;
  if (loom_tool_string_view_contains_nul(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool arguments cannot contain NUL bytes");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string size overflow");
  }
  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&string));
  if (value.size != 0) {
    memcpy(string, value.data, value.size);
  }
  string[value.size] = '\0';
  *out_string = string;
  return iree_ok_status();
}

static void loom_tool_argv_deinitialize(char** argv,
                                        iree_allocator_t allocator) {
  if (argv == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; argv[i] != NULL; ++i) {
    iree_allocator_free(allocator, argv[i]);
  }
  iree_allocator_free(allocator, argv);
}

static iree_status_t loom_tool_argv_allocate(
    iree_string_view_t executable_path, const iree_string_view_t* arguments,
    iree_host_size_t argument_count, iree_allocator_t allocator,
    char*** out_argv) {
  *out_argv = NULL;
  if (iree_string_view_is_empty(executable_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool executable path is required");
  }

  iree_host_size_t argv_count = 0;
  iree_host_size_t total_count = 0;
  if (!iree_host_size_checked_add(argument_count, 1, &argv_count) ||
      !iree_host_size_checked_add(argv_count, 1, &total_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "argv size overflow");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(total_count, sizeof(char*), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "argv allocation size overflow");
  }

  char** argv = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&argv));
  memset(argv, 0, total_size);

  iree_status_t status =
      loom_tool_dup_cstring(executable_path, allocator, &argv[0]);
  for (iree_host_size_t i = 0; i < argument_count && iree_status_is_ok(status);
       ++i) {
    status = loom_tool_dup_cstring(arguments[i], allocator, &argv[i + 1]);
  }
  if (iree_status_is_ok(status)) {
    *out_argv = argv;
  } else {
    loom_tool_argv_deinitialize(argv, allocator);
  }
  return status;
}

void loom_tool_output_deinitialize(loom_tool_output_t* output,
                                   iree_allocator_t allocator) {
  if (output == NULL) {
    return;
  }
  iree_allocator_free(allocator, output->data);
  *output = (loom_tool_output_t){0};
}

void loom_tool_output_normalize_newlines(loom_tool_output_t* output) {
  if (output == NULL || output->data == NULL) {
    return;
  }
  iree_host_size_t read_position = 0;
  iree_host_size_t write_position = 0;
  while (read_position < output->length) {
    if (output->data[read_position] == '\r') {
      iree_host_size_t newline_position = read_position;
      while (newline_position < output->length &&
             output->data[newline_position] == '\r') {
        ++newline_position;
      }
      if (newline_position < output->length &&
          output->data[newline_position] == '\n') {
        output->data[write_position++] = '\n';
        read_position = newline_position + 1;
        continue;
      }
    }
    output->data[write_position++] = output->data[read_position++];
  }
  output->data[write_position] = '\0';
  output->length = write_position;
}

void loom_tool_process_result_deinitialize(loom_tool_process_result_t* result,
                                           iree_allocator_t allocator) {
  if (result == NULL) {
    return;
  }
  loom_tool_output_deinitialize(&result->stdout_bytes, allocator);
  loom_tool_output_deinitialize(&result->stderr_bytes, allocator);
  result->exit_code = 0;
}

bool loom_tool_process_result_succeeded(
    const loom_tool_process_result_t* result) {
  return result != NULL && result->exit_code == 0;
}

//===----------------------------------------------------------------------===//
// Public process API
//===----------------------------------------------------------------------===//

iree_status_t loom_tool_process_run(iree_string_view_t executable_path,
                                    bool search_path,
                                    const iree_string_view_t* arguments,
                                    iree_host_size_t argument_count,
                                    iree_allocator_t allocator,
                                    loom_tool_process_result_t* out_result) {
  *out_result = (loom_tool_process_result_t){0};

  char** argv = NULL;
  IREE_RETURN_IF_ERROR(loom_tool_argv_allocate(
      executable_path, arguments, argument_count, allocator, &argv));
  iree_status_t status =
      loom_tool_process_run_platform(argv, search_path, allocator, out_result);
  loom_tool_argv_deinitialize(argv, allocator);
  return status;
}

//===----------------------------------------------------------------------===//
// Temporary files
//===----------------------------------------------------------------------===//

static bool loom_tool_temp_file_stem_char_is_valid(char value) {
  return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
         (value >= '0' && value <= '9') || value == '_' || value == '-';
}

static iree_status_t loom_tool_temp_file_copy_stem(iree_string_view_t stem,
                                                   char* buffer,
                                                   iree_host_size_t capacity) {
  if (iree_string_view_is_empty(stem)) {
    stem = IREE_SV("tmp");
  }
  if (stem.size >= capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary file stem is too long");
  }
  for (iree_host_size_t i = 0; i < stem.size; ++i) {
    if (!loom_tool_temp_file_stem_char_is_valid(stem.data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid temporary file stem character '%c'",
                              stem.data[i]);
    }
    buffer[i] = stem.data[i];
  }
  buffer[stem.size] = '\0';
  return iree_ok_status();
}

iree_string_view_t loom_tool_temp_file_path(const loom_tool_temp_file_t* file) {
  return iree_make_cstring_view(file->path);
}

iree_status_t loom_tool_temp_file_initialize(iree_string_view_t stem,
                                             loom_tool_temp_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));
  char stem_buffer[32] = {0};
  IREE_RETURN_IF_ERROR(loom_tool_temp_file_copy_stem(
      stem, stem_buffer, IREE_ARRAYSIZE(stem_buffer)));
  return loom_tool_temp_file_initialize_platform(stem_buffer, out_file);
}

iree_status_t loom_tool_temp_file_deinitialize(loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return iree_ok_status();
  }
  return loom_tool_temp_file_deinitialize_platform(file);
}
