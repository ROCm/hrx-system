// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/spirv.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"

static iree_status_t loom_spirv_tool_dup_cstring(iree_string_view_t value,
                                                 iree_allocator_t allocator,
                                                 char** out_string) {
  *out_string = NULL;
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

void loom_spirv_toolchain_initialize_from_environment(
    loom_spirv_toolchain_t* out_toolchain) {
  IREE_ASSERT_ARGUMENT(out_toolchain);
  const char* root_path = getenv("LOOM_SPIRV_TOOLCHAIN_ROOT");
  if (root_path == NULL || root_path[0] == '\0') {
    root_path = getenv("LOOM_SPIRV_TOOLCHAIN_BIN");
  }
  *out_toolchain = (loom_spirv_toolchain_t){
      .root_path = root_path ? iree_make_cstring_view(root_path)
                             : iree_string_view_empty(),
  };
  static const char* kToolPathEnv[LOOM_SPIRV_TOOL_COUNT] = {
      [LOOM_SPIRV_TOOL_SPIRV_AS] = "LOOM_SPIRV_AS",
      [LOOM_SPIRV_TOOL_SPIRV_DIS] = "LOOM_SPIRV_DIS",
      [LOOM_SPIRV_TOOL_SPIRV_VAL] = "LOOM_SPIRV_VAL",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kToolPathEnv); ++i) {
    const char* tool_path = getenv(kToolPathEnv[i]);
    if (tool_path != NULL && tool_path[0] != '\0') {
      out_toolchain->tool_paths[i] = iree_make_cstring_view(tool_path);
    }
  }
}

iree_string_view_t loom_spirv_tool_name(loom_spirv_tool_kind_t tool_kind) {
  switch (tool_kind) {
    case LOOM_SPIRV_TOOL_SPIRV_AS:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("spirv-as.exe");
#else
      return IREE_SV("spirv-as");
#endif
    case LOOM_SPIRV_TOOL_SPIRV_DIS:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("spirv-dis.exe");
#else
      return IREE_SV("spirv-dis");
#endif
    case LOOM_SPIRV_TOOL_SPIRV_VAL:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("spirv-val.exe");
#else
      return IREE_SV("spirv-val");
#endif
    case LOOM_SPIRV_TOOL_COUNT:
      break;
  }
  return iree_string_view_empty();
}

static iree_status_t loom_spirv_tool_executable_path(
    const loom_spirv_toolchain_t* toolchain, loom_spirv_tool_kind_t tool_kind,
    iree_allocator_t allocator, char** out_executable_path,
    bool* out_search_path) {
  IREE_ASSERT_ARGUMENT(toolchain);
  *out_executable_path = NULL;
  *out_search_path = false;

  const iree_string_view_t tool_name = loom_spirv_tool_name(tool_kind);
  if (iree_string_view_is_empty(tool_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown SPIR-V tool kind %d", (int)tool_kind);
  }
  if (tool_kind < LOOM_SPIRV_TOOL_COUNT &&
      !iree_string_view_is_empty(toolchain->tool_paths[tool_kind])) {
    return loom_spirv_tool_dup_cstring(toolchain->tool_paths[tool_kind],
                                       allocator, out_executable_path);
  }
  if (iree_string_view_is_empty(toolchain->root_path)) {
    *out_search_path = true;
    return loom_spirv_tool_dup_cstring(tool_name, allocator,
                                       out_executable_path);
  }
  return iree_file_path_join(toolchain->root_path, tool_name, allocator,
                             out_executable_path);
}

iree_status_t loom_spirv_tool_run(const loom_spirv_toolchain_t* toolchain,
                                  loom_spirv_tool_kind_t tool_kind,
                                  const iree_string_view_t* arguments,
                                  iree_host_size_t argument_count,
                                  iree_allocator_t allocator,
                                  loom_tool_process_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_result);

  char* executable_path = NULL;
  bool search_path = false;
  IREE_RETURN_IF_ERROR(loom_spirv_tool_executable_path(
      toolchain, tool_kind, allocator, &executable_path, &search_path));
  iree_status_t status = loom_tool_process_run(
      iree_make_cstring_view(executable_path), search_path, arguments,
      argument_count, allocator, out_result);
  iree_allocator_free(allocator, executable_path);
  return status;
}

void loom_spirv_tool_output_deinitialize(loom_spirv_tool_output_t* output,
                                         iree_allocator_t allocator) {
  loom_tool_output_deinitialize(output, allocator);
}

static iree_status_t loom_spirv_tool_checked_status(
    loom_spirv_tool_kind_t tool_kind, const loom_tool_process_result_t* result,
    iree_string_view_t action) {
  const iree_string_view_t tool_name = loom_spirv_tool_name(tool_kind);
  const int tool_name_length =
      (int)iree_min(tool_name.size, (iree_host_size_t)INT_MAX);
  const int stdout_length =
      (int)iree_min(result->stdout_text.length, (iree_host_size_t)INT_MAX);
  const int stderr_length =
      (int)iree_min(result->stderr_text.length, (iree_host_size_t)INT_MAX);
  const int action_length =
      (int)iree_min(action.size, (iree_host_size_t)INT_MAX);
  const char* stdout_data =
      result->stdout_text.data ? result->stdout_text.data : "";
  const char* stderr_data =
      result->stderr_text.data ? result->stderr_text.data : "";
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "SPIR-V tool %.*s failed while %.*s with exit code %d\nstdout:\n%.*s\n"
      "stderr:\n%.*s",
      tool_name_length, tool_name.data, action_length, action.data,
      result->exit_code, stdout_length, stdout_data, stderr_length,
      stderr_data);
}

iree_status_t loom_spirv_tool_query_version(
    const loom_spirv_toolchain_t* toolchain, loom_spirv_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_spirv_tool_output_t* out_version_text) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_version_text);
  *out_version_text = (loom_spirv_tool_output_t){0};

  const iree_string_view_t arguments[] = {IREE_SV("--version")};
  loom_tool_process_result_t result = {0};
  iree_status_t status =
      loom_spirv_tool_run(toolchain, tool_kind, arguments,
                          IREE_ARRAYSIZE(arguments), allocator, &result);
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_spirv_tool_checked_status(tool_kind, &result,
                                            IREE_SV("querying version"));
  }
  if (iree_status_is_ok(status)) {
    *out_version_text = result.stdout_text;
    result.stdout_text = (loom_tool_output_t){0};
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

static iree_status_t loom_spirv_tool_write_temp_binary(
    iree_const_byte_span_t binary, iree_allocator_t allocator,
    loom_tool_temp_file_t* out_file) {
  IREE_RETURN_IF_ERROR(
      loom_tool_temp_file_allocate(IREE_SV("loom-spirv"), out_file));
  iree_status_t status = iree_io_file_contents_write(
      loom_tool_temp_file_path(out_file), binary, allocator);
  if (!iree_status_is_ok(status)) {
    loom_tool_temp_file_deinitialize(out_file);
  }
  return status;
}

iree_status_t loom_spirv_tool_disassemble_binary(
    const loom_spirv_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator, loom_spirv_tool_output_t* out_text) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_text);
  *out_text = (loom_spirv_tool_output_t){0};

  loom_tool_temp_file_t input_file = {0};
  iree_status_t status =
      loom_spirv_tool_write_temp_binary(binary, allocator, &input_file);

  loom_tool_process_result_t result = {0};
  if (iree_status_is_ok(status)) {
    const iree_string_view_t arguments[] = {
        loom_tool_temp_file_path(&input_file),
        IREE_SV("-o"),
        IREE_SV("-"),
    };
    status =
        loom_spirv_tool_run(toolchain, LOOM_SPIRV_TOOL_SPIRV_DIS, arguments,
                            IREE_ARRAYSIZE(arguments), allocator, &result);
  }
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_spirv_tool_checked_status(LOOM_SPIRV_TOOL_SPIRV_DIS, &result,
                                            IREE_SV("disassembling binary"));
  }
  if (iree_status_is_ok(status)) {
    *out_text = result.stdout_text;
    result.stdout_text = (loom_tool_output_t){0};
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  loom_tool_temp_file_deinitialize(&input_file);
  return status;
}

iree_status_t loom_spirv_tool_validate_binary(
    const loom_spirv_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(toolchain);
  loom_tool_temp_file_t input_file = {0};
  iree_status_t status =
      loom_spirv_tool_write_temp_binary(binary, allocator, &input_file);

  loom_tool_process_result_t result = {0};
  if (iree_status_is_ok(status)) {
    const iree_string_view_t arguments[] = {
        loom_tool_temp_file_path(&input_file),
    };
    status =
        loom_spirv_tool_run(toolchain, LOOM_SPIRV_TOOL_SPIRV_VAL, arguments,
                            IREE_ARRAYSIZE(arguments), allocator, &result);
  }
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_spirv_tool_checked_status(LOOM_SPIRV_TOOL_SPIRV_VAL, &result,
                                            IREE_SV("validating binary"));
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  loom_tool_temp_file_deinitialize(&input_file);
  return status;
}
