// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/wasm.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"

static iree_status_t loom_wasm_tool_dup_cstring(iree_string_view_t value,
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

void loom_wasm_toolchain_initialize_from_environment(
    loom_wasm_toolchain_t* out_toolchain) {
  IREE_ASSERT_ARGUMENT(out_toolchain);
  const char* llvm_objdump_path = getenv("LOOM_WASM_LLVM_OBJDUMP");
  const char* root_path = getenv("LOOM_WASM_TOOLCHAIN_ROOT");
  if (root_path == NULL || root_path[0] == '\0') {
    root_path = getenv("LOOM_WASM_TOOLCHAIN_BIN");
  }
  *out_toolchain = (loom_wasm_toolchain_t){
      .llvm_objdump_path = llvm_objdump_path
                               ? iree_make_cstring_view(llvm_objdump_path)
                               : iree_string_view_empty(),
      .root_path = root_path ? iree_make_cstring_view(root_path)
                             : iree_string_view_empty(),
  };
}

iree_string_view_t loom_wasm_tool_name(loom_wasm_tool_kind_t tool_kind) {
  switch (tool_kind) {
    case LOOM_WASM_TOOL_LLVM_OBJDUMP:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("llvm-objdump.exe");
#else
      return IREE_SV("llvm-objdump");
#endif
  }
  return iree_string_view_empty();
}

static iree_status_t loom_wasm_tool_executable_path(
    const loom_wasm_toolchain_t* toolchain, loom_wasm_tool_kind_t tool_kind,
    iree_allocator_t allocator, char** out_executable_path,
    bool* out_search_path) {
  IREE_ASSERT_ARGUMENT(toolchain);
  *out_executable_path = NULL;
  *out_search_path = false;

  const iree_string_view_t tool_name = loom_wasm_tool_name(tool_kind);
  if (iree_string_view_is_empty(tool_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Wasm tool kind %d", (int)tool_kind);
  }
  if (tool_kind == LOOM_WASM_TOOL_LLVM_OBJDUMP &&
      !iree_string_view_is_empty(toolchain->llvm_objdump_path)) {
    return loom_wasm_tool_dup_cstring(toolchain->llvm_objdump_path, allocator,
                                      out_executable_path);
  }
  if (iree_string_view_is_empty(toolchain->root_path)) {
    *out_search_path = true;
    return loom_wasm_tool_dup_cstring(tool_name, allocator,
                                      out_executable_path);
  }
  return iree_file_path_join(toolchain->root_path, tool_name, allocator,
                             out_executable_path);
}

iree_status_t loom_wasm_tool_run(const loom_wasm_toolchain_t* toolchain,
                                 loom_wasm_tool_kind_t tool_kind,
                                 const iree_string_view_t* arguments,
                                 iree_host_size_t argument_count,
                                 iree_allocator_t allocator,
                                 loom_tool_process_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_result);

  char* executable_path = NULL;
  bool search_path = false;
  IREE_RETURN_IF_ERROR(loom_wasm_tool_executable_path(
      toolchain, tool_kind, allocator, &executable_path, &search_path));
  iree_status_t status = loom_tool_process_run(
      iree_make_cstring_view(executable_path), search_path, arguments,
      argument_count, allocator, out_result);
  iree_allocator_free(allocator, executable_path);
  return status;
}

void loom_wasm_tool_output_deinitialize(loom_wasm_tool_output_t* output,
                                        iree_allocator_t allocator) {
  loom_tool_output_deinitialize(output, allocator);
}

static iree_status_t loom_wasm_tool_checked_status(
    loom_wasm_tool_kind_t tool_kind, const loom_tool_process_result_t* result,
    iree_string_view_t action) {
  const iree_string_view_t tool_name = loom_wasm_tool_name(tool_kind);
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
      "Wasm tool %.*s failed while %.*s with exit code %d\nstdout:\n%.*s\n"
      "stderr:\n%.*s",
      tool_name_length, tool_name.data, action_length, action.data,
      result->exit_code, stdout_length, stdout_data, stderr_length,
      stderr_data);
}

iree_status_t loom_wasm_tool_query_version(
    const loom_wasm_toolchain_t* toolchain, loom_wasm_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_wasm_tool_output_t* out_version_text) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_version_text);
  *out_version_text = (loom_wasm_tool_output_t){0};

  const iree_string_view_t arguments[] = {IREE_SV("--version")};
  loom_tool_process_result_t result = {0};
  iree_status_t status =
      loom_wasm_tool_run(toolchain, tool_kind, arguments,
                         IREE_ARRAYSIZE(arguments), allocator, &result);
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_wasm_tool_checked_status(tool_kind, &result,
                                           IREE_SV("querying version"));
  }
  if (iree_status_is_ok(status)) {
    *out_version_text = result.stdout_text;
    result.stdout_text = (loom_tool_output_t){0};
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

static iree_status_t loom_wasm_tool_write_temp_binary(
    iree_const_byte_span_t binary, iree_allocator_t allocator,
    loom_tool_temp_file_t* out_file) {
  IREE_RETURN_IF_ERROR(
      loom_tool_temp_file_initialize(IREE_SV("loom-wasm"), out_file));
  iree_status_t status = iree_io_file_contents_write(
      loom_tool_temp_file_path(out_file), binary, allocator);
  if (!iree_status_is_ok(status)) {
    loom_tool_temp_file_deinitialize(out_file);
  }
  return status;
}

iree_status_t loom_wasm_tool_disassemble_binary(
    const loom_wasm_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator, loom_wasm_tool_output_t* out_text) {
  IREE_ASSERT_ARGUMENT(toolchain);
  IREE_ASSERT_ARGUMENT(out_text);
  *out_text = (loom_wasm_tool_output_t){0};

  loom_tool_temp_file_t input_file = {0};
  iree_status_t status =
      loom_wasm_tool_write_temp_binary(binary, allocator, &input_file);

  loom_tool_process_result_t result = {0};
  if (iree_status_is_ok(status)) {
    const iree_string_view_t arguments[] = {
        IREE_SV("--disassemble"),
        IREE_SV("--no-show-raw-insn"),
        IREE_SV("--no-leading-headers"),
        loom_tool_temp_file_path(&input_file),
    };
    status =
        loom_wasm_tool_run(toolchain, LOOM_WASM_TOOL_LLVM_OBJDUMP, arguments,
                           IREE_ARRAYSIZE(arguments), allocator, &result);
  }
  if (iree_status_is_ok(status) &&
      !loom_tool_process_result_succeeded(&result)) {
    status = loom_wasm_tool_checked_status(LOOM_WASM_TOOL_LLVM_OBJDUMP, &result,
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
