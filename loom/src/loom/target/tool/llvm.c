// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/llvm.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"

//===----------------------------------------------------------------------===//
// Tool resolution and result utilities
//===----------------------------------------------------------------------===//

void loom_llvm_toolchain_initialize_from_environment(
    loom_llvm_toolchain_t* out_toolchain) {
  const char* root_path = getenv("LOOM_LLVM_TOOLCHAIN_ROOT");
  if (root_path == NULL || root_path[0] == '\0') {
    root_path = getenv("LOOM_LLVM_TOOLCHAIN_BIN");
  }
  *out_toolchain = (loom_llvm_toolchain_t){
      .root_path = root_path ? iree_make_cstring_view(root_path)
                             : iree_string_view_empty(),
  };
}

iree_string_view_t loom_llvm_tool_name(loom_llvm_tool_kind_t tool_kind) {
  switch (tool_kind) {
    case LOOM_LLVM_TOOL_LLVM_AS:
      return IREE_SV("llvm-as" IREE_PLATFORM_EXECUTABLE_SUFFIX);
    case LOOM_LLVM_TOOL_LLVM_DIS:
      return IREE_SV("llvm-dis" IREE_PLATFORM_EXECUTABLE_SUFFIX);
    case LOOM_LLVM_TOOL_OPT:
      return IREE_SV("opt" IREE_PLATFORM_EXECUTABLE_SUFFIX);
    case LOOM_LLVM_TOOL_LLC:
      return IREE_SV("llc" IREE_PLATFORM_EXECUTABLE_SUFFIX);
    case LOOM_LLVM_TOOL_LLVM_MC:
      return IREE_SV("llvm-mc" IREE_PLATFORM_EXECUTABLE_SUFFIX);
    case LOOM_LLVM_TOOL_LLVM_OBJDUMP:
      return IREE_SV("llvm-objdump" IREE_PLATFORM_EXECUTABLE_SUFFIX);
  }
  return iree_string_view_empty();
}

static iree_status_t loom_llvm_tool_executable_path(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    iree_allocator_t allocator, iree_string_view_t* out_executable_path,
    char** out_executable_path_storage, bool* out_search_path) {
  *out_executable_path = iree_string_view_empty();
  *out_executable_path_storage = NULL;
  *out_search_path = false;

  iree_string_view_t tool_name = loom_llvm_tool_name(tool_kind);
  if (iree_string_view_is_empty(tool_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM tool kind %d", (int)tool_kind);
  }

  if (toolchain == NULL || iree_string_view_is_empty(toolchain->root_path)) {
    *out_executable_path = tool_name;
    *out_search_path = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_file_path_join(
      toolchain->root_path, tool_name, allocator, out_executable_path_storage));
  *out_executable_path = iree_make_cstring_view(*out_executable_path_storage);
  return iree_ok_status();
}

static iree_status_t loom_llvm_tool_checked_status(
    loom_llvm_tool_kind_t tool_kind, const loom_tool_process_result_t* result,
    iree_string_view_t action) {
  iree_string_view_t tool_name = loom_llvm_tool_name(tool_kind);
  int tool_name_length =
      (int)iree_min(tool_name.size, (iree_host_size_t)INT_MAX);
  int stdout_length =
      (int)iree_min(result->stdout_bytes.length, (iree_host_size_t)INT_MAX);
  int stderr_length =
      (int)iree_min(result->stderr_bytes.length, (iree_host_size_t)INT_MAX);
  int action_length = (int)iree_min(action.size, (iree_host_size_t)INT_MAX);
  const char* stdout_data =
      result->stdout_bytes.data ? result->stdout_bytes.data : "";
  const char* stderr_data =
      result->stderr_bytes.data ? result->stderr_bytes.data : "";
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "LLVM tool %.*s failed while %.*s with exit code %d\nstdout:\n%.*s\n"
      "stderr:\n%.*s",
      tool_name_length, tool_name.data, action_length, action.data,
      result->exit_code, stdout_length, stdout_data, stderr_length,
      stderr_data);
}

static iree_status_t loom_llvm_tool_output_read_file(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_tool_output_t* out_output) {
  *out_output = (loom_tool_output_t){0};
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_read(path, allocator, &contents));

  iree_host_size_t length = contents->const_buffer.data_length;
  iree_host_size_t total_size = 0;
  iree_status_t status = iree_ok_status();
  if (!iree_host_size_checked_add(length, 1, &total_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "LLVM tool output file is too large");
  }
  char* data = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&data);
  }
  if (iree_status_is_ok(status)) {
    if (length != 0) {
      memcpy(data, contents->const_buffer.data, length);
    }
    data[length] = '\0';
    *out_output = (loom_tool_output_t){
        .data = data,
        .length = length,
    };
  } else {
    iree_allocator_free(allocator, data);
  }
  iree_io_file_contents_free(contents);
  return status;
}

//===----------------------------------------------------------------------===//
// Public execution API
//===----------------------------------------------------------------------===//

iree_status_t loom_llvm_tool_run(const loom_llvm_toolchain_t* toolchain,
                                 loom_llvm_tool_kind_t tool_kind,
                                 const iree_string_view_t* arguments,
                                 iree_host_size_t argument_count,
                                 iree_allocator_t allocator,
                                 loom_tool_process_result_t* out_result) {
  *out_result = (loom_tool_process_result_t){0};

  iree_string_view_t executable_path = iree_string_view_empty();
  char* executable_path_storage = NULL;
  bool search_path = false;
  iree_status_t status = loom_llvm_tool_executable_path(
      toolchain, tool_kind, allocator, &executable_path,
      &executable_path_storage, &search_path);
  if (iree_status_is_ok(status)) {
    status = loom_tool_process_run(executable_path, search_path, arguments,
                                   argument_count, allocator, out_result);
  }
  iree_allocator_free(allocator, executable_path_storage);
  return status;
}

static iree_status_t loom_llvm_tool_run_checked(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    const iree_string_view_t* arguments, iree_host_size_t argument_count,
    iree_allocator_t allocator, iree_string_view_t action) {
  loom_tool_process_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_llvm_tool_run(toolchain, tool_kind, arguments,
                                          argument_count, allocator, &result));

  iree_status_t status = iree_ok_status();
  if (!loom_tool_process_result_succeeded(&result)) {
    status = loom_llvm_tool_checked_status(tool_kind, &result, action);
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvm_tool_query_version(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_tool_output_t* out_version_text) {
  *out_version_text = (loom_tool_output_t){0};
  iree_string_view_t arguments[] = {IREE_SV("--version")};
  loom_tool_process_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_llvm_tool_run(toolchain, tool_kind, arguments,
                                          IREE_ARRAYSIZE(arguments), allocator,
                                          &result));

  iree_status_t status = iree_ok_status();
  if (loom_tool_process_result_succeeded(&result)) {
    loom_tool_output_normalize_newlines(&result.stdout_bytes);
    *out_version_text = result.stdout_bytes;
    result.stdout_bytes = (loom_tool_output_t){0};
  } else {
    status = loom_llvm_tool_checked_status(
        tool_kind, &result, IREE_SV("querying LLVM tool version"));
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvm_tool_assemble_ir_text_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, iree_allocator_t allocator) {
  iree_string_view_t arguments[] = {
      input_path,
      IREE_SV("-o"),
      output_path,
  };
  return loom_llvm_tool_run_checked(
      toolchain, LOOM_LLVM_TOOL_LLVM_AS, arguments, IREE_ARRAYSIZE(arguments),
      allocator, IREE_SV("assembling LLVM IR text"));
}

iree_status_t loom_llvm_tool_disassemble_bitcode_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator, loom_tool_output_t* out_text) {
  *out_text = (loom_tool_output_t){0};
  iree_string_view_t arguments[] = {
      input_path,
      IREE_SV("-o"),
      IREE_SV("-"),
  };
  loom_tool_process_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_llvm_tool_run(toolchain, LOOM_LLVM_TOOL_LLVM_DIS,
                                          arguments, IREE_ARRAYSIZE(arguments),
                                          allocator, &result));

  iree_status_t status = iree_ok_status();
  if (loom_tool_process_result_succeeded(&result)) {
    loom_tool_output_normalize_newlines(&result.stdout_bytes);
    *out_text = result.stdout_bytes;
    result.stdout_bytes = (loom_tool_output_t){0};
  } else {
    status =
        loom_llvm_tool_checked_status(LOOM_LLVM_TOOL_LLVM_DIS, &result,
                                      IREE_SV("disassembling LLVM bitcode"));
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvm_tool_disassemble_bitcode(
    const loom_llvm_toolchain_t* toolchain, iree_const_byte_span_t bitcode,
    iree_allocator_t allocator, loom_tool_output_t* out_text) {
  *out_text = (loom_tool_output_t){0};

  loom_tool_temp_file_t input_file;
  iree_status_t status =
      loom_tool_temp_file_initialize(IREE_SV("bc"), &input_file);
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(loom_tool_temp_file_path(&input_file),
                                         bitcode, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvm_tool_disassemble_bitcode_file(
        toolchain, loom_tool_temp_file_path(&input_file), allocator, out_text);
  }
  status =
      iree_status_join(status, loom_tool_temp_file_deinitialize(&input_file));
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(out_text, allocator);
  }
  return status;
}

iree_status_t loom_llvm_tool_verify_bitcode_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator) {
  iree_string_view_t arguments[] = {
      IREE_SV("-passes=verify"),
      input_path,
      IREE_SV("-disable-output"),
  };
  return loom_llvm_tool_run_checked(toolchain, LOOM_LLVM_TOOL_OPT, arguments,
                                    IREE_ARRAYSIZE(arguments), allocator,
                                    IREE_SV("verifying LLVM bitcode"));
}

static iree_status_t loom_llvm_tool_run_file_to_file(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    iree_string_view_t input_path, iree_string_view_t output_path,
    iree_string_view_t file_type_argument, iree_string_view_t action,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator) {
  iree_host_size_t argument_count = 0;
  if (!iree_host_size_checked_add(extra_argument_count, 4, &argument_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM tool argument count overflow");
  }
  iree_host_size_t argument_size = 0;
  if (!iree_host_size_checked_mul(argument_count, sizeof(iree_string_view_t),
                                  &argument_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM tool argument allocation overflow");
  }

  iree_string_view_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, argument_size, (void**)&arguments));
  arguments[0] = input_path;
  arguments[1] = file_type_argument;
  arguments[2] = IREE_SV("-o");
  arguments[3] = output_path;
  for (iree_host_size_t i = 0; i < extra_argument_count; ++i) {
    arguments[i + 4] = extra_arguments[i];
  }

  iree_status_t status = loom_llvm_tool_run_checked(
      toolchain, tool_kind, arguments, argument_count, allocator, action);
  iree_allocator_free(allocator, arguments);
  return status;
}

iree_status_t loom_llvm_tool_compile_object_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator) {
  return loom_llvm_tool_run_file_to_file(
      toolchain, LOOM_LLVM_TOOL_LLC, input_path, output_path,
      IREE_SV("-filetype=obj"),
      IREE_SV("compiling LLVM bitcode to an object file"), extra_arguments,
      extra_argument_count, allocator);
}

iree_status_t loom_llvm_tool_compile_assembly_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator) {
  return loom_llvm_tool_run_file_to_file(
      toolchain, LOOM_LLVM_TOOL_LLC, input_path, output_path,
      IREE_SV("-filetype=asm"), IREE_SV("compiling LLVM bitcode to assembly"),
      extra_arguments, extra_argument_count, allocator);
}

iree_status_t loom_llvm_tool_assemble_native_object_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator) {
  return loom_llvm_tool_run_file_to_file(
      toolchain, LOOM_LLVM_TOOL_LLVM_MC, input_path, output_path,
      IREE_SV("-filetype=obj"),
      IREE_SV("assembling native target assembly to an object file"),
      extra_arguments, extra_argument_count, allocator);
}

static iree_status_t loom_llvm_tool_run_bytes_to_file_output(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    iree_const_byte_span_t input_data, const char* input_suffix,
    const char* output_suffix, iree_string_view_t file_type_argument,
    iree_string_view_t action, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_output) {
  *out_output = (loom_tool_output_t){0};

  loom_tool_temp_file_t input_file = {0};
  loom_tool_temp_file_t output_file = {0};
  iree_status_t status = loom_tool_temp_file_initialize(
      iree_make_cstring_view(input_suffix), &input_file);
  if (iree_status_is_ok(status)) {
    status = loom_tool_temp_file_initialize(
        iree_make_cstring_view(output_suffix), &output_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(loom_tool_temp_file_path(&input_file),
                                         input_data, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvm_tool_run_file_to_file(
        toolchain, tool_kind, loom_tool_temp_file_path(&input_file),
        loom_tool_temp_file_path(&output_file), file_type_argument, action,
        extra_arguments, extra_argument_count, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvm_tool_output_read_file(
        loom_tool_temp_file_path(&output_file), allocator, out_output);
  }
  status =
      iree_status_join(status, loom_tool_temp_file_deinitialize(&output_file));
  status =
      iree_status_join(status, loom_tool_temp_file_deinitialize(&input_file));
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(out_output, allocator);
  }
  return status;
}

iree_status_t loom_llvm_tool_compile_object(
    const loom_llvm_toolchain_t* toolchain, iree_const_byte_span_t bitcode,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_object) {
  return loom_llvm_tool_run_bytes_to_file_output(
      toolchain, LOOM_LLVM_TOOL_LLC, bitcode, "bc", "obj",
      IREE_SV("-filetype=obj"),
      IREE_SV("compiling LLVM bitcode to an object file"), extra_arguments,
      extra_argument_count, allocator, out_object);
}

iree_status_t loom_llvm_tool_compile_assembly(
    const loom_llvm_toolchain_t* toolchain, iree_const_byte_span_t bitcode,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_assembly) {
  iree_status_t status = loom_llvm_tool_run_bytes_to_file_output(
      toolchain, LOOM_LLVM_TOOL_LLC, bitcode, "bc", "asm",
      IREE_SV("-filetype=asm"), IREE_SV("compiling LLVM bitcode to assembly"),
      extra_arguments, extra_argument_count, allocator, out_assembly);
  if (iree_status_is_ok(status)) {
    loom_tool_output_normalize_newlines(out_assembly);
  }
  return status;
}

iree_status_t loom_llvm_tool_assemble_native_object(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t assembly,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_object) {
  return loom_llvm_tool_run_bytes_to_file_output(
      toolchain, LOOM_LLVM_TOOL_LLVM_MC,
      iree_make_const_byte_span(assembly.data, assembly.size), "s", "obj",
      IREE_SV("-filetype=obj"),
      IREE_SV("assembling native target assembly to an object file"),
      extra_arguments, extra_argument_count, allocator, out_object);
}

static iree_status_t loom_llvm_tool_run_file_to_stdout(
    const loom_llvm_toolchain_t* toolchain, loom_llvm_tool_kind_t tool_kind,
    iree_string_view_t input_path, iree_string_view_t action,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_text) {
  *out_text = (loom_tool_output_t){0};
  iree_host_size_t argument_count = 0;
  if (!iree_host_size_checked_add(extra_argument_count, 1, &argument_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM tool argument count overflow");
  }
  iree_host_size_t argument_size = 0;
  if (!iree_host_size_checked_mul(argument_count, sizeof(iree_string_view_t),
                                  &argument_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM tool argument allocation overflow");
  }

  iree_string_view_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, argument_size, (void**)&arguments));
  for (iree_host_size_t i = 0; i < extra_argument_count; ++i) {
    arguments[i] = extra_arguments[i];
  }
  arguments[extra_argument_count] = input_path;

  loom_tool_process_result_t result = {0};
  iree_status_t status = loom_llvm_tool_run(toolchain, tool_kind, arguments,
                                            argument_count, allocator, &result);
  if (iree_status_is_ok(status)) {
    if (loom_tool_process_result_succeeded(&result)) {
      loom_tool_output_normalize_newlines(&result.stdout_bytes);
      *out_text = result.stdout_bytes;
      result.stdout_bytes = (loom_tool_output_t){0};
    } else {
      status = loom_llvm_tool_checked_status(tool_kind, &result, action);
    }
  }
  loom_tool_process_result_deinitialize(&result, allocator);
  iree_allocator_free(allocator, arguments);
  return status;
}

iree_status_t loom_llvm_tool_disassemble_object_file(
    const loom_llvm_toolchain_t* toolchain, iree_string_view_t input_path,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_text) {
  return loom_llvm_tool_run_file_to_stdout(
      toolchain, LOOM_LLVM_TOOL_LLVM_OBJDUMP, input_path,
      IREE_SV("disassembling native object file"), extra_arguments,
      extra_argument_count, allocator, out_text);
}

iree_status_t loom_llvm_tool_disassemble_object(
    const loom_llvm_toolchain_t* toolchain, iree_const_byte_span_t object,
    const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator,
    loom_tool_output_t* out_text) {
  *out_text = (loom_tool_output_t){0};

  loom_tool_temp_file_t input_file = {0};
  iree_status_t status =
      loom_tool_temp_file_initialize(IREE_SV("obj"), &input_file);
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(loom_tool_temp_file_path(&input_file),
                                         object, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvm_tool_disassemble_object_file(
        toolchain, loom_tool_temp_file_path(&input_file), extra_arguments,
        extra_argument_count, allocator, out_text);
  }
  status =
      iree_status_join(status, loom_tool_temp_file_deinitialize(&input_file));
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(out_text, allocator);
  }
  return status;
}
