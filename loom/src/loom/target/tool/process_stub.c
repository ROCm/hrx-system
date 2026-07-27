// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process_platform.h"

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_LINUX) && \
    !defined(IREE_PLATFORM_MACOS) && !defined(IREE_PLATFORM_ANDROID)

#include <string.h>

iree_status_t loom_tool_process_run_platform(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  (void)argv;
  (void)search_path;
  (void)allocator;
  *out_result = (loom_tool_process_result_t){0};
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "external tool execution is unsupported on this platform");
}

iree_status_t loom_tool_temp_file_initialize_platform(
    const char* stem, loom_tool_temp_file_t* out_file) {
  (void)stem;
  memset(out_file, 0, sizeof(*out_file));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "external tool temporary files are unsupported on this platform");
}

iree_status_t loom_tool_temp_file_deinitialize_platform(
    loom_tool_temp_file_t* file) {
  if (file == NULL) {
    return iree_ok_status();
  }
  memset(file, 0, sizeof(*file));
  return iree_ok_status();
}

#endif  // Unsupported platforms
