// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal external process runner for compiler-adjacent tools.
//
// This layer is intentionally argv-based and never uses a shell command string.
// Callers provide the executable path and arguments as structured string views;
// stdout/stderr are captured as byte strings owned by the caller allocator.

#ifndef LOOM_TARGET_TOOL_PROCESS_H_
#define LOOM_TARGET_TOOL_PROCESS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_tool_output_t {
  // Captured bytes allocated from the caller allocator, or NULL when empty.
  char* data;
  // Number of bytes in |data|, excluding the trailing NUL terminator.
  iree_host_size_t length;
} loom_tool_output_t;

typedef struct loom_tool_process_result_t {
  // Platform exit code, including signal-derived POSIX codes when applicable.
  int exit_code;
  // Captured stdout bytes.
  loom_tool_output_t stdout_bytes;
  // Captured stderr bytes.
  loom_tool_output_t stderr_bytes;
} loom_tool_process_result_t;

typedef struct loom_tool_temp_file_t {
  // Temporary filesystem path. Valid until deinitialize unlinks it.
  char path[4096];
} loom_tool_temp_file_t;

// Releases output bytes allocated by loom_tool_process_run.
void loom_tool_output_deinitialize(loom_tool_output_t* output,
                                   iree_allocator_t allocator);

// Converts Windows-style newline sequences in textual |output| to LF in place.
// Repeated carriage returns before LF are collapsed because some tools emit
// CRLF through a text-mode stdout that inserts a second carriage return. Other
// bytes, including carriage returns not followed by LF, are preserved.
void loom_tool_output_normalize_newlines(loom_tool_output_t* output);

// Releases stdout/stderr bytes allocated by loom_tool_process_run.
void loom_tool_process_result_deinitialize(loom_tool_process_result_t* result,
                                           iree_allocator_t allocator);

// Returns true when |result| records a zero process exit code.
bool loom_tool_process_result_succeeded(
    const loom_tool_process_result_t* result);

// Invokes |executable_path| with argv-style |arguments| and captures both
// stdout and stderr. The child reads stdin from the platform null device and
// the spawn policy restricts inheritance to that stdin and the redirected
// stdout/stderr handles or file descriptors.
//
// |search_path| controls PATH lookup. Exact execution passes the executable as
// the Win32 application name or uses posix_spawn instead of posix_spawnp.
//
// A nonzero child exit code is represented in |out_result| and still returns
// OK: launch/capture failures are status failures, tool diagnostics are child
// process results.
iree_status_t loom_tool_process_run(iree_string_view_t executable_path,
                                    bool search_path,
                                    const iree_string_view_t* arguments,
                                    iree_host_size_t argument_count,
                                    iree_allocator_t allocator,
                                    loom_tool_process_result_t* out_result);

// Initializes a temporary filesystem path and creates an empty file there. The
// caller may rewrite the file before passing the path to a tool.
iree_status_t loom_tool_temp_file_initialize(iree_string_view_t stem,
                                             loom_tool_temp_file_t* out_file);

// Returns the current filesystem path for |file|.
iree_string_view_t loom_tool_temp_file_path(const loom_tool_temp_file_t* file);

// Deletes the temporary file if it still exists and deinitializes |file|.
// Returns a failure if the file could not be deleted.
iree_status_t loom_tool_temp_file_deinitialize(loom_tool_temp_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TOOL_PROCESS_H_
