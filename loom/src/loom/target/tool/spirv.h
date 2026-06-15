// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Installed SPIR-V toolchain runner.
//
// Loom keeps SPIR-V validation and disassembly outside the primary compiler
// path. This layer is the narrow argv-based boundary used by tests and target
// debugging tools when they need SPIRV-Tools evidence.

#ifndef LOOM_TARGET_SPIRV_TOOL_H_
#define LOOM_TARGET_SPIRV_TOOL_H_

#include "iree/base/api.h"
#include "loom/target/tool/process.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_tool_kind_e {
  LOOM_SPIRV_TOOL_SPIRV_AS = 0,
  LOOM_SPIRV_TOOL_SPIRV_DIS = 1,
  LOOM_SPIRV_TOOL_SPIRV_VAL = 2,
  LOOM_SPIRV_TOOL_COUNT = 3,
} loom_spirv_tool_kind_t;

typedef struct loom_spirv_toolchain_t {
  // Directory containing SPIR-V tool executables, or empty to search PATH.
  iree_string_view_t root_path;
  // Explicit executable paths indexed by loom_spirv_tool_kind_t.
  iree_string_view_t tool_paths[LOOM_SPIRV_TOOL_COUNT];
} loom_spirv_toolchain_t;

typedef loom_tool_output_t loom_spirv_tool_output_t;

// Initializes |out_toolchain| from LOOM_SPIRV_AS, LOOM_SPIRV_DIS,
// LOOM_SPIRV_VAL, LOOM_SPIRV_TOOLCHAIN_ROOT, and LOOM_SPIRV_TOOLCHAIN_BIN,
// falling back to PATH lookup for tools without an explicit path or root.
void loom_spirv_toolchain_initialize_from_environment(
    loom_spirv_toolchain_t* out_toolchain);

// Returns the executable basename for |tool_kind|, without a toolchain root.
iree_string_view_t loom_spirv_tool_name(loom_spirv_tool_kind_t tool_kind);

// Invokes a SPIR-V tool with argv-style arguments and captures stdout/stderr.
//
// This function does not use shell command strings. A nonzero child exit code
// is represented in |out_result| and still returns OK: launch/capture failures
// are status failures, tool diagnostics are child process results.
iree_status_t loom_spirv_tool_run(const loom_spirv_toolchain_t* toolchain,
                                  loom_spirv_tool_kind_t tool_kind,
                                  const iree_string_view_t* arguments,
                                  iree_host_size_t argument_count,
                                  iree_allocator_t allocator,
                                  loom_tool_process_result_t* out_result);

// Releases output bytes allocated by SPIR-V tool helpers.
void loom_spirv_tool_output_deinitialize(loom_spirv_tool_output_t* output,
                                         iree_allocator_t allocator);

// Runs `<tool> --version` and returns captured stdout in |out_version_text|.
iree_status_t loom_spirv_tool_query_version(
    const loom_spirv_toolchain_t* toolchain, loom_spirv_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_spirv_tool_output_t* out_version_text);

// Writes |binary| to a temporary file, runs `spirv-dis <temp> -o -`, and
// returns textual SPIR-V disassembly.
iree_status_t loom_spirv_tool_disassemble_binary(
    const loom_spirv_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator, loom_spirv_tool_output_t* out_text);

// Writes |binary| to a temporary file and runs `spirv-val <temp>`.
iree_status_t loom_spirv_tool_validate_binary(
    const loom_spirv_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_SPIRV_TOOL_H_
