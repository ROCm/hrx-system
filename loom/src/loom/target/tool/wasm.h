// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Installed WebAssembly tool runner.
//
// Loom keeps Wasm binary disassembly outside the compiler path. This layer is
// the narrow argv-based boundary used by tests and target debugging tools when
// they need evidence from a real Wasm consumer.

#ifndef LOOM_TARGET_WASM_TOOL_H_
#define LOOM_TARGET_WASM_TOOL_H_

#include "iree/base/api.h"
#include "loom/target/tool/process.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_tool_kind_e {
  LOOM_WASM_TOOL_LLVM_OBJDUMP = 0,
} loom_wasm_tool_kind_t;

typedef struct loom_wasm_toolchain_t {
  // Exact llvm-objdump path, or empty to use |root_path|/PATH lookup.
  iree_string_view_t llvm_objdump_path;
  // Directory containing Wasm tool executables, or empty to search PATH.
  iree_string_view_t root_path;
} loom_wasm_toolchain_t;

typedef loom_tool_output_t loom_wasm_tool_output_t;

// Initializes |out_toolchain| from LOOM_WASM_LLVM_OBJDUMP first, then
// LOOM_WASM_TOOLCHAIN_ROOT, then LOOM_WASM_TOOLCHAIN_BIN. When no environment
// variable is set the tool runner searches PATH.
void loom_wasm_toolchain_initialize_from_environment(
    loom_wasm_toolchain_t* out_toolchain);

// Returns the executable basename for |tool_kind|, without a toolchain root.
iree_string_view_t loom_wasm_tool_name(loom_wasm_tool_kind_t tool_kind);

// Invokes a Wasm tool with argv-style arguments and captures stdout/stderr.
//
// This function does not use shell command strings. A nonzero child exit code
// is represented in |out_result| and still returns OK: launch/capture failures
// are status failures, tool diagnostics are child process results.
iree_status_t loom_wasm_tool_run(const loom_wasm_toolchain_t* toolchain,
                                 loom_wasm_tool_kind_t tool_kind,
                                 const iree_string_view_t* arguments,
                                 iree_host_size_t argument_count,
                                 iree_allocator_t allocator,
                                 loom_tool_process_result_t* out_result);

// Releases output bytes allocated by Wasm tool helpers.
void loom_wasm_tool_output_deinitialize(loom_wasm_tool_output_t* output,
                                        iree_allocator_t allocator);

// Runs `<tool> --version` and returns captured stdout in |out_version_text|.
iree_status_t loom_wasm_tool_query_version(
    const loom_wasm_toolchain_t* toolchain, loom_wasm_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_wasm_tool_output_t* out_version_text);

// Writes |binary| to a temporary file, runs llvm-objdump disassembly, and
// returns textual Wasm disassembly.
iree_status_t loom_wasm_tool_disassemble_binary(
    const loom_wasm_toolchain_t* toolchain, iree_const_byte_span_t binary,
    iree_allocator_t allocator, loom_wasm_tool_output_t* out_text);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_WASM_TOOL_H_
