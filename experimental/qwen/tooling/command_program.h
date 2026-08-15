// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_
#define EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_

#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/target/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Compiled and materialized command-program root.
typedef struct qwen_tooling_command_program_t qwen_tooling_command_program_t;

// Source and root selection used to prepare one command program.
typedef struct qwen_tooling_command_program_options_t {
  // Diagnostic identifier for the borrowed source contents.
  iree_string_view_t source_identifier;
  // Borrowed sealed Loom module containing the selected root and its kernels.
  iree_const_byte_span_t source_contents;
  // Public command-program root to prepare and materialize.
  iree_string_view_t root_name;
} qwen_tooling_command_program_options_t;

// Compiles and materializes one sealed command-program root for the runtime
// context's device. Immutable parameters are gathered into a fixed device
// buffer while required kernel units compile, then both are retained by the
// returned reusable command buffer.
iree_status_t qwen_tooling_command_program_create(
    qwen_tooling_runtime_context_t* runtime_context,
    const qwen_tooling_command_program_options_t* options,
    iree_allocator_t host_allocator,
    qwen_tooling_command_program_t** out_program);

// Releases a command program. Passing NULL is allowed.
void qwen_tooling_command_program_release(
    qwen_tooling_command_program_t* program);

// Returns immutable ABI metadata borrowed from program.
const loomc_cmd_program_info_t* qwen_tooling_command_program_info(
    const qwen_tooling_command_program_t* program);

// Returns the reusable command buffer borrowed from program.
iree_hal_command_buffer_t* qwen_tooling_command_program_command_buffer(
    const qwen_tooling_command_program_t* program);

// Returns the packed immutable parameter size in bytes.
iree_device_size_t qwen_tooling_command_program_parameter_byte_length(
    const qwen_tooling_command_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_
