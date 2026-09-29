// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_COMMAND_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_COMMAND_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implementation of one program-local kernel requirement, resolved at load
// time.
typedef struct loom_serve_command_entry_t {
  // Borrowed executable loaded for the command buffer's exact queue family.
  iree_hal_executable_t* executable;
  // Executable-local function implementing the requirement.
  iree_hal_executable_function_t function;
} loom_serve_command_entry_t;

// Materializes a parsed portable program into a reusable HAL command buffer.
// This cold path combines logical command arguments with executable reflection;
// it does not inspect source IR or assign meanings to the program's buffer
// slots. |fixed_buffers| and |entries| are in the program's dense requirement
// order. Successful recording retains fixed buffers and executables. Artifact
// bytes and input arrays may be released when this returns. Rebindable slots,
// including the transient slab, must be supplied at issue time; recording
// allocates none of their backing storage. |out_command_buffer| is untouched on
// failure.
iree_status_t loom_serve_command_create(
    const iree_hal_queue_family_t* queue_family,
    const loom_cmd_program_t* program, iree_host_size_t fixed_buffer_count,
    iree_hal_buffer_t* const* fixed_buffers, iree_host_size_t entry_count,
    const loom_serve_command_entry_t* entries, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_COMMAND_H_
