// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_CMD_HAL_COMMAND_BUFFER_H_
#define LOOM_TARGET_ARCH_CMD_HAL_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implementation of one program-local kernel requirement, resolved at load
// time.
typedef struct loom_cmd_program_hal_entry_t {
  // Borrowed executable loaded for the command buffer's exact queue family.
  iree_hal_executable_t* executable;
  // Executable-local function implementing the requirement.
  iree_hal_executable_function_t function;
} loom_cmd_program_hal_entry_t;

// Materializes a parsed portable program into a reusable HAL command buffer.
// This cold path combines logical command arguments with executable reflection;
// it does not inspect source IR or assign meanings to the program's buffer
// slots. |fixed_bindings| and |entries| are in the program's dense requirement
// order. Fixed binding ranges are the roots against which artifact-relative
// command ranges are resolved. Successful recording retains their buffers and
// the executables. Artifact bytes and input arrays may be released when this
// returns. Rebindable slots,
// including the transient slab, must be supplied at issue time; recording
// allocates none of their backing storage. |out_command_buffer| is untouched on
// failure. |mode| selects HAL recording policy, including retained profiling
// metadata when a caller intends to profile these reusable commands.
// A non-NULL range selects complete contiguous barrier waves. NULL selects the
// complete program. Ranges retain the original binding/parameter layout; the
// caller preserves live workspace and orders successive ranges with semaphores.
// Every recording acquires external payload visibility before its first wave
// and releases writes after its last wave. This includes indirect count
// buffers; queue execution provides completion ordering without inferring cache
// actions.
// Timestamps use the queue-family clock and preserve planned capture ordering.
// Sample delivery follows explicit memory dependencies or normal completion;
// unsupported queue capabilities fail through the HAL recording boundary.
iree_status_t loom_cmd_program_hal_command_buffer_create(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode, const loom_cmd_program_t* program,
    const loom_cmd_program_command_range_t* range,
    iree_host_size_t fixed_buffer_count,
    const iree_hal_buffer_binding_t* fixed_bindings,
    iree_host_size_t entry_count, const loom_cmd_program_hal_entry_t* entries,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_HAL_COMMAND_BUFFER_H_
