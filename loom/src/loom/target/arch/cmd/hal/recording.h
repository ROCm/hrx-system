// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE HAL recording for portable command programs.

#ifndef LOOM_TARGET_ARCH_CMD_HAL_RECORDING_H_
#define LOOM_TARGET_ARCH_CMD_HAL_RECORDING_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One package entry resolved against a live executable.
typedef struct loom_cmd_hal_entry_t {
  // Root-local executable slot providing the entry.
  uint32_t executable_index;
  // Executable-local function token used by HAL dispatch commands.
  iree_hal_executable_function_t function;
  // Reflected function layout used to pack logical command arguments.
  iree_hal_executable_function_info_t info;
  // Reflected parameters with |info.parameter_count| entries.
  const iree_hal_executable_function_parameter_t* parameters;
} loom_cmd_hal_entry_t;

// Live resources and reflected entry metadata for one program root.
//
// All arrays are borrowed during recording. Direct resources are retained by
// the command buffer according to its recording mode. Rebindable resources are
// represented by issue-time binding ordinals and are not retained.
typedef struct loom_cmd_hal_recording_inputs_t {
  // Direct buffer ranges with |program.requirements.fixed_buffer_count|
  // entries baked into the command buffer.
  const iree_hal_buffer_ref_t* fixed_buffers;
  // Loaded executable table with
  // |program.requirements.executable_count| entries in root-local slot order.
  iree_hal_executable_t* const* executables;
  // Resolved entry table with |program.requirements.entry_count| entries in
  // root-local entry order.
  const loom_cmd_hal_entry_t* entries;
} loom_cmd_hal_recording_inputs_t;

// Records one validated portable command program into a begun command buffer.
//
// |inputs| is prepared from the same validated package export as |program|.
// Its table lengths therefore exactly match the program requirements. The
// operation validates the portable argument schemas against the live HAL entry
// metadata, obtains all scratch storage from one |host_allocator| allocation,
// and performs no per-command allocation or executable query.
iree_status_t loom_cmd_hal_record_program(
    const loom_cmd_program_t* program,
    const loom_cmd_hal_recording_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_HAL_RECORDING_H_
