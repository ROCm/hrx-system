// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_HAL_H_
#define LOOMC_TARGET_CMD_HAL_H_

#include "iree/hal/api.h"
#include "loomc/target/cmd/program.h"

/// @file
/// Reusable IREE HAL command buffers from portable command programs.
///
/// This optional runtime leaf combines one loaded command-package export with
/// application-owned live HAL executables and fixed buffers. Materialization
/// resolves executable entries, reflects their argument layouts, and records
/// one reusable command buffer. It performs no compilation, cache lookup, or
/// source-module access.

#ifdef __cplusplus
extern "C" {
#endif

/// Required byte alignment for HAL command-program config-data placement.
///
/// A package reports the exact config-data byte length. HAL integrations place
/// each payload at a 64-byte-aligned offset and may round the exact length up
/// to this value when computing a ring-buffer stride.
#define LOOMC_CMD_HAL_CONFIG_ALIGNMENT 64

/// Materialized reusable HAL command program.
///
/// The object owns one recorded HAL command buffer and no compiler state. The
/// package used to create it may be released after materialization completes.
typedef struct loomc_cmd_hal_program_t loomc_cmd_hal_program_t;

/// Options and materialization-time resources for one HAL command program.
typedef struct loomc_cmd_hal_program_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Mode flags used to create the reusable HAL command buffer.
  ///
  /// `IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT` retains fixed buffers and
  /// executables for the command-buffer lifetime. When
  /// `IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED` is set, the caller must preserve
  /// those resources according to the HAL command-buffer contract.
  iree_hal_command_buffer_mode_t command_buffer_mode;

  /// Queue set on which the command buffer will execute.
  iree_hal_queue_affinity_t queue_affinity;

  /// Fixed buffer ranges baked into the command buffer in package slot order.
  const iree_hal_buffer_ref_t* fixed_buffers;

  /// Number of entries in `fixed_buffers`.
  ///
  /// This must exactly match the selected export's `fixed_buffer_count`.
  loomc_host_size_t fixed_buffer_count;

  /// Loaded executables in the selected export's root-local slot order.
  iree_hal_executable_t* const* executables;

  /// Number of entries in `executables`.
  ///
  /// This must exactly match the selected export's `executable_count`.
  loomc_host_size_t executable_count;
} loomc_cmd_hal_program_options_t;

/// Materializes one package export as a reusable HAL command program.
///
/// Every package entry is resolved by name against its root-local executable,
/// and its logical argument schema is checked against the loaded HAL function
/// ABI before recording. Fixed buffers and executables are retained by the
/// command buffer unless `options->command_buffer_mode` requests unretained
/// recording. Issue-time buffers remain indirect binding-table slots.
///
/// @param package Loaded portable command-program package.
/// @param program_export Package-local export token to materialize.
/// @param device HAL device that creates the command buffer.
/// @param options Materialization options and root-local resource tables.
/// @param allocator Host allocator used for persistent program storage and
/// transient materialization scratch.
/// @param out_program Receives one retained program on success.
/// @return OK after the complete command buffer has been recorded.
///
/// @ownership
/// The returned program owns its command buffer. It retains neither `package`
/// nor the options descriptor. The package may be released immediately after
/// this call succeeds. Default HAL recording retains the live executable and
/// fixed-buffer resources, so the caller's references may also be released.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_hal_program_create(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, iree_hal_device_t* device,
    const loomc_cmd_hal_program_options_t* options, loomc_allocator_t allocator,
    loomc_cmd_hal_program_t** out_program);

/// Retains `program` for another owner.
LOOMC_API_EXPORT void loomc_cmd_hal_program_retain(
    loomc_cmd_hal_program_t* program);

/// Releases `program` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_cmd_hal_program_release(
    loomc_cmd_hal_program_t* program);

/// Returns the borrowed reusable command buffer, or `NULL` for `NULL`.
///
/// The returned reference remains live until the program is released. Callers
/// that retain the command buffer independently may use it afterward.
LOOMC_API_EXPORT iree_hal_command_buffer_t*
loomc_cmd_hal_program_command_buffer(const loomc_cmd_hal_program_t* program);

/// Returns the exact issue-time binding-table capacity, or zero for `NULL`.
LOOMC_API_EXPORT loomc_host_size_t
loomc_cmd_hal_program_binding_count(const loomc_cmd_hal_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_HAL_H_
