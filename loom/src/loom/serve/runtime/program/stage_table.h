// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_STAGE_TABLE_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_STAGE_TABLE_H_

#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private storage either belongs to one runner call or spans an invocation's
// barrier-aligned parts. The latter is an explicit final execute_N argument.
typedef enum loom_serve_stage_workspace_e {
  LOOM_SERVE_STAGE_WORKSPACE_RUNNER = 0,
  LOOM_SERVE_STAGE_WORKSPACE_INVOCATION,
} loom_serve_stage_workspace_t;

typedef struct loom_serve_stage_record_options_t {
  // Exact queue family used by every compiled executable and recorded command.
  const iree_hal_queue_family_t* queue_family;
  // Immutable checkpoint roots borrowed only while commands are recorded.
  iree_hal_buffer_t* const* parameters;
  // HAL recording policy, including optional retained profiling metadata.
  iree_hal_command_buffer_mode_t command_mode;
  // Number of source input/output buffers, excluding private workspace.
  uint16_t binding_count;
  // Owner of private storage between command submissions.
  loom_serve_stage_workspace_t workspace;
  // Soft command target for invocation-owned parts; zero records whole roots.
  uint32_t command_quantum;
} loom_serve_stage_record_options_t;

// Source stage ordinals are stable even when recording produces several parts.
typedef struct loom_serve_stage_launch_t {
  // First entry in the complete runner table.
  uint32_t first_part;
  // Number of ordered submissions implementing this source stage.
  uint32_t part_count;
  // Reflected private storage, shared by all parts of this stage.
  loom_cmd_program_transient_requirement_t transient;
} loom_serve_stage_launch_t;

// Immutable recorded table. A request table can borrow a generation's prefix;
// only its appended commands are owned. The prefix must outlive the table.
typedef struct loom_serve_stage_table_t {
  // Allocator owning descriptors and the recorded-part array.
  iree_allocator_t host_allocator;
  // Number of source stages, including the borrowed prefix.
  iree_host_size_t stage_count;
  // Owned invocation-workspace mappings in source declaration order. NULL for
  // runner-owned workspace, where source ordinals directly index parts.
  loom_serve_stage_launch_t* launches;
  // Number of initialized runner entries, including the borrowed prefix.
  iree_host_size_t part_count;
  // Allocated runner-entry capacity, included in retained byte accounting.
  iree_host_size_t part_capacity;
  // First owned part; earlier entries borrow the immutable prefix.
  iree_host_size_t owned_part_begin;
  // Flat runner descriptors; owned suffix retains commands/direct executables.
  loom_serve_stage_t* parts;
} loom_serve_stage_table_t;

// Validates the external source's execution ABI before recording or admission.
// Parameter layout compatibility is checked separately by its residency owner.
iree_status_t loom_serve_stage_record_check(
    const loom_serve_stage_record_options_t* options,
    const loom_serve_jit_stage_t* stage);

// Records checked stages after an optional borrowed prefix. Neither device
// submission nor workspace allocation occurs. Complete barrier waves are never
// split. Failure releases partial recordings and leaves *out_table NULL.
iree_status_t loom_serve_stage_table_allocate(
    const loom_serve_stage_record_options_t* options,
    const loom_serve_stage_table_t* prefix, iree_host_size_t stage_count,
    loom_serve_jit_stage_t* const* stages, iree_allocator_t host_allocator,
    loom_serve_stage_table_t** out_table);

// Releases owned recordings without waiting; accepted queue work independently
// retains its resources. NULL is accepted. Compiler products are not owned
// here.
void loom_serve_stage_table_free(loom_serve_stage_table_t* table);

// Requested descriptor storage, excluding HAL/driver-private command storage.
iree_host_size_t loom_serve_stage_table_storage_size(
    const loom_serve_stage_table_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_STAGE_TABLE_H_
