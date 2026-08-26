// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable wave planning for source command programs.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_SCHEDULE_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_SCHEDULE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantic issue-command kind classified during schedule traversal.
typedef enum loom_cmd_schedule_command_kind_e {
  // Logical workload launch requiring configuration and a kernel body.
  LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_LAUNCH = 0,
  // Configured entry dispatch with one to three direct count values.
  LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT = 1,
  // Configured entry dispatch reading one indirect count view.
  LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_INDIRECT = 2,
} loom_cmd_schedule_command_kind_t;

// Borrowed source SSA values retained by one prepared issue row.
typedef struct loom_cmd_schedule_value_slice_t {
  // Source value IDs in authored order.
  const loom_value_id_t* values;
  // Number of entries in |values|.
  uint16_t count;
} loom_cmd_schedule_value_slice_t;

// One issue command classified while traversing the structured schedule.
//
// The row borrows all source storage. Consumers use this prepared shape rather
// than repeatedly identifying operations and decoding their operand segments.
typedef struct loom_cmd_schedule_command_t {
  // Source issue operation.
  const loom_op_t* source_op;
  // Referenced logical kernel or configured entry symbol.
  loom_symbol_ref_t callee;
  // Device-ABI argument values in authored order.
  loom_cmd_schedule_value_slice_t arguments;
  // Classified source issue semantics.
  loom_cmd_schedule_command_kind_t kind;
  // Kind-specific values used to derive physical workgroup counts.
  union {
    // Logical workload values for KERNEL_LAUNCH.
    loom_cmd_schedule_value_slice_t workloads;
    // Physical count values for either KERNEL_DISPATCH kind.
    loom_cmd_schedule_value_slice_t workgroup_counts;
  } count_inputs;
} loom_cmd_schedule_command_t;

// One contiguous wave in a portable command schedule.
//
// Commands in a wave retain source traversal order but have no dependency
// edges between them. Successive waves are separated by an execution barrier.
typedef struct loom_cmd_schedule_wave_t {
  // Offset of the first command in the containing plan's command table.
  iree_host_size_t command_offset;
  // Number of commands in this wave.
  iree_host_size_t command_count;
} loom_cmd_schedule_wave_t;

// One transient allocation definition in a portable command schedule.
typedef struct loom_cmd_schedule_allocation_t {
  // Source buffer.alloca operation defining the storage identity.
  const loom_op_t* op;
  // First wave during which the defined storage identity exists.
  iree_host_size_t definition_wave;
} loom_cmd_schedule_allocation_t;

// Flattened portable schedule for one command-program body.
//
// All storage is owned by the arena passed to loom_cmd_schedule_plan_build.
typedef struct loom_cmd_schedule_plan_t {
  // Source commands grouped contiguously by wave.
  const loom_cmd_schedule_command_t* commands;
  // Total number of source commands.
  iree_host_size_t command_count;
  // Number of logical kernel launch commands.
  iree_host_size_t kernel_launch_count;
  // Total number of device-ABI argument values across all commands.
  iree_host_size_t argument_value_count;
  // Ordered wave table.
  const loom_cmd_schedule_wave_t* waves;
  // Number of ordered waves.
  iree_host_size_t wave_count;
  // Source allocation definitions in traversal order.
  const loom_cmd_schedule_allocation_t* allocations;
  // Number of entries in |allocations|.
  iree_host_size_t allocation_count;
} loom_cmd_schedule_plan_t;

// Builds the portable wave schedule for a command-program body.
//
// Lexical operations in |program_body| and serial launch-schedule regions are
// ordered. Siblings in concurrent launch-schedule regions begin in the same
// wave; nested serial spans are aligned by wave index, which may conservatively
// add cross-sibling dependencies. The kernel.launch and command schedule ops
// are equivalent structured scheduling forms. Allocation definitions retain
// their first possible wave so storage planning can preserve state from
// definition through last use. Pure leaf dataflow is ignored because it emits
// no command; the launch plan owns any values it contributes to dispatch
// metadata. Other residual source operations must have been specialized away
// and are rejected. Issue rows classify launch and direct/indirect dispatch
// forms once so later command planning never needs to rediscover them.
iree_status_t loom_cmd_schedule_plan_build(const loom_module_t* module,
                                           loom_region_t* program_body,
                                           iree_arena_allocator_t* arena,
                                           loom_cmd_schedule_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_SCHEDULE_H_
