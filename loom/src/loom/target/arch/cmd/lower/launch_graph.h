// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Aggregate launch-count extraction for one command program.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_GRAPH_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_GRAPH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Placement of one source issue command's workgroup-count tuple.
typedef enum loom_cmd_launch_count_kind_e {
  // All three dimensions are exact package values recorded directly.
  LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT = 1,
  // The aggregate host function produces one issue-time tuple.
  LOOM_CMD_LAUNCH_COUNT_KIND_HOST = 2,
  // A stable source view supplies the tuple before command execution.
  LOOM_CMD_LAUNCH_COUNT_KIND_INDIRECT_STATIC = 3,
  // A transient source view receives the tuple during command execution.
  LOOM_CMD_LAUNCH_COUNT_KIND_INDIRECT_DYNAMIC = 4,
} loom_cmd_launch_count_kind_t;

// Aggregate workgroup-count assignment for one source issue command.
typedef struct loom_cmd_launch_count_t {
  // Source launch or dispatch represented by this row.
  const loom_op_t* source_op;
  // Placement selecting the populated payload member.
  loom_cmd_launch_count_kind_t kind;
  union {
    // Exact workgroup count when |kind| is DIRECT.
    loom_target_dispatch_workgroup_count_t direct;
    // Dense xyz tuple ordinal returned by the host function when HOST.
    uint32_t host_tuple_ordinal;
    // Source view when |kind| is either INDIRECT kind.
    loom_value_id_t indirect_source_value;
  } payload;
} loom_cmd_launch_count_t;

// Factored launch graph for one source command program.
//
// The owned module contains one pure host func.def whose arguments correspond
// to the source program's specialization arguments. Its results are flattened
// xyz values for |host_tuple_count| unique dynamic logical-launch tuples.
// Static tuples have no function results and remain in DIRECT rows. Explicit
// indirect dispatches retain their source view and whether its root exists
// before or during command execution.
//
// Launch rows retain source-op pointers, so the source module must outlive this
// object. All other storage is owned by |module| and released together. Wave
// rows are copied from the source schedule so later consumers do not need to
// rediscover command ordering.
typedef struct loom_cmd_launch_graph_t {
  // Owned module containing the aggregate host function.
  loom_module_t* module;
  // Pure aggregate host function inside |module|.
  loom_op_t* host_function_op;
  // Source issue commands in schedule traversal order.
  const loom_cmd_launch_count_t* launches;
  // Number of entries in |launches|.
  iree_host_size_t launch_count;
  // Ordered wave table covering |launches|.
  const loom_cmd_schedule_wave_t* waves;
  // Number of ordered waves.
  iree_host_size_t wave_count;
  // Number of unique dynamic xyz tuples returned by |host_function_op|.
  uint32_t host_tuple_count;
} loom_cmd_launch_graph_t;

// Materializes the aggregate launch-count graph for |source_program_op|.
//
// Each scheduled kernel's exact launch-config region is cloned with its
// workload operands substituted. Pure command-program dataflow required by
// those operands is cloned once, and ordinary canonicalization plus CSE runs
// across the combined function. Equal residual xyz tuples share one dense host
// result ordinal. Exact tuples are returned as direct dispatch metadata.
// Already configured direct dispatches bypass launch-config cloning and retain
// their exact counts. Configured indirect dispatches retain exactly placed
// source views; command-input roots are static and buffer.alloca roots are
// dynamic. Dynamic roots require an earlier execution wave. |source_facts| is
// a borrowed table populated for the source program and each scheduled kernel's
// launch-config region by the owning program plan.
//
// This host slice accepts only workload values derived from command-program
// specialization arguments and pure scalar operations. Buffer-sourced values
// require the separately placed device launch slice and fail closed here.
iree_status_t loom_cmd_launch_graph_materialize(
    const loom_module_t* source_module, loom_op_t* source_program_op,
    const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    bool* out_valid, loom_cmd_launch_graph_t* out_graph);

// Releases all storage owned by |graph| and resets it to zero.
void loom_cmd_launch_graph_deinitialize(loom_cmd_launch_graph_t* graph);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_GRAPH_H_
