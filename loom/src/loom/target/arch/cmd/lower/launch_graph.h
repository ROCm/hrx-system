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
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Placement of one source launch's workgroup-count tuple.
typedef enum loom_cmd_launch_count_kind_e {
  // All three dimensions are exact package values recorded directly.
  LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT = 1,
  // The aggregate host function produces one issue-time tuple.
  LOOM_CMD_LAUNCH_COUNT_KIND_HOST = 2,
} loom_cmd_launch_count_kind_t;

// Aggregate launch-count assignment for one source kernel.launch.
typedef struct loom_cmd_launch_count_t {
  // Source launch represented by this row.
  const loom_op_t* source_op;
  // Placement selecting the populated payload member.
  loom_cmd_launch_count_kind_t kind;
  union {
    // Exact workgroup count when |kind| is DIRECT.
    loom_target_dispatch_workgroup_count_t direct;
    // Dense xyz tuple ordinal returned by the host function when HOST.
    uint32_t host_tuple_ordinal;
  } payload;
} loom_cmd_launch_count_t;

// Kernel definitions resolved across every root in one command program plan.
typedef struct loom_cmd_launch_definition_table_t {
  // Unique kernel definitions in first scheduled occurrence order.
  loom_op_t* const* entries;
  // Number of entries in |entries|.
  uint32_t count;
} loom_cmd_launch_definition_table_t;

// Definition assignments for one command root's scheduled launches.
//
// |definition_ordinals| is aligned with the corresponding root schedule's
// command table. The owning program plan establishes both tables while
// validating scheduled launch symbols, so launch consumers never repeat symbol
// resolution.
typedef struct loom_cmd_launch_resolution_t {
  // Plan-wide kernel definition table.
  const loom_cmd_launch_definition_table_t* definitions;
  // Definition ordinal for each command in the corresponding root schedule.
  const uint32_t* definition_ordinals;
} loom_cmd_launch_resolution_t;

// Source inputs for one command root's aggregate launch graph.
typedef struct loom_cmd_launch_program_source_t {
  // Source command program definition.
  loom_op_t* program_op;
  // Existing schedule defining source launch order.
  const loom_cmd_schedule_plan_t* schedule;
  // Resolved kernel definitions aligned with |schedule|.
  const loom_cmd_launch_resolution_t* resolution;
} loom_cmd_launch_program_source_t;

// Factored launch graph for one source command program.
//
// The containing program plan's shared launch module owns one pure host
// func.def whose arguments correspond to the source program's specialization
// arguments. Its results are flattened xyz values for |host_tuple_count|
// unique dynamic tuples. Static tuples have no function results and remain in
// DIRECT launch rows.
//
// Launch rows retain source-op pointers, so the source module must outlive this
// graph's planning consumers. All other storage is owned by the shared launch
// module. Wave rows are copied from the source schedule so later consumers do
// not need to rediscover command ordering.
typedef struct loom_cmd_launch_graph_t {
  // Pure aggregate host function inside the shared launch module.
  loom_op_t* host_function_op;
  // Source launches in schedule traversal order.
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

// Materializes one shared launch-count program for |sources|.
//
// One private pure helper represents each plan-wide resolved kernel launch
// configuration. Each scheduled launch becomes an ordinary pure func.call to
// its helper. Generic CSE merges equal calls before required inlining expands
// the remaining computations into each root host function. Canonicalization,
// CSE, and DCE then classify exact tuples and compact equal residual tuples.
// Launch factoring is therefore expressed entirely through ordinary IR and
// generic transformations shared with other compiler pipelines.
//
// |source_facts| is populated for every source program and resolved kernel
// launch-config region by the owning program plan. |sources| contains at least
// one root, and |out_graphs| must provide |source_count| entries. On success
// |out_module| owns all graph storage and must be released by the caller.
//
// This host slice accepts only workload values derived from command-program
// specialization arguments and pure scalar operations. Buffer-sourced values
// require the separately placed device launch slice and fail closed here.
iree_status_t loom_cmd_launch_program_materialize(
    const loom_module_t* source_module,
    const loom_cmd_launch_program_source_t* sources,
    iree_host_size_t source_count, const loom_value_fact_table_t* source_facts,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module, loom_cmd_launch_graph_t* out_graphs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_GRAPH_H_
