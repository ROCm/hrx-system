// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel request publication from planned command schedules.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/cmd/lower/launch_count_realization.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One planned root schedule and its per-command requirement projection.
typedef struct loom_cmd_program_kernel_site_root_t {
  // Planned command schedule borrowing the plan module.
  const loom_cmd_schedule_plan_t* schedule;

  // Scratch table indexed by schedule command ordinal and initialized to
  // UINT32_MAX. Source-backed and generated commands receive their plan
  // requirement; external commands retain UINT32_MAX for ordinary declaration
  // binding.
  uint32_t* requirement_indices;
} loom_cmd_program_kernel_site_root_t;

// Compact source facts retained only until command dependency planning ends.
typedef struct loom_cmd_program_kernel_accesses_t {
  // Plan-wide requirement projection; NULL rows are opaque external entries.
  const loom_storage_access_summary_t** by_requirement;
  // Storage whose lifetime extends beyond each publication scratch rewind.
  iree_arena_allocator_t* arena;
} loom_cmd_program_kernel_accesses_t;

// Publishes source-backed kernel classes and assigns their plan requirements.
//
// Schedule rows and the existing source fact table are the authoritative site
// inventory. The function performs no IR walk or symbol-name lookup. Sites are
// grouped by their configured target entry symbol so all roots contribute to
// one closed bounded class collection before its first class request is
// visited.
// Per-kernel analysis storage is rewound after its site projection is complete,
// bounding scratch usage by the largest live kernel rather than their sum.
iree_status_t loom_cmd_program_plan_publish_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* plan_module,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_program_kernel_source_t* kernel_source,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    loom_cmd_program_kernel_accesses_t* accesses,
    iree_arena_allocator_t* scratch_arena);

// Publishes generated count-producer sources and assigns their requirements.
//
// The dense generated-source projection makes every schedule lookup O(1).
// Repeated roots and sites reuse the same plan-wide requirement. Root
// requirement tables must already be allocated and initialized to UINT32_MAX.
iree_status_t loom_cmd_program_plan_publish_generated_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* plan_module,
    const loom_cmd_launch_count_realization_t* realization,
    loom_cmd_program_kernel_request_sink_t sink,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    loom_cmd_program_kernel_accesses_t* accesses,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_
