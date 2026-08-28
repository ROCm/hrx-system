// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel request publication from prepared command schedules.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One prepared root schedule and its per-command requirement projection.
typedef struct loom_cmd_program_kernel_site_root_t {
  // Prepared command schedule borrowing the preparation module.
  const loom_cmd_schedule_plan_t* schedule;

  // Scratch table indexed by schedule command ordinal. Source-backed commands
  // receive their class-specific plan requirement; external commands retain
  // UINT32_MAX for ordinary declaration binding.
  uint32_t* requirement_indices;
} loom_cmd_program_kernel_site_root_t;

// Publishes source-backed kernel classes and assigns their plan requirements.
//
// Schedule rows and the existing source fact table are the authoritative site
// inventory. The function performs no IR walk or symbol-name lookup. Sites are
// grouped by their configured target entry symbol so all roots contribute to
// one closed bounded class collection before its first product is transferred.
// Per-kernel analysis storage is rewound after its site projection is complete,
// bounding scratch usage by the largest live kernel rather than their sum.
iree_status_t loom_cmd_program_plan_publish_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* preparation_module,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_program_kernel_source_t* kernel_source,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_REQUESTS_H_
