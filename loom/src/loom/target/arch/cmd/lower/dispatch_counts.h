// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical workgroup-count placement for portable command dispatches.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_DISPATCH_COUNTS_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_DISPATCH_COUNTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Placement of one dispatch's physical workgroup-count payload.
typedef enum loom_cmd_dispatch_count_kind_e {
  // All dimensions are exact unsigned 32-bit artifact values.
  LOOM_CMD_DISPATCH_COUNT_KIND_DIRECT = 0,
  // A stable source view supplies the tuple before command execution.
  LOOM_CMD_DISPATCH_COUNT_KIND_INDIRECT_STATIC = 1,
  // A transient source view receives the tuple during command execution.
  LOOM_CMD_DISPATCH_COUNT_KIND_INDIRECT_DYNAMIC = 2,
} loom_cmd_dispatch_count_kind_t;

// One prepared workgroup-count placement in schedule command order.
typedef struct loom_cmd_dispatch_count_t {
  // Placement selecting the populated payload member.
  loom_cmd_dispatch_count_kind_t kind;
  union {
    // Exact workgroup count when |kind| is DIRECT.
    loom_target_dispatch_workgroup_count_t direct;
    // Source view when |kind| is either INDIRECT kind.
    loom_value_id_t indirect_source_value;
  } payload;
} loom_cmd_dispatch_count_t;

// Classifies physical workgroup-count placement for every scheduled dispatch.
//
// The schedule must contain only configured kernel.dispatch commands. Exact
// scalar tuples become direct artifact metadata. Exactly placed views rooted in
// command inputs or immutable parameters remain static indirect, while views
// rooted in command buffer.alloca storage become dynamic indirect and require
// an earlier execution wave. No kernel implementation or launch-configuration
// body is inspected or cloned.
//
// Rows are allocated from |arena| in schedule command order. User-authored
// contract failures emit diagnostics, set |out_valid| to false, and return OK.
// Infrastructure failures return non-OK.
iree_status_t loom_cmd_dispatch_count_table_build(
    const loom_module_t* module, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_cmd_dispatch_count_t** out_counts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_DISPATCH_COUNTS_H_
