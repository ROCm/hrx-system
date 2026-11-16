// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Schedule execution for target-low function bodies.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_

#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Schedules one modeled target-low function body and writes an arena-owned
// table. Scheduler working storage is released before returning; the table
// borrows no frontier, pressure, or dependency-index scratch. |model| must
// remain live and its function semantically immutable until this function
// returns, and its resolved target must select a machine schedule model. The
// model's arena-owned results must outlive consumers of the returned table.
// Invalid input constraints are diagnosed through |options->emitter| during
// construction. Schedule failures and feedback remain in the returned table;
// callers emit them with loom_low_schedule_diagnostics_emit after accepting a
// schedule or exhausting repair. All rejections increment error_count; status
// failures are reserved for infrastructure failures.
//
// A schedule(locked) function or kernel preserves authored order within each
// block under every strategy. Its nodes retain source-order boundaries for
// downstream packet formation; dependencies and target hazards still apply.
iree_status_t loom_low_schedule_function(
    const loom_low_function_model_t* model,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_
