// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block schedule publication and retention across scoped transformations.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_BLOCK_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_BLOCK_H_

#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/pressure.h"

#ifdef __cplusplus
extern "C" {
#endif

// Starts a block's output slices and resets block-local issue resources.
void loom_low_schedule_block_begin(loom_low_schedule_build_state_t* state,
                                   uint32_t block_index);

// Publishes an already selected node's order, issue group, and pair uses.
// Pressure and descriptor effects are published separately by the caller.
void loom_low_schedule_block_append(loom_low_schedule_build_state_t* state,
                                    uint32_t node_index, uint32_t issue_cycle);

// Closes the current block's decision slice.
void loom_low_schedule_block_finish(loom_low_schedule_build_state_t* state);

// Materializes an unchanged block after begin, including its original
// diagnostics, effects, and pressure contributions, then finishes the block.
// The transform owner establishes the retained-block contract in types.h.
iree_status_t loom_low_schedule_block_retain(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_table_t* previous);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_BLOCK_H_
