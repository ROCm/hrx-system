// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packet-local final move planning.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/move_plan.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable allocation facts needed to finalize packet-local moves.
typedef struct loom_low_allocation_packet_move_context_t {
  // Function-local placement relations over the move plan's liveness.
  const loom_low_placement_table_t* placement;
  // Allocation-wide finalized move rows and sequencing scratch.
  loom_low_allocation_move_plan_t* move_plan;
} loom_low_allocation_packet_move_context_t;

// Packet-local final move groups.
typedef struct loom_low_allocation_packet_move_plan_t {
  // Per-packet groups in source order.
  loom_low_allocation_packet_move_group_t* groups;
  // Number of records in |groups|.
  iree_host_size_t group_count;
  // Number of final physical moves across |groups|.
  iree_host_size_t move_count;
} loom_low_allocation_packet_move_plan_t;

// Builds packet-local groups referencing allocation-owned final move rows.
iree_status_t loom_low_allocation_packet_move_plan_build(
    const loom_low_allocation_packet_move_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_packet_move_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_
