// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Branch edge-copy planning and final move sequencing.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/move_plan.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable allocation facts needed to plan branch edge copies.
typedef struct loom_low_allocation_edge_copy_context_t {
  // Function-local placement relations over the move plan's liveness.
  const loom_low_placement_table_t* placement;
  // Allocation-wide finalized move rows and sequencing scratch.
  loom_low_allocation_move_plan_t* move_plan;
} loom_low_allocation_edge_copy_context_t;

// Branch edge-copy table rows.
typedef struct loom_low_allocation_edge_copy_plan_t {
  // Edge-copy records grouped by low.br terminator source order.
  loom_low_allocation_edge_copy_t* copies;
  // Number of records in |copies|.
  iree_host_size_t copy_count;
  // Per-low.br groups indexing |copies|.
  loom_low_allocation_edge_copy_group_t* groups;
  // Number of records in |groups|.
  iree_host_size_t group_count;
} loom_low_allocation_edge_copy_plan_t;

// Builds branch edge-copy groups and final physical rows for low.br payloads.
iree_status_t loom_low_allocation_edge_copy_plan_build(
    const loom_low_allocation_edge_copy_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_edge_copy_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_
