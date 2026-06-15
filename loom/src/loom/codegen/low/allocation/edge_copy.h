// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Branch edge-copy planning and scratch selection.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable allocation facts needed to plan branch edge copies.
typedef struct loom_low_allocation_edge_copy_context_t {
  // Body region of the low function.
  loom_region_t* body;
  // Descriptor set selected by the low function target.
  const loom_low_descriptor_set_t* descriptor_set;
  // Liveness ordering used by allocation.
  loom_liveness_order_t liveness_order;
  // Mutable target storage constraints and diagnostic state.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Per-allocation-unit live end points.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Function-local placement relations over |assignment_map.liveness|.
  const loom_low_placement_table_t* placement;
  // Completed assignment lookup map.
  loom_low_allocation_assignment_map_t assignment_map;
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
  // Scratch units reserved for cyclic edge-copy groups.
  loom_low_allocation_edge_copy_temporary_t* temporaries;
  // Number of records in |temporaries|.
  iree_host_size_t temporary_count;
} loom_low_allocation_edge_copy_plan_t;

// Builds branch edge-copy groups and scratch temporaries for low.br payloads.
iree_status_t loom_low_allocation_edge_copy_plan_build(
    const loom_low_allocation_edge_copy_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_edge_copy_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_EDGE_COPY_H_
