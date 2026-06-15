// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packet-local parallel move scratch planning.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_

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

// Immutable allocation facts needed to plan packet-local move temporaries.
typedef struct loom_low_allocation_packet_move_context_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
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
  // Completed assignment lookup map.
  loom_low_allocation_assignment_map_t assignment_map;
} loom_low_allocation_packet_move_context_t;

// Packet-local move temporary table rows.
typedef struct loom_low_allocation_packet_move_plan_t {
  // Per-packet groups indexing |temporaries|.
  loom_low_allocation_packet_move_temporary_group_t* groups;
  // Number of records in |groups|.
  iree_host_size_t group_count;
  // Scratch units reserved for cyclic packet-local move groups.
  loom_low_allocation_packet_move_temporary_t* temporaries;
  // Number of records in |temporaries|.
  iree_host_size_t temporary_count;
} loom_low_allocation_packet_move_plan_t;

// Builds packet-local temporary groups for cyclic copy/slice/concat moves.
iree_status_t loom_low_allocation_packet_move_plan_build(
    const loom_low_allocation_packet_move_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_packet_move_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PACKET_MOVE_H_
