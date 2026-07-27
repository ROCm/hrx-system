// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loop-header assignment relocation toward recurrent backedge payloads.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LOOP_EDGE_RELOCATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LOOP_EDGE_RELOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_loop_edge_relocation_context_t {
  // Module containing the allocated function body.
  const loom_module_t* module;
  // Allocated function body region.
  const loom_region_t* body;
  // CFG graph for body.
  const loom_cfg_graph_t* cfg_graph;
  // Target descriptor set defining register storage aliases.
  const loom_low_descriptor_set_t* descriptor_set;
  // Liveness facts owning assignment value ordinals and sparse segments.
  const loom_liveness_analysis_t* liveness;
  // Function-local placement relations.
  const loom_low_placement_table_t* placement;
  // Mutable target constraints whose assignment extents are rebuilt.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Per-unit storage end points used by conflict checks.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Materialized storage leases from the initial assignment.
  const loom_low_allocation_storage_lease_state_t* storage_leases;
  // Mutable completed assignment table.
  loom_low_allocation_assignment_t* assignments;
  // Number of entries in assignments.
  iree_host_size_t assignment_count;
  // Assignment indices by liveness-local value ordinal.
  const uint32_t* assignment_indices_by_value_ordinal;
  // Arena used for relocation candidates and consumption-query scratch.
  iree_arena_allocator_t* arena;
} loom_low_allocation_loop_edge_relocation_context_t;

typedef struct loom_low_allocation_loop_edge_relocation_result_t {
  // Number of loop-header values moved to their backedge source locations.
  iree_host_size_t relocated_value_count;
  // Number of non-header values recolored into storage vacated by headers.
  iree_host_size_t recolored_value_count;
  // Number of loop headers whose assignment layout changed.
  iree_host_size_t relocated_header_count;
} loom_low_allocation_loop_edge_relocation_result_t;

// Relocates safe scalar loop-header assignments to recurrent low.br backedge
// source locations. Relocations form a dependency-closed subset: a destination
// remains unchanged when moving another value depends on it vacating storage
// but its own relocation cannot preserve allocation, alias, or lease
// invariants.
iree_status_t loom_low_allocation_loop_edge_relocate(
    const loom_low_allocation_loop_edge_relocation_context_t* context,
    loom_low_allocation_loop_edge_relocation_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LOOP_EDGE_RELOCATION_H_
