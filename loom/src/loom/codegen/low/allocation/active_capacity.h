// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed storage bounds for the active assignment calendar and unit index.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_CAPACITY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_CAPACITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_active_capacity_t {
  // One past the last storage end, including dead definition reservations.
  iree_host_size_t program_point_count;
  // Upper bound on simultaneously indexed atomic units, including aliases.
  iree_host_size_t unit_count;
} loom_low_allocation_active_capacity_t;

// Bounds active storage before assignment. Placement-connected values may be
// reserved from their component's earliest definition; each remains charged
// through its own storage end. This includes early coalescing reservations
// without assuming that one physical unit has only one SSA owner.
//
// Work is linear in values, placement relations, units, and program points.
// Temporary storage is released before returning the two scalar bounds.
iree_status_t loom_low_allocation_active_capacity_calculate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_block_pool_t* block_pool,
    loom_low_allocation_active_capacity_t* out_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_CAPACITY_H_
