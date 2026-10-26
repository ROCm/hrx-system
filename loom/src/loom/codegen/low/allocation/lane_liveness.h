// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LANE_LIVENESS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LANE_LIVENESS_H_

#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/cfg_loop_nest.h"

#ifdef __cplusplus
extern "C" {
#endif

// Construction-only cyclic storage facts. Ordinary sparse reservations still
// describe active-lane interference; these bounds also protect retired lanes
// from writes that temporarily expand the execution mask.
typedef struct loom_low_allocation_lane_liveness_t {
  // Borrowed semantic loops, independent of block layout.
  const loom_cfg_loop_nest_t* nest;
  // Scratch-owned loop extents and ancestor jumps in canonical loop order.
  struct loom_low_allocation_lane_loop_t* loops;
  // Scratch-owned defining loop per value, or NONE for unmasked/non-loop
  // values. NULL when the function has no cyclic lane-storage obligations.
  uint16_t* defining_loops;
} loom_low_allocation_lane_liveness_t;

// Builds loop layout extents and indexes definitions once. Scratch costs
// O(V+L); targets without masked classes and acyclic functions allocate none.
iree_status_t loom_low_allocation_lane_liveness_initialize(
    const loom_low_descriptor_set_t* descriptors,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_cfg_loop_nest_t* nest,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_lane_liveness_t* out_lane_liveness);

// Retains the observed units throughout every defining loop escaped by this
// read. Queries the existing nest in O(log L), without revisiting IR or CFG.
// Updates physical bounds only; ordinary sparse reservations remain intact.
void loom_low_allocation_lane_liveness_observe(
    const loom_low_allocation_lane_liveness_t* lane_liveness,
    loom_value_ordinal_t value_ordinal, uint16_t block_index,
    uint32_t unit_point_start, uint32_t unit_count,
    loom_low_allocation_unit_liveness_t* unit_liveness);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LANE_LIVENESS_H_
