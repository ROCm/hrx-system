// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/lane_liveness.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/ir/module.h"

typedef struct loom_low_allocation_lane_loop_t {
  // Half-open point envelope of every block in this semantic loop.
  loom_liveness_segment_t extent;
  // Ancestor reached by the exponentially growing jump, or self at a root.
  uint16_t jump_index;
  // Number of enclosing loops.
  uint16_t depth;
} loom_low_allocation_lane_loop_t;

iree_status_t loom_low_allocation_lane_liveness_initialize(
    const loom_low_descriptor_set_t* descriptors,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_cfg_loop_nest_t* nest,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_lane_liveness_t* out_lane_liveness) {
  *out_lane_liveness = (loom_low_allocation_lane_liveness_t){.nest = nest};
  if (nest->loop_count == 0) {
    return iree_ok_status();
  }
  bool has_masked_class = false;
  for (uint16_t i = 0; i < descriptors->reg_class_count; ++i) {
    has_masked_class |= iree_any_bit_set(descriptors->reg_classes[i].flags,
                                         LOOM_LOW_REG_CLASS_FLAG_LANE_MASKED);
  }
  if (!has_masked_class) {
    return iree_ok_status();
  }

  loom_low_allocation_lane_loop_t* loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, nest->loop_count, sizeof(*loops), (void**)&loops));
  for (iree_host_size_t i = 0; i < nest->loop_count; ++i) {
    loops[i] = (loom_low_allocation_lane_loop_t){
        .extent = {.start_point = UINT32_MAX},
    };
  }
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    const uint16_t loop = loom_cfg_loop_nest_innermost(nest, (uint16_t)i);
    if (loop == LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    loops[loop].extent.start_point = iree_min(loops[loop].extent.start_point,
                                              liveness->blocks[i].start_point);
    loops[loop].extent.end_point = iree_max(loops[loop].extent.end_point,
                                            liveness->blocks[i].end_point + 1u);
  }
  // Children precede parents in the retained nest. Inclusive loop extents
  // therefore require one reduction, not a member walk for each loop.
  for (iree_host_size_t i = 0; i < nest->loop_count; ++i) {
    const uint16_t parent = nest->loops[i].parent_loop_index;
    if (parent == LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    loops[parent].extent.start_point =
        iree_min(loops[parent].extent.start_point, loops[i].extent.start_point);
    loops[parent].extent.end_point =
        iree_max(loops[parent].extent.end_point, loops[i].extent.end_point);
  }
  // Equal consecutive jump lengths combine into the next level. One jump
  // per node gives logarithmic ancestor search with linear construction space.
  for (iree_host_size_t cursor = nest->loop_count; cursor > 0; --cursor) {
    const uint16_t loop = (uint16_t)(cursor - 1u);
    const uint16_t parent = nest->loops[loop].parent_loop_index;
    if (parent == LOOM_CFG_LOOP_NEST_NONE) {
      loops[loop].jump_index = loop;
      continue;
    }
    loops[loop].depth = loops[parent].depth + 1u;
    const uint16_t first = loops[parent].jump_index;
    const uint16_t second = loops[first].jump_index;
    loops[loop].jump_index = loops[parent].depth - loops[first].depth ==
                                     loops[first].depth - loops[second].depth
                                 ? second
                                 : parent;
  }

  uint16_t* defining_loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, liveness->value_count, sizeof(*defining_loops),
      (void**)&defining_loops));
  memset(defining_loops, 0xFF, liveness->value_count * sizeof(*defining_loops));
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, i);
    if (interval == NULL || interval->definition_point == UINT32_MAX ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval) ||
        !iree_any_bit_set(
            descriptors->reg_classes[interval->value_class.register_class_id]
                .flags,
            LOOM_LOW_REG_CLASS_FLAG_LANE_MASKED)) {
      continue;
    }
    const loom_value_t* value =
        loom_module_value(value_domain->module, interval->value_id);
    const loom_block_t* block = loom_value_is_block_arg(value)
                                    ? loom_value_def_block(value)
                                    : loom_value_def_op(value)->parent_block;
    // Flat CFG definitions carry their block index directly. For structured
    // regions, the retained point map identifies the enclosing CFG block.
    const uint16_t block_index =
        block->parent_region == value_domain->region
            ? loom_block_region_index(block)
            : loom_liveness_block_index_at_point(liveness,
                                                 interval->definition_point);
    defining_loops[i] = loom_cfg_loop_nest_innermost(nest, block_index);
  }
  out_lane_liveness->loops = loops;
  out_lane_liveness->defining_loops = defining_loops;
  return iree_ok_status();
}

void loom_low_allocation_lane_liveness_observe(
    const loom_low_allocation_lane_liveness_t* lane_liveness,
    loom_value_ordinal_t value_ordinal, uint16_t block_index,
    uint32_t unit_point_start, uint32_t unit_count,
    loom_low_allocation_unit_liveness_t* unit_liveness) {
  const loom_cfg_loop_nest_t* nest = lane_liveness->nest;
  uint16_t loop = lane_liveness->defining_loops[value_ordinal];
  if (loop == LOOM_CFG_LOOP_NEST_NONE ||
      loom_cfg_loop_nest_contains(nest, loop, block_index)) {
    return;
  }
  for (;;) {
    const uint16_t jump = lane_liveness->loops[loop].jump_index;
    if (jump != loop && !loom_cfg_loop_nest_contains(nest, jump, block_index)) {
      loop = jump;
      continue;
    }
    const uint16_t parent = nest->loops[loop].parent_loop_index;
    if (parent == LOOM_CFG_LOOP_NEST_NONE ||
        loom_cfg_loop_nest_contains(nest, parent, block_index)) {
      break;
    }
    loop = parent;
  }
  const loom_liveness_segment_t extent = lane_liveness->loops[loop].extent;
  loom_low_allocation_unit_liveness_value_t* value =
      &unit_liveness->values[value_ordinal];
  value->acquisition_start_point =
      iree_min(value->acquisition_start_point, extent.start_point);
  for (uint32_t i = 0; i < unit_count; ++i) {
    const uint32_t unit = unit_point_start + i;
    unit_liveness->start_points[unit] =
        iree_min(unit_liveness->start_points[unit], extent.start_point);
    unit_liveness->end_points[unit] =
        iree_max(unit_liveness->end_points[unit], extent.end_point);
  }
}
