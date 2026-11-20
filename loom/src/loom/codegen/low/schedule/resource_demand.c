// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_demand.h"

#include <string.h>

#include "iree/base/internal/math.h"

iree_status_t loom_low_schedule_resource_demand_begin(
    const loom_low_schedule_model_t* model, iree_arena_allocator_t* arena,
    loom_low_schedule_resource_demand_t* demand) {
  const iree_host_size_t count =
      (iree_host_size_t)model->resource_count + model->schedule_class_row_count;
  if (!demand->fixed_unit_cycles) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count, sizeof(uint64_t), (void**)&demand->fixed_unit_cycles));
    demand->class_costs = demand->fixed_unit_cycles + model->resource_count;
  }
  memset(demand->fixed_unit_cycles, 0, count * sizeof(uint64_t));
  return iree_ok_status();
}

void loom_low_schedule_resource_demand_add(
    const loom_low_schedule_resource_demand_t* demand,
    const loom_low_schedule_model_t* model,
    const loom_low_schedule_class_t* schedule_class) {
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* use =
        &model->issue_uses[schedule_class->issue_use_start + i];
    if (use->kind == LOOM_LOW_ISSUE_USE_KIND_REQUIRED) {
      const uint16_t group =
          model->resources[use->resource_id].calendar.group_ordinal;
      demand->fixed_unit_cycles[group] = iree_math_saturating_add_u64(
          demand->fixed_unit_cycles[group], (uint64_t)use->units * use->cycles);
    }
  }
}

uint64_t loom_low_schedule_resource_demand_cost(
    const loom_low_schedule_resource_demand_t* demand,
    const loom_low_schedule_model_t* model,
    const loom_low_schedule_class_t* schedule_class) {
  uint64_t* cached =
      &demand->class_costs[schedule_class - model->schedule_classes];
  if (*cached) {
    return *cached - 1;
  }
  uint64_t cost = 0;
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* use =
        &model->issue_uses[schedule_class->issue_use_start + i];
    if (use->kind == LOOM_LOW_ISSUE_USE_KIND_REQUIRED) {
      const loom_low_resource_t* resource = &model->resources[use->resource_id];
      const uint64_t work = (uint64_t)use->units * use->cycles *
                            LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE;
      const uint64_t interference = iree_math_saturating_mul_u64(
          demand->fixed_unit_cycles[resource->calendar.group_ordinal], work);
      cost = iree_math_saturating_add_u64(
          cost, interference / resource->capacity_per_cycle);
    }
  }
  *cached = iree_math_saturating_add_u64(cost, 1);
  return *cached - 1;
}
