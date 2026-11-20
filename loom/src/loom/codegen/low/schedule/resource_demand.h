// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_DEMAND_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_DEMAND_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// A resource's capacity fits in uint16_t. Sixteen fractional bits retain a
// nonzero price even for a single use of the widest resource.
#define LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE (UINT64_C(1) << 16)

typedef struct loom_low_schedule_resource_demand_t {
  // Unit cycles consumed by single-form operations, indexed by contention
  // group.
  uint64_t* fixed_unit_cycles;
  // Cached cost plus one by physical schedule-class row. Zero means unpriced.
  uint64_t* class_costs;
} loom_low_schedule_resource_demand_t;

// Clears the active block's demand. The first block with equivalent forms
// allocates the storage, which subsequent blocks reuse.
iree_status_t loom_low_schedule_resource_demand_begin(
    const loom_low_schedule_model_t* model, iree_arena_allocator_t* arena,
    loom_low_schedule_resource_demand_t* demand);

// Adds the required unit cycles of a class whose encoding is fixed. A shared
// reservation can overlap other reservations and supplies no additive demand.
void loom_low_schedule_resource_demand_add(
    const loom_low_schedule_resource_demand_t* demand,
    const loom_low_schedule_model_t* model,
    const loom_low_schedule_class_t* schedule_class);

// Prices interference with the completed fixed demand. A class pays for its
// required unit cycles times each group's fixed unit cycles divided by group
// capacity, in units of LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE. This ranks
// equivalent forms after exact issue costs tie.
uint64_t loom_low_schedule_resource_demand_cost(
    const loom_low_schedule_resource_demand_t* demand,
    const loom_low_schedule_model_t* model,
    const loom_low_schedule_class_t* schedule_class);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_DEMAND_H_
