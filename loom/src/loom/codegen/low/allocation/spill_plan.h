// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Spill plan construction for allocation assignments.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Computes the byte size and alignment required for spilling |assignment|.
iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment);

// Appends the spill materialization plan for |assignment|.
iree_status_t loom_low_allocation_spill_plan_record(
    const loom_module_t* module, loom_region_t* body,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, uint16_t alloc_unit_bits,
    loom_low_spill_slot_space_t spill_slot_space,
    loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t* inout_spill_plan_count);

// Appends a spill remark for |assignment_index|.
void loom_low_allocation_spill_remark_record(
    loom_low_allocation_remark_t* remarks, iree_host_size_t* inout_remark_count,
    uint32_t assignment_index, uint32_t budget_units, uint32_t required_units);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_PLAN_H_
