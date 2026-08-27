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

// Predicted memory traffic for materializing one spilled value.
typedef struct loom_low_allocation_spill_plan_traffic_t {
  // Predicted stores needed by the current synthetic spill plan.
  uint32_t store_count;
  // Predicted bytes stored by the current synthetic spill plan.
  uint64_t store_bytes;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint32_t reload_count;
  // Predicted bytes reloaded by the current synthetic spill plan.
  uint64_t reload_bytes;
} loom_low_allocation_spill_plan_traffic_t;

enum {
  // Equal-byte full reloads are only worthwhile after removing enough reload
  // packets to offset the longer live range of the reloaded tuple.
  LOOM_LOW_ALLOCATION_DENSE_SLICE_RELOAD_MIN_SLICE_COUNT = 8u,
};

// Computes the byte size and alignment required for spilling |assignment|.
iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment);

// Returns true when |slice_op| can be materialized as a reload from one unit of
// a spilled value with |spill_byte_size| bytes.
bool loom_low_allocation_spill_plan_slice_reload_byte_offset(
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_op_t* slice_op, uint16_t operand_index,
    uint32_t* out_unit_byte_size, int64_t* out_reload_offset);

// Returns true when a block-local slice group should share one full reload.
bool loom_low_allocation_spill_plan_use_full_slice_reload(
    uint32_t slice_count, uint64_t narrow_reload_bytes,
    uint32_t spill_byte_size);

// Computes the predicted memory traffic for spilling |value_id|.
iree_status_t loom_low_allocation_spill_plan_traffic(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits,
    loom_low_allocation_spill_plan_traffic_t* out_traffic);

// Appends the spill materialization plan for |assignment|.
iree_status_t loom_low_allocation_spill_plan_record(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
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
