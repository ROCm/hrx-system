// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_COPY_H_
#define LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_COPY_H_

#include "loom/analysis/view_regions.h"
#include "loom/transforms/kernel/private_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// A copy transfers the reaching values at this point in the program. Only
// cells observed by scalar accesses occupy plan space, even for a large copy.
struct loom_private_storage_copy_t {
  // Operation erased when both allocations are promoted.
  loom_op_t* op;
  // Private allocation read by the copy.
  loom_private_storage_allocation_t* source;
  // Private allocation written by the copy.
  loom_private_storage_allocation_t* target;
  // First source byte relative to the source allocation.
  int64_t source_offset;
  // First target byte relative to the target allocation.
  int64_t target_offset;
  // Number of contiguous bytes transferred.
  int64_t byte_length;
  // First intersected target cell in plan order.
  iree_host_size_t cell_begin;
  // Number of intersected target cells.
  iree_host_size_t cell_count;
  // Source cell for each target cell, or NULL for unknown contents.
  const loom_private_storage_cell_t** source_cells;
  // Next copy in the same function.
  loom_private_storage_copy_t* next;
};

// Retains an in-bounds, nonoverlapping transfer. Unproven ranges and external
// allocations leave out_copy NULL and follow ordinary access classification.
iree_status_t loom_private_storage_record_copy(
    loom_private_storage_plan_t* plan, loom_op_t* op,
    loom_private_storage_allocation_t* source, int64_t source_offset,
    loom_private_storage_allocation_t* target, int64_t target_offset,
    int64_t byte_length, loom_private_storage_copy_t** out_copy);

// Recognizes one scalar load/store per iteration as a byte transfer when
// both accesses walk contiguous ranges in the same direction. The caller
// supplies the private roots of the two views.
iree_status_t loom_private_storage_record_copy_loop(
    loom_private_storage_plan_t* plan, loom_view_region_table_t* regions,
    const loom_value_fact_table_t* facts, loom_op_t* loop, loom_op_t* load,
    loom_private_storage_allocation_t* source, loom_op_t* store,
    loom_private_storage_allocation_t* target,
    loom_private_storage_copy_t** out_copy);

// Maps byte ranges to the completed sparse scalar partition before value flow.
iree_status_t loom_private_storage_partition_copies(
    loom_private_storage_plan_t* plan);

// Keeps an entire connected component in memory if any member must retain an
// access, including a load whose reaching value could not be established.
void loom_private_storage_resolve_copy_selection(
    loom_private_storage_plan_t* plan);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_COPY_H_
