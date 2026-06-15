// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_order.h"

#include "loom/codegen/low/allocation/live_range.h"

static bool loom_low_allocation_interval_order_less(
    const loom_liveness_interval_t* lhs, const loom_liveness_interval_t* rhs) {
  if (lhs->start_point != rhs->start_point) {
    return lhs->start_point < rhs->start_point;
  }
  if (lhs->end_point != rhs->end_point) {
    return lhs->end_point < rhs->end_point;
  }
  return lhs->value_id < rhs->value_id;
}

static void loom_low_allocation_interval_order_sort(
    const loom_liveness_interval_t** intervals, iree_host_size_t count) {
  for (iree_host_size_t i = 1; i < count; ++i) {
    const loom_liveness_interval_t* value = intervals[i];
    iree_host_size_t j = i;
    while (j > 0 &&
           loom_low_allocation_interval_order_less(value, intervals[j - 1])) {
      intervals[j] = intervals[j - 1];
      --j;
    }
    intervals[j] = value;
  }
}

iree_status_t loom_low_allocation_interval_order_build(
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_interval_order_t* out_order) {
  *out_order = (loom_low_allocation_interval_order_t){0};
  iree_host_size_t interval_count = 0;
  iree_host_size_t unit_count = 0;
  for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
    const loom_liveness_interval_t* interval = &liveness->intervals[i];
    if (loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      ++interval_count;
      if (interval->unit_count > IREE_HOST_SIZE_MAX - unit_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "allocation unit count exceeds host size");
      }
      unit_count += interval->unit_count;
    }
  }
  if (interval_count == 0) {
    return iree_ok_status();
  }

  const loom_liveness_interval_t** intervals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, interval_count, sizeof(*intervals), (void**)&intervals));
  iree_host_size_t interval_index = 0;
  for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
    const loom_liveness_interval_t* interval = &liveness->intervals[i];
    if (loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      intervals[interval_index++] = interval;
    }
  }
  loom_low_allocation_interval_order_sort(intervals, interval_count);

  *out_order = (loom_low_allocation_interval_order_t){
      .intervals = intervals,
      .interval_count = interval_count,
      .unit_count = unit_count,
  };
  return iree_ok_status();
}
