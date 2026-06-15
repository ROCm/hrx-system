// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocatable liveness interval ordering for low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"

#ifdef __cplusplus
extern "C" {
#endif

// Deterministic storage-allocation order over allocatable intervals.
typedef struct loom_low_allocation_interval_order_t {
  // Allocatable intervals sorted by allocation order.
  const loom_liveness_interval_t** intervals;
  // Number of entries in |intervals|.
  iree_host_size_t interval_count;
  // Sum of allocatable units across |intervals|.
  iree_host_size_t unit_count;
} loom_low_allocation_interval_order_t;

// Builds the allocatable interval order for |liveness|.
iree_status_t loom_low_allocation_interval_order_build(
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_interval_order_t* out_order);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_
