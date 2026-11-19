// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BACKING_PLAN_H_
#define LOOM_RUNTIME_STORAGE_BACKING_PLAN_H_

#include "loom/runtime/storage/backing_pages.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_virtual_buffer_t loom_runtime_virtual_buffer_t;

// Consecutive logical pages belonging to a single contiguous physical lease.
typedef struct loom_runtime_backing_run_t {
  // Borrowed virtual reservation, kept alive by the operation's caller.
  loom_runtime_virtual_buffer_t* buffer;
  // First logical page in the reservation.
  iree_host_size_t index;
  // Number of independently replaceable pages in the lease.
  iree_host_size_t count;
  // First selected page in caller order, used by streamed source I/O.
  iree_host_size_t ordinal;
  // Physical parent and byte offset of the first page.
  loom_runtime_backing_slot_t slot;
} loom_runtime_backing_run_t;

// Private to one operation, immutable once its native work is submitted.
typedef struct loom_runtime_backing_plan_t {
  // Number of captured physical runs.
  iree_host_size_t count;
  // Capacity of the run storage.
  iree_host_size_t capacity;
  // Initial operation storage or allocator-owned overflow.
  loom_runtime_backing_run_t* runs;
  // Borrowed initial storage owned by the enclosing operation allocation.
  loom_runtime_backing_run_t* initial;
} loom_runtime_backing_plan_t;

void loom_runtime_backing_plan_initialize(loom_runtime_backing_plan_t* plan,
                                          iree_host_size_t capacity,
                                          loom_runtime_backing_run_t* initial);
void loom_runtime_backing_plan_deinitialize(loom_runtime_backing_plan_t* plan,
                                            iree_allocator_t allocator);

// Records one complete lease. Failure leaves previous runs intact.
iree_status_t loom_runtime_backing_plan_append(
    loom_runtime_backing_plan_t* plan, iree_allocator_t allocator,
    loom_runtime_backing_run_t run);

// Extends the first run only when logical and physical pages are consecutive.
bool loom_runtime_backing_run_extend(loom_runtime_backing_run_t* previous,
                                     const loom_runtime_backing_run_t* next,
                                     iree_device_size_t stride);

// Captures a physical piece, coalescing adjacent pieces as they arrive.
iree_status_t loom_runtime_backing_plan_capture(
    loom_runtime_backing_plan_t* plan, iree_allocator_t allocator,
    loom_runtime_backing_run_t run, iree_device_size_t stride);

// Restores selection order after whole-parent claims preceded packed claims.
void loom_runtime_backing_plan_order_by_ordinal(
    loom_runtime_backing_plan_t* plan);

// Orders independently captured existing mappings by target and page.
void loom_runtime_backing_plan_order_by_target(
    loom_runtime_backing_plan_t* plan);

// Resolves a logical I/O ordinal after source readiness.
const loom_runtime_backing_run_t* loom_runtime_backing_plan_find(
    const loom_runtime_backing_plan_t* plan, iree_host_size_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_STORAGE_BACKING_PLAN_H_
