// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_RANGES_H_
#define LOOM_RUNTIME_STORAGE_RANGES_H_

#include "iree/base/bitmap.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/vm/ref.h"
#include "loom/runtime/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

// Contiguous logical ranges over source-owned geometry. Roots keep their native
// addresses; a range owns corresponding regions in every root. The source owns
// byte strides and indexing, while final queue captures own interval reuse.
typedef struct loom_runtime_ranges_t {
  // Allocator owning bitmap words and reservation metadata.
  iree_allocator_t host_allocator;
  // Serializes acquisition with final queue release and observer detachment.
  iree_slim_mutex_t mutex;
  // Occupied logical units, including intervals awaiting final queue release.
  iree_bitmap_t occupied;
  // Monotonic final-release generation for capacity-sensitive scheduling.
  uint64_t generation;
  // Wake-only observer invoked under the mutex on final capacity return.
  loom_runtime_wake_callback_t observer;
  // Number of stable roots lent by the enclosing model.
  iree_host_size_t root_count;
  // Borrowed root table, valid through every reservation and accepted capture.
  iree_hal_buffer_t* const* roots;
} loom_runtime_ranges_t;

// Geometry is validated at source loading. Zero capacity initializes an inert
// owner. Partial initialization is safe to deinitialize. The owner joins actual
// root retirement before destroying this object or the borrowed root table.
iree_status_t loom_runtime_ranges_initialize(iree_host_size_t capacity,
                                             iree_host_size_t root_count,
                                             iree_hal_buffer_t* const* roots,
                                             iree_allocator_t host_allocator,
                                             loom_runtime_ranges_t* out_ranges);
void loom_runtime_ranges_deinitialize(loom_runtime_ranges_t* ranges);

void loom_runtime_ranges_set_observer(loom_runtime_ranges_t* ranges,
                                      loom_runtime_wake_callback_t observer);
uint64_t loom_runtime_ranges_generation(loom_runtime_ranges_t* ranges);

// Acquires count contiguous logical units, or returns acquired=false without
// error or ownership when none fit. The caller established 0<count<=capacity.
// A nonnull reservation is an owned handle from this range owner, possibly
// relinquished by reset. An active handle is already admitted. Allocation
// failure leaves the handle unchanged and returns no interval.
iree_status_t loom_runtime_ranges_try_acquire(loom_runtime_ranges_t* ranges,
                                              iree_host_size_t count,
                                              iree_vm_ref_t* reservation,
                                              bool* out_acquired);

// Queries an admitted handle. Each root view retains this particular interval
// through final queue release without changing its address or binding extent.
bool loom_runtime_range_is_active(iree_vm_ref_t reservation);
iree_host_size_t loom_runtime_range_begin(iree_vm_ref_t reservation);
iree_host_size_t loom_runtime_range_buffer_count(iree_vm_ref_t reservation);
iree_hal_buffer_t* loom_runtime_range_buffer(iree_vm_ref_t reservation,
                                             iree_host_size_t ordinal);

// Relinquishes the current interval and cached root views. The handle remains
// valid and can acquire another interval; previously captured views retain the
// old interval independently. Final handle release also resets it.
void loom_runtime_range_reset(iree_vm_ref_t reservation);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_STORAGE_RANGES_H_
