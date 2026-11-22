// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BACKING_PAGES_H_
#define LOOM_RUNTIME_STORAGE_BACKING_PAGES_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/runtime/storage/statistics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_backing_pages_t loom_runtime_backing_pages_t;
typedef struct loom_runtime_backing_parent_t loom_runtime_backing_parent_t;

// A physical lease held until private unmap or checked whole-target detach.
// Slot operations belong to one serialized owner; native completion never
// mutates its bitmap or availability lists.
typedef struct loom_runtime_backing_slot_t {
  // Parent retained by this lease; NULL denotes no physical ownership.
  loom_runtime_backing_parent_t* parent;
  // Byte origin within the parent's logical mapping-source range.
  iree_device_size_t offset;
} loom_runtime_backing_slot_t;

typedef uint8_t loom_runtime_backing_kind_t;
enum loom_runtime_backing_kind_e {
  LOOM_RUNTIME_BACKING_PACKED = 0,
  LOOM_RUNTIME_BACKING_WEIGHT = 1,
  LOOM_RUNTIME_BACKING_DEDICATED = 2,
};

// Captured cold growth. All metadata and credits exist before submit; abort
// requires returning its leases first. The caller serializes capture/submit.
typedef struct loom_runtime_backing_growth_t {
  // Newly prepared parents; NULL when existing capacity satisfied the request.
  loom_runtime_backing_parent_t* parents;
  // Remaining complete-cohort admission, split into prepared parents.
  iree_hal_memory_credit_t credit;
} loom_runtime_backing_growth_t;

// The source and device outlive this owner and every accepted native
// completion. Parent extents are qualified against the source before IO.
iree_status_t loom_runtime_backing_pages_allocate(
    iree_hal_pool_t* source, iree_hal_device_t* device,
    iree_hal_pool_reservation_request_t packed,
    iree_hal_pool_reservation_request_t parent,
    iree_hal_pool_reservation_request_t weight, void (*wake)(void*),
    void* wake_user_data, iree_allocator_t host_allocator,
    loom_runtime_backing_pages_t** out_pages);
void loom_runtime_backing_pages_free(loom_runtime_backing_pages_t* pages);

// Seeds a newly created inventory with ready, unclaimed packed-page parents.
// Reserves the complete native cost before submission and joins all accepted
// acquisitions before returning, including on failure.
iree_status_t loom_runtime_backing_pages_prewarm(
    loom_runtime_backing_pages_t* pages, iree_hal_memory_budget_t* budget,
    uint64_t minimum_bytes);

// Complete additional physical cost after reusable slots/whole parents. The
// dedicated cost is captured from the source at buffer-geometry construction.
uint64_t loom_runtime_backing_pages_growth_cost(
    const loom_runtime_backing_pages_t* pages, uint64_t packed_count,
    uint64_t weight_count, uint64_t dedicated_cost);

// Captures exactly the additional packed parents admitted for this cohort.
// Call after whole-parent claims, before packed claims. Existing free slots
// still satisfy the remainder; no speculative capacity is acquired.
iree_status_t loom_runtime_backing_pages_prepare_packed(
    loom_runtime_backing_pages_t* pages, uint64_t packed_count,
    loom_runtime_backing_growth_t* growth);

// Claims up to maximum_count consecutive logical slots from one parent.
// Whole-parent and dedicated leases each satisfy one logical slot.
// Submitting growth is separate so metadata failure cannot start partial IO.
iree_status_t loom_runtime_backing_pages_claim(
    loom_runtime_backing_pages_t* pages, loom_runtime_backing_kind_t kind,
    iree_hal_pool_reservation_request_t request, uint64_t dedicated_cost,
    iree_host_size_t maximum_count, loom_runtime_backing_growth_t* growth,
    loom_runtime_backing_slot_t* out_slot, iree_host_size_t* out_count);
void loom_runtime_backing_pages_submit(loom_runtime_backing_growth_t* growth);
void loom_runtime_backing_pages_abort(loom_runtime_backing_growth_t* growth);
// Returns consecutive leases to their original parent after native retirement.
void loom_runtime_backing_pages_return(loom_runtime_backing_slot_t slot,
                                       iree_host_size_t slot_count);

// One observation per pending parent per owner transaction. The returned
// semaphore is borrowed through the slot lifetime. Ready parents need no wait.
uint64_t loom_runtime_backing_pages_begin_observation(
    loom_runtime_backing_pages_t* pages);
iree_hal_semaphore_t* loom_runtime_backing_pages_observe(
    loom_runtime_backing_slot_t slot, uint64_t observation);
// Requires successful completion of that parent's allocation readiness.
iree_hal_buffer_t* loom_runtime_backing_pages_buffer(
    loom_runtime_backing_slot_t slot);

// Publishes completed native acquisitions and disposes idle failed parents.
void loom_runtime_backing_pages_collect(loom_runtime_backing_pages_t* pages);
// Returns idle parents without touching live or pending mapping leases.
void loom_runtime_backing_pages_trim(loom_runtime_backing_pages_t* pages,
                                     uint64_t native_bytes_to_return,
                                     uint64_t protected_packed_count,
                                     uint64_t protected_weight_count);
// Physical admission still represented by parents, including native growth.
uint64_t loom_runtime_backing_pages_retained(
    const loom_runtime_backing_pages_t* pages);
// Logical payload currently available for reuse, excluding native padding.
uint64_t loom_runtime_backing_pages_cached(
    const loom_runtime_backing_pages_t* pages);
// Copies the tail of the physical inventory in ascending acquisition order.
uint64_t loom_runtime_backing_pages_query_slabs(
    const loom_runtime_backing_pages_t* pages, iree_host_size_t capacity,
    loom_runtime_slab_layout_t* out_slabs);
bool loom_runtime_backing_pages_is_pending(
    const loom_runtime_backing_pages_t* pages);
// Outer-thread join. Publishes all pending parents and returns their status.
iree_status_t loom_runtime_backing_pages_await(
    loom_runtime_backing_pages_t* pages);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_STORAGE_BACKING_PAGES_H_
