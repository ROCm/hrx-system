// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_HOST_MEMORY_H_
#define AMDF_SRC_PLATFORM_HOST_MEMORY_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned preparation state for a WRITE_BACK source and its ordinary host view.
typedef struct amdf_platform_host_backing_t {
  // Native section/file handle, or UINTPTR_MAX before acquisition/after
  // release. Explicit alias updates borrow this handle while the backing is
  // live.
  uintptr_t handle;
  // Persistent host mapping owned alongside the native source.
  void* pointer;
} amdf_platform_host_backing_t;

// Prepares remappable system backing and its persistent host view in a live
// owner's initially empty slot. This one-shot operation initializes both fields
// before acquiring native resources. Failure retains partial ownership for
// destruction; it performs no hidden rollback.
amdf_status_t amdf_platform_host_backing_prepare(
    uint64_t byte_length, amdf_platform_host_backing_t* backing);

// Materializes fresh backing and its writable host translations before the
// first native export. Applies explicit NUMA placement before allocating any
// pages. Linux restricts exports on kernels before 7.2 to base pages because
// their udmabuf scatterlists mishandle large folios. Newer kernels attempt
// synchronous huge-folio promotion. The complete range is populated regardless
// of page size; huge-page availability is not an allocation requirement.
// Required placement/population failures leave the backing owned by its
// unpublished constructor for rollback.
amdf_status_t amdf_platform_host_backing_populate(
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_memory_placement_t placement, amdf_allocator_t host_allocator);

// Removes the persistent view before releasing its source. A failed removal
// retains native ownership; completed steps clear their corresponding fields.
// Callers may discard metadata at terminal teardown after observing that state.
amdf_status_t amdf_platform_host_backing_destroy(
    amdf_platform_host_backing_t* backing, uint64_t byte_length);

// Allocates write-back system virtual memory. The validated length is a
// multiple of the platform host allocation granularity; no accelerator is
// involved.
amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer);

// Releases exactly one successful host allocation. Failure retains that range.
amdf_status_t amdf_platform_host_memory_free(void* pointer,
                                             uint64_t byte_length);

// Explicit placement constraints supported by fresh host allocations. This
// metadata query discovers no topology and performs no native operation.
amdf_memory_placement_kinds_t amdf_platform_host_memory_placement_kinds(void);

// Applies a validated placement to a fresh, exclusively owned range and
// populates its pages. A failure leaves the range owned by its constructor,
// which must roll it back before publishing it. Never changes thread policy.
amdf_status_t amdf_platform_host_memory_place(void* pointer,
                                              uint64_t byte_length,
                                              amdf_memory_placement_t placement,
                                              amdf_allocator_t host_allocator);

// Qualifies explicit CPU cache maintenance for platform-owned write-back pages.
// The returned flush program also implements invalidation after changing its
// host_operation. Failure leaves the output unchanged.
amdf_status_t amdf_platform_host_memory_query_cache_transition(
    amdf_cache_transition_t* out_transition);

// Writes back and invalidates the validated range with full fences before and
// after the operation. The line size is the previously qualified host fact.
amdf_status_t amdf_platform_host_memory_cache_control(void* pointer,
                                                      uint64_t byte_length,
                                                      uint32_t line_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_HOST_MEMORY_H_
