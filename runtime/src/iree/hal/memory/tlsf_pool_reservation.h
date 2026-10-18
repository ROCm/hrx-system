// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_TLSF_POOL_RESERVATION_H_
#define IREE_HAL_MEMORY_TLSF_POOL_RESERVATION_H_

#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/tlsf.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_tlsf_pool_release_node_t {
  // Intrusive next pointer in pool->pending_release_head or
  // pool-owned release-node lists.
  struct iree_hal_tlsf_pool_release_node_t* next;

  // Prepared range containing this reservation. For suballocations this is
  // embedded in the owning slab; dedicated ranges share this record storage.
  const iree_hal_pool_buffer_range_t* range;

  // TLSF block handle, or NONE for a dedicated parent reservation.
  iree_hal_memory_tlsf_block_index_t block_index;

  // Backing bytes charged to this reservation.
  iree_device_size_t charged_length;

  // Backing offset within |range| for this reservation.
  iree_device_size_t backing_offset;

  // ASAN backing layout for this reservation.
  iree_hal_asan_allocation_layout_t asan_layout;
} iree_hal_tlsf_pool_release_node_t;

// Fixed record geometry captured when the allocator is constructed.
typedef struct iree_hal_tlsf_pool_reservation_layout_t {
  // Record plus inline frontier storage, in bytes.
  iree_host_size_t node_size;
  // Inline exact history following the release record.
  struct {
    // Byte offset from the record base.
    iree_host_size_t offset;
    // Maximum exact frontier width held by every record.
    uint8_t capacity;
  } frontier;
} iree_hal_tlsf_pool_reservation_layout_t;

// Builds the bounded layout from the allocator's resolved frontier capacity.
iree_hal_tlsf_pool_reservation_layout_t iree_hal_tlsf_pool_reservation_layout(
    uint8_t frontier_capacity);

// Request geometry validated once at the public acquisition boundary.
typedef struct iree_hal_tlsf_pool_request_geometry_t {
  // Aligned backing bytes required by the public user range.
  iree_device_size_t length;
  // Required absolute backing alignment.
  iree_device_size_t alignment;
  // Hidden guard geometry, empty when protection is disabled.
  iree_hal_asan_allocation_layout_t asan;
} iree_hal_tlsf_pool_request_geometry_t;

// Validates external request geometry against captured backing limits. The
// zero-initialized output receives geometry stable across cold retries.
iree_status_t iree_hal_tlsf_pool_reservation_geometry(
    const iree_hal_pool_reservation_request_t* request,
    iree_device_size_t minimum_alignment,
    const iree_hal_pool_capabilities_t* capabilities,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_tlsf_pool_request_geometry_t* out_geometry);

// Prepares one dedicated ordinary range from an acquired parent token. Consumes
// the token on success or failure, preserving its exact history on rollback.
iree_status_t iree_hal_tlsf_pool_dedicated_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* backing_request,
    const iree_hal_asan_pool_options_t* asan,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    const iree_hal_tlsf_pool_reservation_layout_t* layout,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_tlsf_pool_release_node_t** out_node);

// Merges untouched alignment margins before returning the whole parent token.
// False preserves an unrepresentable history until caller-quiescent teardown.
bool iree_hal_tlsf_pool_dedicated_merge_return_frontier(
    iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout);

// Full parent extent held by a dedicated record, including alignment margins.
iree_device_size_t iree_hal_tlsf_pool_dedicated_backing_length(
    const iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout);

// Returns exact history and releases the prepared view and record. Runs on the
// captured memory owner or a cold rollback path, outside allocator locks.
void iree_hal_tlsf_pool_dedicated_release(
    iree_hal_pool_t* backing_pool, iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout,
    iree_allocator_t host_allocator);

// Exact prior-use or returned history following the stable release record.
static inline iree_async_frontier_t* iree_hal_tlsf_pool_reservation_frontier(
    const iree_hal_tlsf_pool_release_node_t* node,
    iree_host_size_t frontier_offset) {
  return (iree_async_frontier_t*)((uint8_t*)node + frontier_offset);
}

// Materializes a complete transaction over the records' prepared ranges.
// Successful owning views return their reservations; the caller keeps the pool
// alive until all views and explicit reservation epochs have returned.
iree_status_t iree_hal_tlsf_pool_reservation_materialize(
    iree_hal_pool_t* pool, iree_host_size_t frontier_offset,
    iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_MEMORY_TLSF_POOL_RESERVATION_H_
