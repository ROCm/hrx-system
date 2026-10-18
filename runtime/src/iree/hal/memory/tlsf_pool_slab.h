// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_TLSF_POOL_SLAB_H_
#define IREE_HAL_MEMORY_TLSF_POOL_SLAB_H_

#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/tlsf.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_tlsf_pool_slab_t {
  // Next owned slab, or next detached slab awaiting release outside the mutex.
  struct iree_hal_tlsf_pool_slab_t* next;

  // Previous owned slab, allowing constant-time removal of unused backing.
  struct iree_hal_tlsf_pool_slab_t* previous;

  // Next slab whose free state changed during a metadata transaction.
  struct iree_hal_tlsf_pool_slab_t* candidate_next;

  // Whether this slab is already linked in the return-candidate list.
  bool return_candidate;

  // Cold transaction preparations preserving this slab until their next
  // attempt.
  uint32_t preparation_count;

  // One presecured history credit, returned with this backing ownership.
  bool history_owned;

  // Offset allocator for the address range backed by |slab|.
  iree_hal_memory_tlsf_t tlsf;

  // Prepared backing range; borrowed from source_range for a finite arena.
  iree_hal_pool_buffer_range_t range;

  // Ordinary parent reservation owned until this slab is returned.
  iree_hal_pool_reservation_t reservation;

  // Exact history returned with reservation before TLSF metadata is freed.
  const iree_async_frontier_t* release_frontier;
} iree_hal_tlsf_pool_slab_t;

// Resolves raw allocator geometry with optional protection margins.
iree_status_t iree_hal_tlsf_pool_slab_calculate_geometry(
    const iree_hal_memory_tlsf_options_t* options,
    const iree_hal_asan_pool_options_t* asan, iree_device_size_t* out_alignment,
    iree_device_size_t* out_length);

// Resolves a slab's full parent extent and supported native alignment.
iree_status_t iree_hal_tlsf_pool_slab_resolve_request(
    const iree_hal_pool_capabilities_t* capabilities,
    const iree_hal_memory_tlsf_options_t* options,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_reservation_request_t* out_request);

// Initializes a finite arena borrowing its creator's retained buffer range.
iree_status_t iree_hal_tlsf_pool_slab_initialize_range(
    iree_hal_pool_buffer_range_t range, iree_hal_memory_tlsf_options_t options,
    iree_allocator_t host_allocator, iree_hal_tlsf_pool_slab_t** out_slab);

// Consumes an acquired parent token, including on failure. Its exact history
// remains attached through materialization, initialization, and rollback.
// Consumes optional detached |reusable_slab| metadata, including on failure.
iree_status_t iree_hal_tlsf_pool_slab_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    iree_hal_memory_tlsf_options_t options,
    const iree_hal_asan_pool_options_t* asan, iree_allocator_t host_allocator,
    iree_hal_tlsf_pool_slab_t* reusable_slab,
    iree_hal_tlsf_pool_slab_t** out_slab);

// Returns all parent ownership while preserving the free offset metadata.
// The caller has detached this slab from allocation and preparation readers.
void iree_hal_tlsf_pool_slab_release_backing(iree_hal_pool_t* backing_pool,
                                             iree_hal_tlsf_pool_slab_t* slab);

// Releases the prepared view, returns parent ownership, and destroys metadata.
// Finite arenas borrow their range and have no parent token to return.
void iree_hal_tlsf_pool_slab_release(iree_hal_pool_t* backing_pool,
                                     iree_hal_tlsf_pool_slab_t* slab,
                                     iree_allocator_t host_allocator);

// Includes untouched alignment margins in the complete returned history.
bool iree_hal_tlsf_pool_slab_query_full_free(
    iree_hal_tlsf_pool_slab_t* slab, const iree_async_frontier_t** out_frontier,
    iree_hal_memory_tlsf_block_flags_t* out_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_MEMORY_TLSF_POOL_SLAB_H_
