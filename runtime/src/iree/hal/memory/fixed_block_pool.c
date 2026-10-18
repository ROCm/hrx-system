// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/fixed_block_pool_slab.h"
#include "iree/hal/memory/tracing.h"

enum {
  IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY = 8,
  IREE_HAL_FIXED_BLOCK_POOL_INLINE_FRONTIER_CAPACITY = 8,
};

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_fixed_block_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Ordinary backing pool retained once; NULL for a finite prepared range.
  iree_hal_pool_t* backing_pool;

  // Captured backing capabilities inherited by buffer views.
  iree_hal_pool_capabilities_t capabilities;

  // Immutable allocation geometry shared by every slab.
  iree_hal_fixed_block_pool_geometry_t geometry;

  // Exact source request, constructed once and reused for cold growth.
  iree_hal_pool_reservation_request_t backing_request;

  // Overflow captures secured on growth and independent of backing lifetime.
  iree_hal_pool_history_t history;

  // Serializes candidate snapshots and complete batch claims. All native work,
  // allocation and eligibility queries remain outside this mutex.
  iree_slim_mutex_t acquisition_mutex;

  // Owned slab inventory, guarded by acquisition_mutex.
  struct {
    // First owned slab.
    iree_hal_fixed_block_pool_slab_t* head;
    // Last owned slab, used for constant-time publication.
    iree_hal_fixed_block_pool_slab_t* tail;
  } slabs;

  // One idle metadata set, independent of backing ownership and released by
  // trim. Protected by acquisition_mutex.
  struct {
    // Offset metadata whose complete parent reservation has returned.
    iree_hal_fixed_block_pool_slab_t* slab;
    // Durable storage sized for the largest completed acquisition batch.
    struct iree_hal_fixed_block_pool_acquire_t* acquire;
  } reusable;

  // Empty-range publication independent of native allocation progress.
  struct {
    // Protects candidate links and each releasing caller's final slab access.
    iree_slim_mutex_t mutex;
    // Changed slabs awaiting detachment, each linked at most once.
    iree_hal_fixed_block_pool_slab_t* candidates;
  } publication;

  // Host allocator used for pool and slab metadata.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations.
  iree_hal_memory_trace_t trace;

  // Approximate owned backing bytes for constant-time statistics.
  iree_atomic_int64_t bytes_committed;

  // Approximate bytes managed by raw blocks, excluding untouched margins.
  iree_atomic_int64_t bytes_managed;

  // Approximate number of owned ranges for constant-time statistics.
  iree_atomic_int32_t slab_count;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // Logical byte budget for live reservations. 0 means unlimited.
  iree_device_size_t budget_limit;

  // Approximate live reservation bytes for lock-free stats queries.
  iree_atomic_int64_t bytes_reserved;

  // Approximate live reservation count for lock-free stats queries.
  iree_atomic_int32_t reservation_count;

  // Total reservations committed by successful transactions.
  iree_atomic_int64_t reserve_count;

  // Total reservations returned by release transactions.
  iree_atomic_int64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  iree_atomic_int64_t reuse_count;

  // Reserves where dominance check failed.
  iree_atomic_int64_t reuse_miss_count;

  // Reserves from fresh (never-used) blocks.
  iree_atomic_int64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  iree_atomic_int64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  iree_atomic_int64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  iree_atomic_int64_t wait_count;
} iree_hal_fixed_block_pool_t;

typedef struct iree_hal_fixed_block_pool_materialize_state_t
    iree_hal_fixed_block_pool_materialize_state_t;

// Per-buffer element in an owning materialization transaction.
typedef struct iree_hal_fixed_block_pool_materialize_element_t {
  // Shared transaction state controlling the ownership commit.
  iree_hal_fixed_block_pool_materialize_state_t* state;

  // Reservation released when the committed buffer is destroyed.
  iree_hal_pool_reservation_t reservation;

  // Materialized buffer staged until the complete transaction succeeds.
  iree_hal_buffer_t* buffer;
} iree_hal_fixed_block_pool_materialize_element_t;

// Shared state for an owning materialization transaction.
struct iree_hal_fixed_block_pool_materialize_state_t {
  // Borrowed from the wrapped buffer's creator. Pool owners must keep the pool
  // alive until all buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  // Host allocator used for this state object.
  iree_allocator_t host_allocator;

  // Number of materialized buffers still referencing this transaction.
  iree_atomic_int32_t reference_count;

  // True after every buffer was materialized and reservation ownership moved.
  bool ownership_committed;

  // Per-buffer transaction elements.
  iree_hal_fixed_block_pool_materialize_element_t elements[];
};

// Staged result for one reservation acquisition. Transactions use staging so
// public output arrays remain untouched unless the operation succeeds.
typedef struct iree_hal_fixed_block_pool_acquire_element_t {
  // Slab pinned while this transaction borrows a candidate from it.
  iree_hal_fixed_block_pool_slab_t* slab;

  // Candidate with a copied frontier until commit, then the acquired block.
  iree_hal_memory_fixed_block_allocator_allocation_t allocation;

  // Eligibility result established outside the acquisition mutex.
  iree_hal_pool_acquire_result_t result;

  // Request layout prepared before any block is claimed.
  iree_hal_asan_allocation_layout_t asan_layout;
} iree_hal_fixed_block_pool_acquire_element_t;

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable;
static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID =
    "iree-hal-fixed-block-pool";

static bool iree_hal_fixed_block_pool_query_completed_epoch(
    void* user_data, iree_async_axis_t axis, uint64_t epoch) {
  return iree_async_frontier_tracker_query_epoch(user_data, axis, epoch);
}

//===----------------------------------------------------------------------===//
// Frontier helpers
//===----------------------------------------------------------------------===//

static bool iree_hal_fixed_block_pool_frontier_is_satisfied(
    const iree_hal_fixed_block_pool_t* pool,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* death_frontier,
    iree_hal_memory_fixed_block_allocator_block_flags_t block_flags) {
  if (!death_frontier) {
    return block_flags == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE;
  }
  if (block_flags & IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED) {
    return false;
  }
  if (requester_frontier) {
    const iree_async_frontier_comparison_t comparison =
        iree_async_frontier_compare(requester_frontier, death_frontier);
    if (comparison == IREE_ASYNC_FRONTIER_AFTER ||
        comparison == IREE_ASYNC_FRONTIER_EQUAL) {
      return true;
    }
  }
  if (!pool->base.epoch_query.fn) {
    return false;
  }

  iree_host_size_t requester_index = 0;
  for (uint8_t i = 0; i < death_frontier->entry_count; ++i) {
    const iree_async_axis_t axis = death_frontier->entries[i].axis;
    const uint64_t epoch = death_frontier->entries[i].epoch;
    while (requester_frontier &&
           requester_index < requester_frontier->entry_count &&
           requester_frontier->entries[requester_index].axis < axis) {
      ++requester_index;
    }
    if (requester_frontier &&
        requester_index < requester_frontier->entry_count &&
        requester_frontier->entries[requester_index].axis == axis &&
        requester_frontier->entries[requester_index].epoch >= epoch) {
      continue;
    }
    if (!pool->base.epoch_query.fn(pool->base.epoch_query.user_data, axis,
                                   epoch)) {
      return false;
    }
  }
  return true;
}

// Acquisitions hold the metadata lock. Concurrent releases only reduce the
// charge, so an addition checked against this snapshot cannot exceed the limit.
static bool iree_hal_fixed_block_pool_try_charge_transaction(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t charged_length) {
  if (pool->budget_limit != 0) {
    const iree_device_size_t current = (iree_device_size_t)iree_atomic_load(
        &pool->bytes_reserved, iree_memory_order_relaxed);
    if (current > pool->budget_limit ||
        charged_length > pool->budget_limit - current) {
      return false;
    }
  }
  iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)charged_length,
                        iree_memory_order_relaxed);
  return true;
}

static void iree_hal_fixed_block_pool_uncharge_reservation(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t charged_length) {
  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)charged_length,
                        iree_memory_order_relaxed);
}

static bool iree_hal_fixed_block_pool_can_wait_for_allocation(
    iree_hal_pool_reserve_flags_t flags,
    const iree_hal_memory_fixed_block_allocator_allocation_t* allocation) {
  return iree_all_bits_set(flags,
                           IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER) &&
         allocation->death_frontier &&
         !iree_all_bits_set(
             allocation->block_flags,
             IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);
}

static iree_status_t iree_hal_fixed_block_pool_calculate_asan_layout(
    const iree_hal_fixed_block_pool_t* pool, iree_device_size_t user_length,
    iree_device_size_t user_alignment,
    iree_hal_asan_allocation_layout_t* out_layout) {
  IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
      &pool->asan_options, user_length, user_alignment, out_layout));
  return iree_hal_asan_extend_allocation_layout(
      pool->geometry.backing_block_size, out_layout);
}

static void iree_hal_fixed_block_pool_return_allocation(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab,
    const iree_hal_memory_fixed_block_allocator_allocation_t* allocation,
    iree_device_size_t byte_length,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info) {
  const bool asan_enabled =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options);
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset =
      allocation->offset + (asan_enabled ? asan_layout->user_offset : 0);
  out_reservation->byte_length = byte_length;
  out_reservation->block_handle =
      (uint64_t)(uintptr_t)&slab->blocks[allocation->block_index];

  // Tainted blocks never reach this helper: frontier_is_satisfied rejects
  // them (so they never become OK/OK_FRESH) and can_wait_for_allocation
  // excludes them (so they never become OK_NEEDS_WAIT). Preserve the exact
  // prerequisite even when this requester already covers it.
  memset(out_info, 0, sizeof(*out_info));
  out_info->reuse_frontier = allocation->death_frontier;
  out_info->result = result;

  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
  switch (result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_atomic_fetch_add(&pool->wait_count, 1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid successful fixed-block pool result: %u",
                  result);
      break;
  }
  if (asan_enabled) {
    slab->asan_layouts[allocation->block_index] = *asan_layout;
  }
  iree_hal_memory_trace_alloc(&pool->trace,
                              &slab->blocks[allocation->block_index],
                              out_reservation->byte_length);
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

static void iree_hal_fixed_block_pool_link_slab(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab) {
  slab->previous = pool->slabs.tail;
  slab->next = NULL;
  if (pool->slabs.tail) {
    pool->slabs.tail->next = slab;
  } else {
    pool->slabs.head = slab;
  }
  pool->slabs.tail = slab;
}

static void iree_hal_fixed_block_pool_unlink_slab(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab) {
  if (slab->previous) {
    slab->previous->next = slab->next;
  } else {
    pool->slabs.head = slab->next;
  }
  if (slab->next) {
    slab->next->previous = slab->previous;
  } else {
    pool->slabs.tail = slab->previous;
  }
}

// The publication mutex makes the releasing thread's final slab access atomic
// with respect to another caller detaching it. It never protects raw
// allocation, frontier inspection or native work.
static void iree_hal_fixed_block_pool_note_candidate(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(
        requires_capability(&pool->publication.mutex)) {
  if (!slab->candidate_queued) {
    slab->candidate_queued = true;
    slab->candidate_next = pool->publication.candidates;
    pool->publication.candidates = slab;
  }
}

static void iree_hal_fixed_block_pool_unpin_slab(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(
        requires_capability(&pool->acquisition_mutex)) {
  if (--slab->pins == 0 && pool->backing_pool) {
    iree_slim_mutex_lock(&pool->publication.mutex);
    if (iree_atomic_load(&slab->live_count, iree_memory_order_relaxed) == 0) {
      iree_hal_fixed_block_pool_note_candidate(pool, slab);
    }
    iree_slim_mutex_unlock(&pool->publication.mutex);
  }
}

// Detaches only slabs whose ownership changed. Concurrent callers take
// disjoint candidate lists; the acquisition lock excludes new claims while an
// empty slab is unlinked. Parent returns never run with either metadata lock.
static void iree_hal_fixed_block_pool_return_unused_slabs(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t min_bytes_to_keep,
    bool retain_metadata) {
  iree_slim_mutex_lock(&pool->publication.mutex);
  iree_hal_fixed_block_pool_slab_t* candidates = pool->publication.candidates;
  pool->publication.candidates = NULL;
  iree_slim_mutex_unlock(&pool->publication.mutex);
  while (candidates) {
    iree_hal_fixed_block_pool_slab_t* slab = candidates;
    candidates = slab->candidate_next;
    iree_slim_mutex_lock(&pool->acquisition_mutex);
    iree_slim_mutex_lock(&pool->publication.mutex);
    slab->candidate_queued = false;
    const iree_device_size_t committed = (iree_device_size_t)iree_atomic_load(
        &pool->bytes_committed, iree_memory_order_relaxed);
    const bool detach =
        slab->pins == 0 &&
        iree_atomic_load(&slab->live_count, iree_memory_order_relaxed) == 0 &&
        slab->range.length <=
            committed - iree_min(committed, min_bytes_to_keep);
    if (detach) {
      iree_hal_fixed_block_pool_unlink_slab(pool, slab);
    }
    iree_slim_mutex_unlock(&pool->publication.mutex);
    iree_slim_mutex_unlock(&pool->acquisition_mutex);
    if (!detach) {
      continue;
    }

    if (iree_hal_fixed_block_pool_slab_merge_return(slab, &pool->geometry,
                                                    &pool->history)) {
      iree_atomic_fetch_sub(&pool->bytes_committed, slab->range.length,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_sub(
          &pool->bytes_managed,
          slab->block_count * pool->geometry.backing_block_size,
          iree_memory_order_relaxed);
      iree_atomic_fetch_sub(&pool->slab_count, 1, iree_memory_order_relaxed);
      slab->history_owned = false;
      iree_hal_pool_history_retire_backing(&pool->history);
      if (retain_metadata) {
        iree_hal_fixed_block_pool_slab_release_backing(slab,
                                                       pool->backing_pool);
        iree_slim_mutex_lock(&pool->acquisition_mutex);
        if (!pool->reusable.slab) {
          pool->reusable.slab = slab;
          slab = NULL;
        }
        iree_slim_mutex_unlock(&pool->acquisition_mutex);
      }
      if (slab) {
        iree_hal_fixed_block_pool_slab_free(slab, pool->backing_pool,
                                            pool->host_allocator);
      }
    } else {
      // Individual block histories remain useful when their union does not
      // fit the configured whole-range return frontier.
      iree_slim_mutex_lock(&pool->acquisition_mutex);
      iree_hal_fixed_block_pool_link_slab(pool, slab);
      iree_slim_mutex_unlock(&pool->acquisition_mutex);
      iree_async_notification_signal_if_observed(pool->base.notification,
                                                 INT32_MAX);
    }
  }
}

static void iree_hal_fixed_block_pool_publish_slab(
    iree_hal_fixed_block_pool_t* pool, iree_hal_fixed_block_pool_slab_t* slab) {
  iree_hal_fixed_block_pool_link_slab(pool, slab);
  iree_atomic_fetch_add(&pool->bytes_committed, slab->range.length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->bytes_managed,
                        slab->block_count * pool->geometry.backing_block_size,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, 1, iree_memory_order_relaxed);
}

static iree_device_size_t iree_hal_fixed_block_pool_size_alignment(
    iree_device_size_t size) {
  return (iree_device_size_t)1 << iree_math_count_trailing_zeros_u64(size);
}

static iree_status_t iree_hal_fixed_block_pool_resolve_geometry(
    const iree_hal_fixed_block_pool_options_t* options,
    const iree_hal_pool_capabilities_t* capabilities,
    iree_device_size_t source_offset,
    iree_hal_fixed_block_pool_geometry_t* out_geometry) {
  memset(out_geometry, 0, sizeof(*out_geometry));
  if (!options->block_size ||
      !iree_device_size_is_valid_alignment(options->alignment) ||
      options->frontier_capacity > UINT8_MAX ||
      options->blocks_per_slab >
          IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid fixed-block pool geometry");
  }
  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(&options->asan));
  iree_device_size_t alignment = options->alignment ? options->alignment : 1;
  iree_device_size_t max_alignment = capabilities->max_allocation_alignment;
  if (source_offset) {
    max_alignment = iree_min(
        max_alignment, iree_hal_fixed_block_pool_size_alignment(source_offset));
  }
  if (alignment > max_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block alignment exceeds backing support");
  }
  iree_device_size_t backing_block_size = options->block_size;
  if (iree_hal_asan_pool_options_is_enabled(&options->asan)) {
    max_alignment =
        iree_min(max_alignment,
                 iree_max(alignment, iree_hal_fixed_block_pool_size_alignment(
                                         options->block_size)));
    iree_hal_asan_allocation_layout_t layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &options->asan, options->block_size, max_alignment, &layout));
    alignment = iree_max(alignment, layout.backing_offset_alignment);
    backing_block_size = layout.backing_length;
  }
  if (alignment > capabilities->max_allocation_alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "protected block alignment exceeds backing support");
  }
  // Padding separates independently maintained storage regions. It does not
  // increase the native address alignment advertised for each block.
  alignment = iree_max(alignment, capabilities->maintenance_alignment);
  if (!iree_device_size_checked_align(backing_block_size, alignment,
                                      &backing_block_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixed block backing size overflows");
  }
  max_alignment =
      iree_min(max_alignment,
               iree_hal_fixed_block_pool_size_alignment(backing_block_size));
  const iree_device_size_t capacity =
      capabilities->max_allocation_size
          ? capabilities->max_allocation_size / backing_block_size
          : IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS;
  const uint32_t block_count = options->blocks_per_slab
                                   ? options->blocks_per_slab
                                   : (uint32_t)iree_min(64, capacity);
  iree_device_size_t length = 0;
  if (!block_count || block_count > capacity ||
      !iree_device_size_checked_mul(block_count, backing_block_size, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixed-block slab exceeds backing capacity");
  }
  *out_geometry = (iree_hal_fixed_block_pool_geometry_t){
      .block_size = options->block_size,
      .backing_block_size = backing_block_size,
      // Fixed blocks cannot compensate for an under-aligned source by moving
      // individual allocations. Prepare the full advertised alignment once.
      .alignment = max_alignment,
      .blocks_per_slab = block_count,
      .frontier_capacity =
          options->frontier_capacity
              ? (uint8_t)options->frontier_capacity
              : IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_DEFAULT_FRONTIER_CAPACITY,
  };
  return iree_ok_status();
}

static iree_hal_pool_reservation_request_t
iree_hal_fixed_block_pool_backing_request(
    const iree_hal_fixed_block_pool_geometry_t* geometry) {
  return (iree_hal_pool_reservation_request_t){
      .allocation_size =
          geometry->blocks_per_slab * geometry->backing_block_size,
      .min_alignment = geometry->alignment,
  };
}

void iree_hal_fixed_block_pool_options_initialize(
    iree_hal_fixed_block_pool_options_t* options) {
  memset(options, 0, sizeof(*options));
}

iree_status_t iree_hal_fixed_block_pool_query_backing_request(
    iree_hal_pool_t* backing_pool,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_hal_pool_reservation_request_t* out_request) {
  IREE_RETURN_IF_ERROR(
      iree_hal_pool_validate_asan_options(backing_pool, &options->asan));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(backing_pool, &capabilities);
  iree_hal_fixed_block_pool_geometry_t geometry;
  IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_resolve_geometry(
      options, &capabilities, 0, &geometry));
  *out_request = iree_hal_fixed_block_pool_backing_request(&geometry);
  return iree_ok_status();
}

static iree_status_t iree_hal_fixed_block_pool_create_impl(
    iree_hal_pool_t* backing_pool, const iree_hal_pool_buffer_range_t* range,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  iree_hal_pool_capabilities_t capabilities = {0};
  if (backing_pool) {
    IREE_RETURN_IF_ERROR(
        iree_hal_pool_validate_asan_options(backing_pool, &options->asan));
    iree_hal_pool_query_capabilities(backing_pool, &capabilities);
  } else {
    capabilities = (iree_hal_pool_capabilities_t){
        .memory_type = iree_hal_buffer_memory_type(range->buffer),
        .allowed_access = iree_hal_buffer_allowed_access(range->buffer),
        .supported_usage = iree_hal_buffer_allowed_usage(range->buffer),
        .queue_family_affinity =
            iree_hal_buffer_allocation_placement(range->buffer)
                .queue_family_affinity,
        .atomic_operations = range->memory.backing->atomic_operations,
        .max_allocation_size = range->length,
        .max_allocation_alignment = range->memory.backing->allocation_alignment,
        .maintenance_alignment = range->memory.backing->maintenance_alignment,
    };
  }
  iree_hal_fixed_block_pool_geometry_t geometry;
  IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_resolve_geometry(
      options, &capabilities, range ? range->memory.offset : 0, &geometry));

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_fixed_block_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  iree_async_proactor_t* proactor =
      backing_pool ? backing_pool->notification->proactor
                   : range->memory.backing->notification->proactor;
  iree_async_notification_t* notification = NULL;
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_fixed_block_pool_vtable,
        backing_pool ? backing_pool->memory_contract : range->memory.contract,
        notification,
        backing_pool ? backing_pool->wait_sources
                     : (iree_hal_pool_wait_source_list_t){0},
        backing_pool ? backing_pool->frontier_tracker
                     : range->memory.backing->tracker,
        host_allocator, &pool->base);
  }
  iree_async_notification_release(notification);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  pool->base.epoch_query =
      backing_pool ? backing_pool->epoch_query
                   : (iree_hal_pool_epoch_query_t){
                         .fn = iree_hal_fixed_block_pool_query_completed_epoch,
                         .user_data = range->memory.backing->tracker,
                     };
  pool->base.asan_enabled =
      iree_hal_asan_pool_options_is_enabled(&options->asan);
  if (backing_pool) {
    pool->base.allocation_interceptor = backing_pool->allocation_interceptor;
  }
  pool->backing_pool = backing_pool;
  iree_hal_pool_retain(backing_pool);
  pool->host_allocator = host_allocator;
  pool->capabilities = capabilities;
  pool->geometry = geometry;
  pool->backing_request = iree_hal_fixed_block_pool_backing_request(&geometry);
  pool->asan_options = options->asan;
  pool->budget_limit = options->budget_limit;
  iree_slim_mutex_initialize(&pool->acquisition_mutex);
  iree_slim_mutex_initialize(&pool->publication.mutex);
  status = iree_hal_pool_history_initialize(
      pool->base.frontier_tracker, geometry.frontier_capacity, &pool->history);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_trace_initialize_pool(
        options->trace_name, IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID, &pool->trace);
  }
  if (iree_status_is_ok(status) && range) {
    iree_async_single_frontier_t compact_storage;
    const iree_async_frontier_t* initial_frontier = NULL;
    status = iree_hal_pool_history_prepare_backing(
        &pool->history, range->memory.reuse_frontier, &compact_storage,
        &initial_frontier);
    if (iree_status_is_ok(status)) {
      iree_hal_fixed_block_pool_slab_t* slab = NULL;
      status = iree_hal_fixed_block_pool_slab_allocate(
          range, &geometry, &options->asan, initial_frontier, host_allocator,
          &slab);
      if (iree_status_is_ok(status)) {
        slab->history_owned = true;
        iree_hal_fixed_block_pool_publish_slab(pool, slab);
      } else {
        iree_hal_pool_history_retire_backing(&pool->history);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_pool = &pool->base;
  } else {
    iree_hal_fixed_block_pool_destroy(&pool->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_fixed_block_pool_create(
    iree_hal_pool_t* backing_pool,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  return iree_hal_fixed_block_pool_create_impl(backing_pool, NULL, options,
                                               host_allocator, out_pool);
}

iree_status_t iree_hal_fixed_block_pool_create_from_buffer(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  if (options->blocks_per_slab) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "finite block capacity comes from its buffer");
  }
  iree_device_size_t alignment = options->alignment ? options->alignment : 1;
  if (iree_hal_asan_pool_options_is_enabled(&options->asan)) {
    alignment =
        iree_max(alignment, iree_max(options->asan.backing_alignment,
                                     options->asan.shadow_granule_size));
  }
  iree_hal_pool_buffer_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_pool_buffer_range_initialize(
      buffer, offset, length, alignment, &options->asan, &range));
  iree_status_t status = iree_hal_fixed_block_pool_create_impl(
      NULL, &range, options, host_allocator, out_pool);
  iree_hal_pool_buffer_range_deinitialize(&range);
  return status;
}

static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_fixed_block_pool_slab_t* slab = pool->slabs.head;
  while (slab) {
    iree_hal_fixed_block_pool_slab_t* next = slab->next;
    IREE_ASSERT(
        iree_atomic_load(&slab->live_count, iree_memory_order_relaxed) == 0,
        "pool destruction requires returned reservations");
    if (pool->backing_pool && !iree_hal_fixed_block_pool_slab_merge_return(
                                  slab, &pool->geometry, &pool->history)) {
      // Destruction of unrepresentable history requires caller quiescence.
      iree_async_frontier_initialize(slab->return_frontier, 0);
    }
    if (slab->history_owned) {
      slab->history_owned = false;
      iree_hal_pool_history_retire_backing(&pool->history);
    }
    iree_hal_fixed_block_pool_slab_free(slab, pool->backing_pool,
                                        pool->host_allocator);
    slab = next;
  }
  if (pool->reusable.slab) {
    iree_hal_fixed_block_pool_slab_free(pool->reusable.slab, NULL,
                                        pool->host_allocator);
  }
  iree_allocator_free(pool->host_allocator, pool->reusable.acquire);
  iree_hal_pool_history_deinitialize(&pool->history);
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_hal_pool_release(pool->backing_pool);
  iree_hal_pool_deinitialize(base_pool);
  iree_slim_mutex_deinitialize(&pool->publication.mutex);
  iree_slim_mutex_deinitialize(&pool->acquisition_mutex);
  iree_allocator_free(pool->host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_fixed_block_pool_validate_reservation_request(
    const iree_hal_fixed_block_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_asan_allocation_layout_t* out_layout) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->min_alignment ? request->min_alignment : 1;
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation size must be > 0");
  }
  if (!iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment (%" PRIdsz
                            ") must be a power of two",
                            alignment);
  }
  if (size > pool->geometry.block_size) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (alignment > pool->geometry.alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reservation alignment exceeds fixed-block support");
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_calculate_asan_layout(
        pool, size, alignment, out_layout));
  }
  return iree_ok_status();
}

// Copies a candidate frontier into transaction-owned storage. Acquisitions
// remain excluded while |source| borrows the allocator's metadata.
static void iree_hal_fixed_block_pool_copy_candidate(
    const iree_hal_memory_fixed_block_allocator_allocation_t* source,
    iree_async_frontier_t* frontier_storage,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_candidate) {
  *out_candidate = *source;
  if (source->death_frontier) {
    memcpy(
        frontier_storage, source->death_frontier,
        sizeof(*frontier_storage) + source->death_frontier->entry_count *
                                        sizeof(frontier_storage->entries[0]));
    out_candidate->death_frontier = frontier_storage;
  }
}

static void iree_hal_fixed_block_pool_replace_candidate_slab(
    iree_hal_fixed_block_pool_t* pool,
    iree_hal_fixed_block_pool_acquire_element_t* element,
    iree_hal_fixed_block_pool_slab_t* slab) {
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  if (slab) {
    ++slab->pins;
  }
  if (element->slab) {
    iree_hal_fixed_block_pool_unpin_slab(pool, element->slab);
  }
  element->slab = slab;
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
}

// Ready candidates fill the prefix and pending fallbacks the suffix. Pins keep
// snapshots valid while eligibility runs unlocked, without claiming blocks or
// publishing temporary scarcity to competing transactions.
static iree_host_size_t iree_hal_fixed_block_pool_select_candidates(
    iree_hal_fixed_block_pool_t* pool, iree_host_size_t request_count,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags, iree_host_size_t frontier_stride,
    uint8_t* frontier_storage,
    iree_hal_fixed_block_pool_acquire_element_t* elements) {
  iree_async_frontier_t* scratch_frontier =
      (iree_async_frontier_t*)(frontier_storage +
                               request_count * frontier_stride);
  iree_host_size_t ready_count = 0;
  iree_host_size_t pending_count = 0;
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  iree_hal_fixed_block_pool_slab_t* slab = pool->slabs.head;
  if (slab) {
    ++slab->pins;
  }
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
  uint32_t start_block_index = 0;
  while (slab && ready_count < request_count) {
    iree_hal_memory_fixed_block_allocator_allocation_t candidate;
    iree_slim_mutex_lock(&pool->acquisition_mutex);
    const bool found = iree_hal_memory_fixed_block_allocator_query_candidate(
        slab->allocator, start_block_index, &candidate);
    if (found) {
      iree_hal_fixed_block_pool_copy_candidate(&candidate, scratch_frontier,
                                               &candidate);
    } else {
      iree_hal_fixed_block_pool_slab_t* next = slab->next;
      if (next) {
        ++next->pins;
      }
      iree_hal_fixed_block_pool_unpin_slab(pool, slab);
      slab = next;
      start_block_index = 0;
    }
    iree_slim_mutex_unlock(&pool->acquisition_mutex);
    if (!found) {
      continue;
    }
    start_block_index = candidate.block_index + 1;

    iree_host_size_t selected_index = 0;
    iree_hal_pool_acquire_result_t result;
    if (iree_hal_fixed_block_pool_frontier_is_satisfied(
            pool, requester_frontier, candidate.death_frontier,
            candidate.block_flags)) {
      if (ready_count + pending_count == request_count) {
        --pending_count;
      }
      selected_index = ready_count++;
      result = candidate.death_frontier ? IREE_HAL_POOL_ACQUIRE_OK
                                        : IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    } else {
      iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                            iree_memory_order_relaxed);
      if (ready_count + pending_count == request_count ||
          !iree_hal_fixed_block_pool_can_wait_for_allocation(flags,
                                                             &candidate)) {
        continue;
      }
      selected_index = request_count - ++pending_count;
      result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
    }
    iree_hal_fixed_block_pool_replace_candidate_slab(
        pool, &elements[selected_index], slab);
    iree_hal_fixed_block_pool_copy_candidate(
        &candidate,
        (iree_async_frontier_t*)(frontier_storage +
                                 selected_index * frontier_stride),
        &elements[selected_index].allocation);
    elements[selected_index].result = result;
  }
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  if (slab) {
    iree_hal_fixed_block_pool_unpin_slab(pool, slab);
  }
  for (iree_host_size_t i = ready_count; i < request_count - pending_count;
       ++i) {
    if (elements[i].slab) {
      iree_hal_fixed_block_pool_unpin_slab(pool, elements[i].slab);
      elements[i].slab = NULL;
    }
  }
  // Compact pending candidates after the ready prefix. Only selection state
  // moves: ASAN layouts belong to request positions, not candidate blocks.
  for (iree_host_size_t i = 0; i < pending_count; ++i) {
    const iree_host_size_t source = request_count - pending_count + i;
    const iree_host_size_t target = ready_count + i;
    if (source == target) {
      continue;
    }
    elements[target].slab = elements[source].slab;
    elements[source].slab = NULL;
    elements[target].result = elements[source].result;
    iree_hal_fixed_block_pool_copy_candidate(
        &elements[source].allocation,
        (iree_async_frontier_t*)(frontier_storage + target * frontier_stride),
        &elements[target].allocation);
  }
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
  iree_hal_fixed_block_pool_return_unused_slabs(pool, 0,
                                                /*retain_metadata=*/true);
  return ready_count + pending_count;
}

// Returns false only when a candidate changed during unlocked selection. A
// failed validation or budget check leaves every candidate available, so there
// is no rollback and no capacity event for a competing acquirer to observe.
static bool iree_hal_fixed_block_pool_commit_candidates(
    iree_hal_fixed_block_pool_t* pool, iree_host_size_t request_count,
    iree_device_size_t charged_length,
    iree_hal_fixed_block_pool_acquire_element_t* elements,
    iree_hal_pool_acquire_result_t* out_result) {
  bool current = true;
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  for (iree_host_size_t i = 0; i < request_count && current; ++i) {
    current = iree_hal_memory_fixed_block_allocator_candidate_is_current(
        elements[i].slab->allocator, &elements[i].allocation);
  }
  if (current) {
    if (iree_hal_fixed_block_pool_try_charge_transaction(pool,
                                                         charged_length)) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        iree_hal_memory_fixed_block_allocator_acquire_candidate(
            elements[i].slab->allocator, elements[i].allocation.block_index,
            &elements[i].allocation);
        iree_atomic_fetch_add(&elements[i].slab->live_count, 1,
                              iree_memory_order_relaxed);
      }
      *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    } else {
      *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    }
  }
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
  return current;
}

// Stack-resident on ready paths; parent acquisition promotes this state once
// and keeps the selected prefix until whole-batch publication or rollback.
typedef struct iree_hal_fixed_block_pool_acquire_t {
  // Pool borrowed through the submitting caller's completion contract.
  iree_hal_fixed_block_pool_t* pool;
  // Captured public transaction arguments.
  struct {
    // Number of request and output array elements.
    iree_host_size_t count;
    // Validated requests, borrowed through terminal completion.
    const iree_hal_pool_reservation_request_t* requests;
    // Requester ordering proof, independent of physical completion.
    const iree_async_frontier_t* frontier;
    // Reservations published only after the complete transaction succeeds.
    iree_hal_pool_reservation_t* reservations;
    // Results and prior-use histories published with completion.
    iree_hal_pool_acquire_info_t* infos;
    // Exactly one accepted terminal completion.
    iree_hal_pool_acquire_callback_t completion;
    // Behavior selected at the public acquisition boundary.
    iree_hal_pool_reserve_flags_t flags;
  } call;
  // Selected blocks, including a claimed prefix and unclaimed snapshots.
  iree_hal_fixed_block_pool_acquire_element_t* elements;
  // Snapshot array plus one scratch frontier for unlocked eligibility checks.
  uint8_t* frontiers;
  // Byte stride of each frontier snapshot.
  iree_host_size_t frontier_stride;
  // Number of already claimed elements, with their charges and histories.
  iree_host_size_t acquired_count;
  // Index identifying the first request a declined transaction cannot satisfy.
  iree_host_size_t unavailable_index;
  // Trailing element capacity, or zero for caller-stack transaction storage.
  iree_host_size_t element_capacity;
  // Prefix capacity was withheld across an unlocked parent preparation.
  bool exposed_prefix;
  // New slabs pinned against trim until this transaction finishes.
  iree_hal_fixed_block_pool_slab_t* preparations;
  // Terminal infrastructure failure, transferred to completion.
  iree_status_t status;
  // Whole-batch capacity outcome; successful items retain individual results.
  iree_hal_pool_acquire_result_t result;
  // Stable parent request output and publication handoff.
  struct {
    // Submission and callback must both finish before resuming.
    iree_atomic_int32_t pending;
    // A parent result is awaiting consumption by the pump.
    bool active;
    // Acquired token consumed by slab setup on success.
    iree_hal_pool_reservation_t reservation;
    // Exact prior-use history, borrowed from the acquired token.
    iree_hal_pool_acquire_info_t info;
    // Parent capacity result, published before the completion handoff.
    iree_hal_pool_acquire_result_t result;
  } parent;
} iree_hal_fixed_block_pool_acquire_t;

// Allocates durable transaction storage on a cold path. Promotion follows a
// successful claim, so all retained histories already point at stable blocks.
static iree_status_t iree_hal_fixed_block_pool_acquire_promote(
    iree_hal_fixed_block_pool_acquire_t** inout_acquire) {
  iree_hal_fixed_block_pool_acquire_t* source = *inout_acquire;
  iree_host_size_t total_size = 0;
  iree_host_size_t elements_offset = 0;
  iree_host_size_t frontiers_offset = 0;
  const iree_host_size_t capacity =
      iree_max(source->call.count,
               IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY);
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*source), &total_size,
      IREE_STRUCT_FIELD(capacity, iree_hal_fixed_block_pool_acquire_element_t,
                        &elements_offset),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(capacity, source->frontier_stride,
                                      uint8_t, IREE_ASYNC_FRONTIER_ALIGNMENT,
                                      &frontiers_offset),
      IREE_STRUCT_FIELD(source->frontier_stride, uint8_t, NULL)));
  iree_hal_fixed_block_pool_acquire_t* acquire = NULL;
  iree_host_size_t allocation_capacity = capacity;
  iree_slim_mutex_lock(&source->pool->acquisition_mutex);
  if (source->pool->reusable.acquire &&
      source->pool->reusable.acquire->element_capacity >= capacity) {
    acquire = source->pool->reusable.acquire;
    source->pool->reusable.acquire = NULL;
    allocation_capacity = acquire->element_capacity;
  }
  iree_slim_mutex_unlock(&source->pool->acquisition_mutex);
  if (!acquire) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(source->pool->host_allocator,
                                               total_size, (void**)&acquire));
  }
  *acquire = *source;
  acquire->element_capacity = allocation_capacity;
  acquire->elements =
      (iree_hal_fixed_block_pool_acquire_element_t*)((uint8_t*)acquire +
                                                     elements_offset);
  acquire->frontiers = (uint8_t*)acquire + frontiers_offset;
  // Large transactions allocate before initialization; promotion after a miss
  // copies the already initialized elements and their acquired histories.
  if (source->elements) {
    memcpy(acquire->elements, source->elements,
           source->call.count * sizeof(*source->elements));
  }
  *inout_acquire = acquire;
  return iree_ok_status();
}

static void iree_hal_fixed_block_pool_acquire_finish(
    iree_hal_fixed_block_pool_acquire_t* acquire) {
  iree_hal_fixed_block_pool_t* pool = acquire->pool;
  if (iree_status_is_ok(acquire->status)) {
    acquire->status = iree_hal_pool_history_seal_acquire(&pool->history);
  }
  const bool committed = iree_status_is_ok(acquire->status) &&
                         acquire->acquired_count == acquire->call.count;
  if (iree_status_is_ok(acquire->status)) {
    if (committed) {
      acquire->result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
      for (iree_host_size_t i = 0; i < acquire->call.count; ++i) {
        iree_hal_fixed_block_pool_acquire_element_t* element =
            &acquire->elements[i];
        iree_hal_fixed_block_pool_return_allocation(
            pool, element->slab, &element->allocation,
            acquire->call.requests[i].allocation_size, &element->asan_layout,
            element->result, &acquire->call.reservations[i],
            &acquire->call.infos[i]);
        if (element->result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ||
            (element->result == IREE_HAL_POOL_ACQUIRE_OK &&
             acquire->result == IREE_HAL_POOL_ACQUIRE_OK_FRESH)) {
          acquire->result = element->result;
        }
      }
    } else {
      iree_atomic_fetch_add(acquire->result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED
                                ? &pool->exhausted_count
                                : &pool->over_budget_count,
                            1, iree_memory_order_relaxed);
      memset(acquire->call.infos, 0,
             acquire->call.count * sizeof(*acquire->call.infos));
      acquire->call.infos[acquire->unavailable_index] =
          (iree_hal_pool_acquire_info_t){
              .result = acquire->result,
          };
    }
  }
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  if (!committed) {
    for (iree_host_size_t i = 0; i < acquire->acquired_count; ++i) {
      iree_hal_fixed_block_pool_acquire_element_t* element =
          &acquire->elements[i];
      iree_hal_memory_fixed_block_allocator_restore(
          element->slab->allocator, element->allocation.block_index);
      iree_atomic_fetch_sub(&element->slab->live_count, 1,
                            iree_memory_order_relaxed);
    }
    iree_hal_fixed_block_pool_uncharge_reservation(
        pool, acquire->acquired_count * pool->geometry.backing_block_size);
  }
  for (iree_host_size_t i = 0; i < acquire->call.count; ++i) {
    if (acquire->elements[i].slab) {
      iree_hal_fixed_block_pool_unpin_slab(pool, acquire->elements[i].slab);
    }
  }
  while (acquire->preparations) {
    iree_hal_fixed_block_pool_slab_t* slab = acquire->preparations;
    acquire->preparations = slab->preparation_next;
    iree_hal_fixed_block_pool_unpin_slab(pool, slab);
  }
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
  iree_hal_fixed_block_pool_return_unused_slabs(pool, 0,
                                                /*retain_metadata=*/true);
  if (!committed && acquire->exposed_prefix) {
    iree_async_notification_signal_if_observed(pool->base.notification,
                                               INT32_MAX);
  }
  const iree_hal_pool_acquire_callback_t completion = acquire->call.completion;
  const iree_status_t status = acquire->status;
  const iree_hal_pool_acquire_result_t result = acquire->result;
  if (acquire->element_capacity) {
    iree_hal_fixed_block_pool_acquire_t* discarded = acquire;
    iree_slim_mutex_lock(&pool->acquisition_mutex);
    if (!pool->reusable.acquire ||
        pool->reusable.acquire->element_capacity < acquire->element_capacity) {
      discarded = pool->reusable.acquire;
      pool->reusable.acquire = acquire;
    }
    iree_slim_mutex_unlock(&pool->acquisition_mutex);
    iree_allocator_free(pool->host_allocator, discarded);
  }
  completion.fn(completion.user_data, status, result);
}

// The mutual completion/pump recursion is broken for inline completions by
// joining the callback with the return from its submission.
static void iree_hal_fixed_block_pool_acquire_progress(
    iree_hal_fixed_block_pool_acquire_t* acquire);

static void iree_hal_fixed_block_pool_acquire_parent_complete(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_fixed_block_pool_acquire_t* acquire = user_data;
  acquire->status = status;
  acquire->parent.result = result;
  if (iree_atomic_fetch_sub(&acquire->parent.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_fixed_block_pool_acquire_progress(acquire);
  }
}

static void iree_hal_fixed_block_pool_acquire_progress(
    iree_hal_fixed_block_pool_acquire_t* acquire) {
  iree_hal_fixed_block_pool_t* pool = acquire->pool;
  while (iree_status_is_ok(acquire->status)) {
    if (acquire->parent.active) {
      acquire->parent.active = false;
      acquire->result = acquire->parent.result;
      if (acquire->result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
          acquire->result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        break;
      }
      iree_async_single_frontier_t compact_storage;
      const iree_async_frontier_t* initial_frontier = NULL;
      acquire->status = iree_hal_pool_history_prepare_backing(
          &pool->history, acquire->parent.info.reuse_frontier, &compact_storage,
          &initial_frontier);
      if (!iree_status_is_ok(acquire->status)) {
        iree_hal_pool_release_reservations(pool->backing_pool, 1,
                                           &acquire->parent.reservation,
                                           acquire->parent.info.reuse_frontier);
        break;
      }
      iree_hal_fixed_block_pool_slab_t* slab = NULL;
      iree_slim_mutex_lock(&pool->acquisition_mutex);
      iree_hal_fixed_block_pool_slab_t* reusable_slab = pool->reusable.slab;
      pool->reusable.slab = NULL;
      iree_slim_mutex_unlock(&pool->acquisition_mutex);
      acquire->status = iree_hal_fixed_block_pool_slab_prepare(
          pool->backing_pool, &pool->backing_request, &pool->geometry,
          &pool->asan_options, acquire->parent.reservation,
          acquire->parent.info, initial_frontier, pool->host_allocator,
          reusable_slab, &slab);
      if (!iree_status_is_ok(acquire->status)) {
        iree_hal_pool_history_retire_backing(&pool->history);
        break;
      }
      slab->history_owned = true;
      slab->pins = 1;
      slab->preparation_next = acquire->preparations;
      acquire->preparations = slab;
      iree_slim_mutex_lock(&pool->acquisition_mutex);
      iree_hal_fixed_block_pool_publish_slab(pool, slab);
      iree_slim_mutex_unlock(&pool->acquisition_mutex);
      iree_async_notification_signal_if_observed(pool->base.notification,
                                                 INT32_MAX);
    }
    const iree_host_size_t remaining =
        acquire->call.count - acquire->acquired_count;
    const iree_device_size_t remaining_charge =
        remaining * pool->geometry.backing_block_size;
    const iree_device_size_t reserved = (iree_device_size_t)iree_atomic_load(
        &pool->bytes_reserved, iree_memory_order_relaxed);
    if (pool->budget_limit &&
        remaining_charge >
            pool->budget_limit - iree_min(reserved, pool->budget_limit)) {
      acquire->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
      acquire->unavailable_index = 0;
      break;
    }
    iree_hal_fixed_block_pool_acquire_element_t* elements =
        acquire->elements + acquire->acquired_count;
    const iree_host_size_t selected_count =
        iree_hal_fixed_block_pool_select_candidates(
            pool, remaining, acquire->call.frontier, acquire->call.flags,
            acquire->frontier_stride,
            acquire->frontiers +
                acquire->acquired_count * acquire->frontier_stride,
            elements);
    if (selected_count < remaining && !pool->backing_pool) {
      acquire->result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      acquire->unavailable_index = acquire->acquired_count + selected_count;
      break;
    }
    if (selected_count) {
      if (!iree_hal_fixed_block_pool_commit_candidates(
              pool, selected_count,
              selected_count * pool->geometry.backing_block_size, elements,
              &acquire->result)) {
        continue;
      }
      if (acquire->result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        acquire->unavailable_index = 0;
        break;
      }
      acquire->acquired_count += selected_count;
    }
    if (acquire->acquired_count == acquire->call.count) {
      break;
    }
    acquire->unavailable_index = acquire->acquired_count;
    acquire->exposed_prefix |= acquire->acquired_count != 0;
    if (!acquire->element_capacity) {
      iree_status_t status =
          iree_hal_fixed_block_pool_acquire_promote(&acquire);
      acquire->status = status;
      if (!iree_status_is_ok(acquire->status)) {
        break;
      }
    }
    acquire->parent.active = true;
    acquire->parent.info = (iree_hal_pool_acquire_info_t){0};
    iree_atomic_store(&acquire->parent.pending, 2, iree_memory_order_relaxed);
    iree_status_t status = iree_hal_pool_acquire_reservations(
        pool->backing_pool, 1, &pool->backing_request, acquire->call.frontier,
        &acquire->parent.reservation, &acquire->parent.info,
        (iree_hal_pool_acquire_callback_t){
            .fn = iree_hal_fixed_block_pool_acquire_parent_complete,
            .user_data = acquire,
        },
        acquire->call.flags);
    if (!iree_status_is_ok(status)) {
      iree_hal_fixed_block_pool_acquire_parent_complete(
          acquire, status, IREE_HAL_POOL_ACQUIRE_NONE);
    }
    if (iree_atomic_fetch_sub(&acquire->parent.pending, 1,
                              iree_memory_order_acq_rel) != 1) {
      return;
    }
  }
  iree_hal_fixed_block_pool_acquire_finish(acquire);
}

static iree_status_t iree_hal_fixed_block_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_fixed_block_pool_acquire_element_t
      inline_elements[IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY];
  iree_alignas(IREE_ASYNC_FRONTIER_ALIGNMENT) uint8_t
      inline_frontiers[(IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY +
                        1) *
                       (sizeof(iree_async_frontier_t) +
                        IREE_HAL_FIXED_BLOCK_POOL_INLINE_FRONTIER_CAPACITY *
                            sizeof(iree_async_frontier_entry_t))];
  const bool needs_storage =
      request_count > IREE_ARRAYSIZE(inline_elements) ||
      pool->geometry.frontier_capacity >
          IREE_HAL_FIXED_BLOCK_POOL_INLINE_FRONTIER_CAPACITY;
  iree_hal_fixed_block_pool_acquire_t inline_acquire = {
      .pool = pool,
      .call =
          {
              .count = request_count,
              .requests = requests,
              .frontier = requester_frontier,
              .reservations = out_reservations,
              .infos = out_infos,
              .completion = completion,
              .flags = flags,
          },
      .elements = needs_storage ? NULL : inline_elements,
      .frontiers = inline_frontiers,
      .frontier_stride = sizeof(iree_async_frontier_t) +
                         pool->geometry.frontier_capacity *
                             sizeof(iree_async_frontier_entry_t),
  };
  iree_hal_fixed_block_pool_acquire_t* acquire = &inline_acquire;
  if (needs_storage) {
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_acquire_promote(&acquire));
  }
  memset(acquire->elements, 0, request_count * sizeof(*acquire->elements));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_fixed_block_pool_validate_reservation_request(
        pool, &requests[i], &acquire->elements[i].asan_layout);
  }
  iree_device_size_t charged_length = 0;
  if (iree_status_is_ok(status) &&
      !iree_device_size_checked_mul(
          request_count, pool->geometry.backing_block_size, &charged_length)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "fixed-block transaction charge overflows");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_fixed_block_pool_acquire_progress(acquire);
  } else if (acquire->element_capacity) {
    iree_allocator_free(pool->host_allocator, acquire);
  }
  return status;
}

static void iree_hal_fixed_block_pool_release_one_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_fixed_block_pool_block_t* block =
      (iree_hal_fixed_block_pool_block_t*)(uintptr_t)reservation->block_handle;
  iree_hal_fixed_block_pool_slab_t* slab = block->slab;
  const uint32_t block_index = (uint32_t)(block - slab->blocks);
  iree_hal_memory_trace_free(&pool->trace, block);
  iree_async_single_frontier_t compact_storage;
  if (death_frontier &&
      death_frontier->entry_count > pool->geometry.frontier_capacity &&
      pool->history.timeline) {
    iree_async_frontier_t* compact =
        iree_async_single_frontier_as_frontier(&compact_storage);
    iree_async_frontier_initialize(compact, 0);
    iree_hal_pool_history_collapse(&pool->history, compact, death_frontier);
    death_frontier = compact;
  }
  iree_hal_memory_fixed_block_allocator_release(slab->allocator, block_index,
                                                death_frontier);
  // The raw bitmap is reusable now; the live count keeps this final slab
  // access ordered before cold detachment, including immediate block reuse.
  iree_slim_mutex_lock(&pool->publication.mutex);
  const int32_t previous_live_count =
      iree_atomic_fetch_sub(&slab->live_count, 1, iree_memory_order_relaxed);
  if (pool->backing_pool && previous_live_count == 1) {
    iree_hal_fixed_block_pool_note_candidate(pool, slab);
  }
  iree_slim_mutex_unlock(&pool->publication.mutex);
  iree_hal_fixed_block_pool_uncharge_reservation(
      pool, pool->geometry.backing_block_size);
  iree_atomic_fetch_sub(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
}

static void iree_hal_fixed_block_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_fixed_block_pool_release_one_reservation(
        base_pool, &reservations[i], death_frontier);
  }
  iree_hal_fixed_block_pool_return_unused_slabs(pool, 0,
                                                /*retain_metadata=*/true);
  iree_async_notification_signal_if_observed(pool->base.notification,
                                             INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_fixed_block_pool_advise_asan_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_fixed_block_pool_block_t* block =
        (iree_hal_fixed_block_pool_block_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_fixed_block_pool_slab_t* slab = block->slab;
    const uint32_t block_index = (uint32_t)(block - slab->blocks);
    iree_hal_pool_buffer_range_advise_asan(
        &slab->range, block_index * pool->geometry.backing_block_size, flags,
        &slab->asan_layouts[block_index]);
  }
}

static void iree_hal_fixed_block_pool_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_fixed_block_pool_materialize_element_t* element =
      (iree_hal_fixed_block_pool_materialize_element_t*)user_data;
  iree_hal_fixed_block_pool_materialize_state_t* state = element->state;
  if (state->ownership_committed) {
    iree_hal_pool_advise_asan_reservations(
        state->pool, 1, &element->reservation,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    iree_hal_pool_release_reservations(state->pool, 1, &element->reservation,
                                       NULL);
  }
  const int32_t previous_count = iree_atomic_fetch_sub(
      &state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count == 1) {
    iree_allocator_free(state->host_allocator, state);
  }
}

static void iree_hal_fixed_block_pool_query_reservation_views(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* out_views) {
  (void)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_fixed_block_pool_block_t* block =
        (iree_hal_fixed_block_pool_block_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_fixed_block_pool_slab_t* slab = block->slab;
    iree_hal_pool_buffer_range_query_reservation_view(
        &slab->range, reservations[i].offset, reservations[i].byte_length,
        iree_hal_memory_fixed_block_allocator_block_death_frontier(
            slab->allocator, (uint32_t)(block - slab->blocks)),
        &out_views[i]);
  }
}

static iree_status_t iree_hal_fixed_block_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;

  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    if (reservations[i].byte_length < requests[i].allocation_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has %" PRIdsz
          " bytes but its allocation request requires %" PRIdsz,
          i, reservations[i].byte_length, requests[i].allocation_size);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_hal_fixed_block_pool_materialize_state_t* state = NULL;
  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY] = {
          0};
  iree_hal_buffer_t** staged_buffers = inline_buffers;
  bool staged_buffers_allocated = false;
  iree_status_t status = iree_ok_status();
  if (transfer_ownership) {
    if (reservation_count > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization count exceeds INT32_MAX");
    }
    iree_host_size_t state_size = 0;
    if (!iree_host_size_checked_mul_add(
            reservation_count,
            sizeof(iree_hal_fixed_block_pool_materialize_element_t),
            sizeof(*state), &state_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization state size overflow");
    }
    status =
        iree_allocator_malloc(pool->host_allocator, state_size, (void**)&state);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    memset(state, 0, state_size);
    state->pool = base_pool;
    state->host_allocator = pool->host_allocator;
    iree_atomic_store(&state->reference_count, (int32_t)reservation_count,
                      iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      state->elements[i].state = state;
      state->elements[i].reservation = reservations[i];
    }
  } else if (reservation_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(
        pool->host_allocator, reservation_count, sizeof(*staged_buffers),
        (void**)&staged_buffers);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    staged_buffers_allocated = true;
    memset(staged_buffers, 0, reservation_count * sizeof(*staged_buffers));
  }

  iree_host_size_t materialized_count = 0;
  while (materialized_count < reservation_count && iree_status_is_ok(status)) {
    iree_hal_buffer_release_callback_t release_callback =
        iree_hal_buffer_release_callback_null();
    iree_hal_buffer_t** staged_buffer = &staged_buffers[materialized_count];
    if (state) {
      iree_hal_fixed_block_pool_materialize_element_t* element =
          &state->elements[materialized_count];
      release_callback.fn = iree_hal_fixed_block_pool_buffer_release;
      release_callback.user_data = element;
      staged_buffer = &element->buffer;
    }
    iree_hal_fixed_block_pool_block_t* block =
        (iree_hal_fixed_block_pool_block_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    iree_hal_fixed_block_pool_slab_t* slab = block->slab;
    status = iree_hal_pool_buffer_range_materialize(
        &slab->range, reservations[materialized_count].offset,
        reservations[materialized_count].byte_length,
        iree_hal_memory_fixed_block_allocator_block_death_frontier(
            slab->allocator, (uint32_t)(block - slab->blocks)),
        release_callback, pool->host_allocator, staged_buffer);
    if (iree_status_is_ok(status)) {
      ++materialized_count;
    }
  }
  if (iree_status_is_ok(status)) {
    if (state) {
      state->ownership_committed = true;
    }
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      out_buffers[i] = state ? state->elements[i].buffer : staged_buffers[i];
    }
  } else {
    if (state) {
      iree_atomic_store(&state->reference_count, (int32_t)materialized_count,
                        iree_memory_order_relaxed);
    }
    for (iree_host_size_t i = 0; i < materialized_count; ++i) {
      iree_hal_buffer_release(state ? state->elements[i].buffer
                                    : staged_buffers[i]);
    }
    if (state && materialized_count == 0) {
      iree_allocator_free(pool->host_allocator, state);
    }
  }
  if (staged_buffers_allocated) {
    iree_allocator_free(pool->host_allocator, staged_buffers);
  }
  return status;
}

static void iree_hal_fixed_block_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  *out_capabilities = pool->capabilities;
  out_capabilities->min_allocation_size = 1;
  out_capabilities->max_allocation_size = pool->geometry.block_size;
  out_capabilities->max_allocation_alignment = pool->geometry.alignment;
}

static iree_status_t iree_hal_fixed_block_pool_validate_asan(
    const iree_hal_pool_t* base_pool,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  if (pool->backing_pool) {
    return iree_hal_pool_validate_asan_options(pool->backing_pool, options);
  }
  const iree_hal_buffer_range_advice_t* advice =
      pool->slabs.head->range.memory.backing->advice;
  if (!advice) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer has no native ASAN range advice");
  }
  return advice->validate_asan(advice->user_data, options);
}

static void iree_hal_fixed_block_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  const iree_device_size_t managed_bytes = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_managed, iree_memory_order_relaxed);
  out_stats->bytes_free =
      managed_bytes - iree_min(managed_bytes, out_stats->bytes_reserved);
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->budget_limit = pool->budget_limit;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count =
      (uint32_t)iree_atomic_load(&pool->slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count =
      (uint64_t)iree_atomic_load(&pool->reuse_count, iree_memory_order_relaxed);
  out_stats->reuse_miss_count = (uint64_t)iree_atomic_load(
      &pool->reuse_miss_count, iree_memory_order_relaxed);
  out_stats->fresh_count =
      (uint64_t)iree_atomic_load(&pool->fresh_count, iree_memory_order_relaxed);
  out_stats->exhausted_count = (uint64_t)iree_atomic_load(
      &pool->exhausted_count, iree_memory_order_relaxed);
  out_stats->over_budget_count = (uint64_t)iree_atomic_load(
      &pool->over_budget_count, iree_memory_order_relaxed);
  out_stats->wait_count =
      (uint64_t)iree_atomic_load(&pool->wait_count, iree_memory_order_relaxed);
}

static void iree_hal_fixed_block_pool_trim(
    iree_hal_pool_t* base_pool, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_slim_mutex_lock(&pool->acquisition_mutex);
  if (pool->backing_pool) {
    iree_slim_mutex_lock(&pool->publication.mutex);
    for (iree_hal_fixed_block_pool_slab_t* slab = pool->slabs.head; slab;
         slab = slab->next) {
      iree_hal_fixed_block_pool_note_candidate(pool, slab);
    }
    iree_slim_mutex_unlock(&pool->publication.mutex);
  }
  iree_hal_fixed_block_pool_slab_t* reusable_slab = pool->reusable.slab;
  iree_hal_fixed_block_pool_acquire_t* reusable_acquire =
      pool->reusable.acquire;
  pool->reusable.slab = NULL;
  pool->reusable.acquire = NULL;
  iree_slim_mutex_unlock(&pool->acquisition_mutex);
  if (pool->backing_pool) {
    iree_hal_fixed_block_pool_return_unused_slabs(pool, min_bytes_to_keep,
                                                  /*retain_metadata=*/false);
  }
  if (reusable_slab) {
    iree_hal_fixed_block_pool_slab_free(reusable_slab, NULL,
                                        pool->host_allocator);
  }
  iree_allocator_free(pool->host_allocator, reusable_acquire);
  iree_hal_pool_history_trim(&pool->history);
  (void)flags;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable = {
    .destroy = iree_hal_fixed_block_pool_destroy,
    .acquire_reservations = iree_hal_fixed_block_pool_acquire_reservations,
    .release_reservations = iree_hal_fixed_block_pool_release_reservations,
    .query_reservation_views =
        iree_hal_fixed_block_pool_query_reservation_views,
    .materialize_reservations =
        iree_hal_fixed_block_pool_materialize_reservations,
    .query_capabilities = iree_hal_fixed_block_pool_query_capabilities,
    .validate_asan = iree_hal_fixed_block_pool_validate_asan,
    .query_stats = iree_hal_fixed_block_pool_query_stats,
    .trim = iree_hal_fixed_block_pool_trim,
    .advise_asan_reservations =
        iree_hal_fixed_block_pool_advise_asan_reservations,
};
