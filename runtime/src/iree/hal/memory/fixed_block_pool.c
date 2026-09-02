// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/hal/memory/tracing.h"

enum { IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY = 8 };

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_fixed_block_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Provider backing the single fixed-block slab.
  iree_hal_slab_provider_t* slab_provider;

  // Notification signaled when a reservation release may unblock waiters.
  iree_async_notification_t* notification;

  // Lock-free offset allocator for fixed-size blocks within |slab|.
  iree_hal_memory_fixed_block_allocator_t* block_allocator;

  // Physical memory backing all fixed blocks.
  iree_hal_slab_t slab;

  // Optional completion predicate for try-before-fence reuse.
  iree_hal_pool_epoch_query_t epoch_query;

  // Host allocator used for pool metadata.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Immutable memory properties provided by |slab_provider|.
  iree_hal_slab_provider_properties_t slab_properties;

  // User-visible byte capacity of each fixed block.
  iree_device_size_t user_block_size;

  // Backing byte size of every block in |block_allocator|.
  iree_device_size_t backing_block_size;

  // Number of blocks managed by |block_allocator|.
  uint32_t block_count;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // ASAN layout for each live block. NULL when ASAN is disabled.
  iree_hal_asan_allocation_layout_t* asan_block_layouts;

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
  // Reservation produced for the request.
  iree_hal_pool_reservation_t reservation;

  // Acquisition metadata produced for the request.
  iree_hal_pool_acquire_info_t info;
} iree_hal_fixed_block_pool_acquire_element_t;

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable;
static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID =
    "iree-hal-fixed-block-pool";

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
  if (!pool->epoch_query.fn) return false;

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
    if (!pool->epoch_query.fn(pool->epoch_query.user_data, axis, epoch)) {
      return false;
    }
  }
  return true;
}

static void iree_hal_fixed_block_pool_restore_rejected_blocks(
    iree_hal_fixed_block_pool_t* pool, uint32_t rejected_block_count,
    const uint32_t* rejected_block_indices) {
  for (uint32_t i = 0; i < rejected_block_count; ++i) {
    iree_hal_memory_fixed_block_allocator_restore(pool->block_allocator,
                                                  rejected_block_indices[i]);
  }
}

static bool iree_hal_fixed_block_pool_try_charge_reservation(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t charged_length) {
  if (pool->budget_limit == 0) {
    iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)charged_length,
                          iree_memory_order_relaxed);
    return true;
  }
  int64_t expected =
      iree_atomic_load(&pool->bytes_reserved, iree_memory_order_relaxed);
  for (;;) {
    const iree_device_size_t current = (iree_device_size_t)expected;
    if (current > pool->budget_limit ||
        charged_length > pool->budget_limit - current) {
      return false;
    }
    const int64_t desired = (int64_t)(current + charged_length);
    if (iree_atomic_compare_exchange_weak(&pool->bytes_reserved, &expected,
                                          desired, iree_memory_order_relaxed,
                                          iree_memory_order_relaxed)) {
      return true;
    }
  }
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

static iree_device_size_t iree_hal_fixed_block_pool_max_user_alignment(
    iree_device_size_t user_block_size) {
  IREE_ASSERT(user_block_size > 0);
  return (iree_device_size_t)1
         << iree_math_count_trailing_zeros_u64(user_block_size);
}

static iree_status_t iree_hal_fixed_block_pool_calculate_asan_layout(
    const iree_hal_fixed_block_pool_t* pool, iree_device_size_t user_length,
    iree_device_size_t user_alignment,
    iree_hal_asan_allocation_layout_t* out_layout) {
  IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
      &pool->asan_options, user_length, user_alignment, out_layout));
  return iree_hal_asan_extend_allocation_layout(pool->backing_block_size,
                                                out_layout);
}

static iree_status_t iree_hal_fixed_block_pool_return_allocation(
    iree_hal_fixed_block_pool_t* pool,
    const iree_hal_memory_fixed_block_allocator_allocation_t* allocation,
    iree_device_size_t byte_length,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  const bool asan_enabled =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options);
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset =
      allocation->offset + (asan_enabled ? asan_layout->user_offset : 0);
  out_reservation->byte_length = byte_length;
  out_reservation->block_handle = allocation->block_index;
  out_reservation->slab_index = 0;

  // Tainted blocks never reach this helper: frontier_is_satisfied rejects
  // them (so they never become OK/OK_FRESH) and can_wait_for_allocation
  // excludes them (so they never become OK_NEEDS_WAIT). Only NEEDS_WAIT
  // returns a wait_frontier here.
  memset(out_info, 0, sizeof(*out_info));
  if (result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    out_info->wait_frontier = allocation->death_frontier;
  }
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
    pool->asan_block_layouts[allocation->block_index] = *asan_layout;
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &pool->slab, allocation->offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED, asan_layout);
  }
  iree_hal_memory_trace_alloc(
      &pool->trace, (uint8_t*)pool->slab.base_ptr + out_reservation->offset,
      out_reservation->byte_length);
  *out_result = result;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_create(
    iree_hal_fixed_block_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_slab_provider_validate_asan_options(slab_provider,
                                                       &options.asan));

  iree_hal_memory_fixed_block_allocator_options_t block_allocator_options =
      options.block_allocator_options;
  iree_device_size_t user_block_size = block_allocator_options.block_size;
  iree_device_size_t backing_block_size = user_block_size;
  if (iree_hal_asan_pool_options_is_enabled(&options.asan)) {
    if (user_block_size == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "block_size must be > 0");
    }
    const iree_device_size_t max_user_alignment =
        iree_hal_fixed_block_pool_max_user_alignment(user_block_size);
    iree_hal_asan_allocation_layout_t block_layout;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_asan_calculate_allocation_layout(
            &options.asan, user_block_size, max_user_alignment, &block_layout));
    if (!iree_device_size_checked_align(block_layout.backing_length,
                                        block_layout.backing_offset_alignment,
                                        &backing_block_size)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "fixed-block ASAN backing block size overflows aligning %" PRIu64
          " bytes to %" PRIu64,
          (uint64_t)block_layout.backing_length,
          (uint64_t)block_layout.backing_offset_alignment);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_asan_extend_allocation_layout(backing_block_size,
                                                   &block_layout));
    block_allocator_options.block_size = backing_block_size;
  }

  iree_device_size_t slab_length = 0;
  if (!iree_device_size_checked_mul(block_allocator_options.block_count,
                                    block_allocator_options.block_size,
                                    &slab_length)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixed-block pool slab length overflows: block_count=%u "
        "block_size=%" PRIdsz,
        (unsigned)block_allocator_options.block_count,
        block_allocator_options.block_size);
  }

  iree_hal_fixed_block_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_pool_initialize(&iree_hal_fixed_block_pool_vtable, &pool->base);
  pool->host_allocator = host_allocator;
  pool->epoch_query = epoch_query;
  pool->user_block_size = user_block_size;
  pool->backing_block_size = backing_block_size;
  pool->block_count = block_allocator_options.block_count;
  pool->asan_options = options.asan;
  pool->budget_limit = options.budget_limit;

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  iree_hal_slab_provider_query_properties(slab_provider,
                                          &pool->slab_properties);

  iree_status_t status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID, host_allocator,
      &pool->trace);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_fixed_block_allocator_allocate(
        block_allocator_options, pool->host_allocator, &pool->block_allocator);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    status = iree_allocator_malloc_array(
        pool->host_allocator, pool->block_count,
        sizeof(*pool->asan_block_layouts), (void**)&pool->asan_block_layouts);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_slab_provider_acquire_slab(pool->slab_provider,
                                                 slab_length, &pool->slab);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_fixed_block_pool_destroy((iree_hal_pool_t*)pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_pool = (iree_hal_pool_t*)pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_memory_fixed_block_allocator_free(pool->block_allocator);
  if (pool->slab.length > 0) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &pool->slab);
  }
  iree_allocator_free(pool->host_allocator, pool->asan_block_layouts);
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_fixed_block_pool_validate_reservation_request(
    const iree_hal_fixed_block_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;
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
  if (size > pool->user_block_size) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (alignment > pool->user_block_size ||
      (pool->user_block_size % alignment) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " is incompatible with fixed block size %" PRIdsz,
                            alignment, pool->user_block_size);
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_asan_allocation_layout_t asan_layout;
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_calculate_asan_layout(
        pool, size, alignment, &asan_layout));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_fixed_block_pool_acquire_one_reservation(
    iree_hal_pool_t* base_pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;

  iree_hal_asan_allocation_layout_t asan_layout = {0};
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_calculate_asan_layout(
        pool, size, alignment, &asan_layout));
  }

  if (!iree_hal_fixed_block_pool_try_charge_reservation(
          pool, pool->backing_block_size)) {
    iree_atomic_fetch_add(&pool->over_budget_count, 1,
                          iree_memory_order_relaxed);
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    out_info->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    return iree_ok_status();
  }

  uint32_t
      rejected_block_indices[IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS];
  uint32_t rejected_block_count = 0;
  iree_hal_memory_fixed_block_allocator_allocation_t selected_allocation;
  iree_hal_pool_acquire_result_t selected_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  bool has_selected_allocation = false;
  iree_hal_memory_fixed_block_allocator_allocation_t wait_allocation;
  bool has_wait_allocation = false;
  iree_status_t status = iree_ok_status();

  while (iree_status_is_ok(status) && !has_selected_allocation) {
    iree_hal_memory_fixed_block_allocator_allocation_t allocation;
    iree_hal_memory_fixed_block_allocator_acquire_result_t allocation_result =
        IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
    status = iree_hal_memory_fixed_block_allocator_try_acquire(
        pool->block_allocator, &allocation, &allocation_result);
    if (iree_status_is_ok(status) &&
        allocation_result ==
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED) {
      if (has_wait_allocation) {
        selected_allocation = wait_allocation;
        selected_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
        has_selected_allocation = true;
        has_wait_allocation = false;
      } else {
        selected_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      }
      break;
    }
    if (!iree_status_is_ok(status)) break;

    const bool frontier_is_satisfied =
        iree_hal_fixed_block_pool_frontier_is_satisfied(
            pool, requester_frontier, allocation.death_frontier,
            allocation.block_flags);
    if (!frontier_is_satisfied) {
      iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                            iree_memory_order_relaxed);
      if (!has_wait_allocation &&
          iree_hal_fixed_block_pool_can_wait_for_allocation(flags,
                                                            &allocation)) {
        wait_allocation = allocation;
        has_wait_allocation = true;
        continue;
      }
      rejected_block_indices[rejected_block_count++] = allocation.block_index;
      continue;
    }

    selected_allocation = allocation;
    selected_result = allocation.death_frontier
                          ? IREE_HAL_POOL_ACQUIRE_OK
                          : IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    has_selected_allocation = true;
  }

  if (iree_status_is_ok(status)) {
    iree_hal_fixed_block_pool_restore_rejected_blocks(
        pool, rejected_block_count, rejected_block_indices);
    rejected_block_count = 0;
    if (has_wait_allocation) {
      iree_hal_memory_fixed_block_allocator_restore(
          pool->block_allocator, wait_allocation.block_index);
      has_wait_allocation = false;
    }
    if (has_selected_allocation) {
      status = iree_hal_fixed_block_pool_return_allocation(
          pool, &selected_allocation, size, &asan_layout, selected_result,
          out_reservation, out_info, out_result);
      if (!iree_status_is_ok(status)) {
        iree_hal_fixed_block_pool_uncharge_reservation(
            pool, pool->backing_block_size);
        iree_hal_memory_fixed_block_allocator_restore(
            pool->block_allocator, selected_allocation.block_index);
      }
    } else {
      iree_atomic_fetch_add(&pool->exhausted_count, 1,
                            iree_memory_order_relaxed);
      memset(out_reservation, 0, sizeof(*out_reservation));
      memset(out_info, 0, sizeof(*out_info));
      out_info->result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      iree_hal_fixed_block_pool_uncharge_reservation(pool,
                                                     pool->backing_block_size);
    }
  } else {
    iree_hal_fixed_block_pool_restore_rejected_blocks(
        pool, rejected_block_count, rejected_block_indices);
    if (has_wait_allocation) {
      iree_hal_memory_fixed_block_allocator_restore(
          pool->block_allocator, wait_allocation.block_index);
    }
    iree_hal_fixed_block_pool_uncharge_reservation(pool,
                                                   pool->backing_block_size);
  }
  return status;
}

// Rolls back a reservation acquired by the current transaction before it was
// made visible to the caller.
static void iree_hal_fixed_block_pool_rollback_reservation(
    iree_hal_fixed_block_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* info) {
  const uint32_t block_index = (uint32_t)reservation->block_handle;
  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)pool->slab.base_ptr + reservation->offset);
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    const iree_device_size_t backing_offset =
        (iree_device_size_t)block_index * pool->backing_block_size;
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &pool->slab, backing_offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED,
        &pool->asan_block_layouts[block_index]);
    memset(&pool->asan_block_layouts[block_index], 0,
           sizeof(pool->asan_block_layouts[block_index]));
  }
  iree_hal_memory_fixed_block_allocator_restore(pool->block_allocator,
                                                block_index);
  iree_hal_fixed_block_pool_uncharge_reservation(pool,
                                                 pool->backing_block_size);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, -1, iree_memory_order_relaxed);
  switch (info->result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, -1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, -1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_atomic_fetch_add(&pool->wait_count, -1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid fixed-block rollback result: %u",
                  info->result);
      break;
  }
}

static iree_status_t iree_hal_fixed_block_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_validate_reservation_request(
        pool, &requests[i]));
  }

  iree_hal_fixed_block_pool_acquire_element_t
      inline_elements[IREE_HAL_FIXED_BLOCK_POOL_INLINE_TRANSACTION_CAPACITY];
  iree_hal_fixed_block_pool_acquire_element_t* elements = inline_elements;
  bool elements_allocated = false;
  iree_status_t status = iree_ok_status();
  if (request_count > IREE_ARRAYSIZE(inline_elements)) {
    status = iree_allocator_malloc_array(pool->host_allocator, request_count,
                                         sizeof(*elements), (void**)&elements);
    elements_allocated = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    memset(elements, 0, request_count * sizeof(*elements));
  }

  iree_host_size_t acquired_count = 0;
  bool did_rollback = false;
  iree_hal_pool_acquire_result_t transaction_result =
      IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  while (acquired_count < request_count && iree_status_is_ok(status)) {
    iree_hal_pool_acquire_result_t item_result = IREE_HAL_POOL_ACQUIRE_NONE;
    status = iree_hal_fixed_block_pool_acquire_one_reservation(
        base_pool, &requests[acquired_count], requester_frontier, flags,
        &elements[acquired_count].reservation, &elements[acquired_count].info,
        &item_result);
    if (!iree_status_is_ok(status)) break;
    switch (item_result) {
      case IREE_HAL_POOL_ACQUIRE_OK:
        if (transaction_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
          transaction_result = IREE_HAL_POOL_ACQUIRE_OK;
        }
        break;
      case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
        break;
      case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
        transaction_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
        break;
      case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
      case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
        transaction_result = item_result;
        break;
      case IREE_HAL_POOL_ACQUIRE_NONE:
      default:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "fixed-block pool produced unknown acquire result %u", item_result);
        break;
    }
    if (!iree_status_is_ok(status) ||
        item_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
        item_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
      break;
    }
    ++acquired_count;
  }

  if (iree_status_is_ok(status) && acquired_count != request_count) {
    did_rollback = acquired_count != 0;
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_fixed_block_pool_rollback_reservation(
          pool, &elements[i].reservation, &elements[i].info);
      memset(&elements[i], 0, sizeof(elements[i]));
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = elements[i].info;
    }
    if (acquired_count == request_count) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        out_reservations[i] = elements[i].reservation;
      }
    }
    *out_result = transaction_result;
  } else {
    did_rollback = acquired_count != 0;
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_fixed_block_pool_rollback_reservation(
          pool, &elements[i].reservation, &elements[i].info);
    }
  }
  if (did_rollback) {
    iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
  }

  if (elements_allocated) {
    iree_allocator_free(pool->host_allocator, elements);
  }
  return status;
}

static void iree_hal_fixed_block_pool_release_one_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;

  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)pool->slab.base_ptr + reservation->offset);

  const uint32_t block_index = (uint32_t)reservation->block_handle;
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    const iree_hal_asan_allocation_layout_t* asan_layout =
        &pool->asan_block_layouts[block_index];
    const iree_device_size_t backing_offset =
        (iree_device_size_t)block_index * pool->backing_block_size;
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &pool->slab, backing_offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED, asan_layout);
  }

  iree_hal_memory_fixed_block_allocator_release(pool->block_allocator,
                                                block_index, death_frontier);

  iree_hal_fixed_block_pool_uncharge_reservation(pool,
                                                 pool->backing_block_size);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
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
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_fixed_block_pool_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_fixed_block_pool_materialize_element_t* element =
      (iree_hal_fixed_block_pool_materialize_element_t*)user_data;
  iree_hal_fixed_block_pool_materialize_state_t* state = element->state;
  if (state->ownership_committed) {
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
    if (!iree_status_is_ok(status)) return status;
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
    if (!iree_status_is_ok(status)) return status;
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
    status = iree_hal_slab_provider_wrap_buffer(
        pool->slab_provider, &pool->slab,
        reservations[materialized_count].offset,
        reservations[materialized_count].byte_length,
        requests[materialized_count].params, release_callback, staged_buffer);
    if (iree_status_is_ok(status)) ++materialized_count;
  }
  if (iree_status_is_ok(status)) {
    if (state) state->ownership_committed = true;
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
  out_capabilities->memory_type = pool->slab_properties.memory_type;
  out_capabilities->supported_usage = pool->slab_properties.supported_usage;
  out_capabilities->atomic_operations = pool->slab_properties.atomic_operations;
  out_capabilities->min_allocation_size = 1;
  out_capabilities->max_allocation_size = pool->user_block_size;
}

static void iree_hal_fixed_block_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_memory_fixed_block_allocator_stats_t block_stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool->block_allocator,
                                                    &block_stats);
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  const iree_device_size_t managed_bytes =
      (iree_device_size_t)block_stats.block_count * pool->backing_block_size;
  out_stats->bytes_free = managed_bytes - out_stats->bytes_reserved;
  out_stats->bytes_committed = pool->slab.length;
  out_stats->budget_limit = pool->budget_limit;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count = 1;
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

static iree_status_t iree_hal_fixed_block_pool_trim(
    iree_hal_pool_t* base_pool) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_slab_provider_trim(pool->slab_provider,
                              IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_EXCESS);
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_fixed_block_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable = {
    .destroy = iree_hal_fixed_block_pool_destroy,
    .acquire_reservations = iree_hal_fixed_block_pool_acquire_reservations,
    .release_reservations = iree_hal_fixed_block_pool_release_reservations,
    .materialize_reservations =
        iree_hal_fixed_block_pool_materialize_reservations,
    .query_capabilities = iree_hal_fixed_block_pool_query_capabilities,
    .query_stats = iree_hal_fixed_block_pool_query_stats,
    .trim = iree_hal_fixed_block_pool_trim,
    .notification = iree_hal_fixed_block_pool_notification,
};
