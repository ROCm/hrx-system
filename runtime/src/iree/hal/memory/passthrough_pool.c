// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/hal/memory/tracing.h"

enum { IREE_HAL_PASSTHROUGH_POOL_INLINE_TRANSACTION_CAPACITY = 8 };

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_passthrough_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Provider used to acquire one slab per reservation.
  iree_hal_slab_provider_t* slab_provider;

  // Notification signaled when a reservation release may unblock waiters.
  iree_async_notification_t* notification;

  // Host allocator used for pool metadata and reservation state.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Immutable memory properties provided by |slab_provider|.
  iree_hal_slab_provider_properties_t slab_properties;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // Approximate live reservation bytes for lock-free stats queries.
  iree_atomic_int64_t bytes_reserved;

  // Approximate live reservation count for lock-free stats queries.
  iree_atomic_int32_t reservation_count;

  // Approximate live slab count for lock-free stats queries.
  iree_atomic_int32_t slab_count;

  // Total reservations committed by successful transactions.
  iree_atomic_int64_t reserve_count;

  // Total reservations returned by release transactions.
  iree_atomic_int64_t release_count;
} iree_hal_passthrough_pool_t;

// Per-reservation slab state owned by the reservation until release.
typedef struct iree_hal_passthrough_pool_reservation_state_t {
  // Borrowed from the source pool. Pool owners must keep the pool alive until
  // all reservations and buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  // Slab acquired from the pool's provider for this reservation.
  iree_hal_slab_t slab;

  // Backing bytes charged to this reservation.
  iree_device_size_t charged_length;

  // ASAN backing layout for this reservation.
  iree_hal_asan_allocation_layout_t asan_layout;

  // One reference for the live reservation token plus one reference for each
  // materialized buffer view. The slab is released when the last reference
  // drops, which allows the reservation transaction to release before borrowed
  // backing views are decommitted.
  iree_atomic_int32_t reference_count;

  // Set exactly once when the reservation token is released. Owning
  // materialized buffers consume that reservation release in their destroy
  // callback; borrowed views only drop their own reference.
  iree_atomic_int32_t reservation_released;
} iree_hal_passthrough_pool_reservation_state_t;

typedef struct iree_hal_passthrough_pool_materialize_state_t
    iree_hal_passthrough_pool_materialize_state_t;

// Per-buffer element in an owning materialization transaction.
typedef struct iree_hal_passthrough_pool_materialize_element_t {
  // Shared transaction state controlling the ownership commit.
  iree_hal_passthrough_pool_materialize_state_t* state;

  // Reservation state released when the committed buffer is destroyed.
  iree_hal_passthrough_pool_reservation_state_t* reservation_state;

  // Materialized buffer staged until the complete transaction succeeds.
  iree_hal_buffer_t* buffer;
} iree_hal_passthrough_pool_materialize_element_t;

// Shared state for an owning materialization transaction.
struct iree_hal_passthrough_pool_materialize_state_t {
  // Host allocator used for this state object.
  iree_allocator_t host_allocator;

  // Number of materialized buffers still referencing this transaction.
  iree_atomic_int32_t reference_count;

  // True after every buffer was materialized and reservation ownership moved.
  bool ownership_committed;

  // Per-buffer transaction elements.
  iree_hal_passthrough_pool_materialize_element_t elements[];
};

// Staged result for one reservation acquisition. Transactions use staging so
// public output arrays remain untouched unless the operation succeeds.
typedef struct iree_hal_passthrough_pool_acquire_element_t {
  // Reservation produced for the request.
  iree_hal_pool_reservation_t reservation;

  // Acquisition metadata produced for the request.
  iree_hal_pool_acquire_info_t info;
} iree_hal_passthrough_pool_acquire_element_t;

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable;
static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_PASSTHROUGH_POOL_TRACE_ID =
    "iree-hal-passthrough-pool";

//===----------------------------------------------------------------------===//
// Reservation state helpers
//===----------------------------------------------------------------------===//

static void iree_hal_passthrough_pool_reservation_state_release_reference(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  iree_hal_passthrough_pool_t* pool =
      (iree_hal_passthrough_pool_t*)reservation_state->pool;
  const int32_t previous_count = iree_atomic_fetch_sub(
      &reservation_state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count != 1) return;

  iree_hal_slab_provider_release_slab(pool->slab_provider,
                                      &reservation_state->slab);
  iree_atomic_fetch_add(&pool->slab_count, -1, iree_memory_order_relaxed);
  iree_allocator_free(pool->host_allocator, reservation_state);
}

static void iree_hal_passthrough_pool_reservation_state_release_reservation(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  iree_hal_passthrough_pool_t* pool =
      (iree_hal_passthrough_pool_t*)reservation_state->pool;
  const int32_t already_released = iree_atomic_exchange(
      &reservation_state->reservation_released, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(already_released, 0);

  const iree_device_size_t user_offset =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
          ? reservation_state->asan_layout.user_offset
          : 0;
  iree_hal_memory_trace_free(&pool->trace,
                             reservation_state->slab.base_ptr + user_offset);
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &reservation_state->slab,
        /*backing_offset=*/0, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED,
        &reservation_state->asan_layout);
  }

  iree_atomic_fetch_add(&pool->bytes_reserved,
                        -(int64_t)reservation_state->charged_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);

  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_borrowed_view_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)user_data;
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_owned_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_materialize_element_t* element =
      (iree_hal_passthrough_pool_materialize_element_t*)user_data;
  iree_hal_passthrough_pool_materialize_state_t* state = element->state;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      element->reservation_state;
  if (state->ownership_committed) {
    iree_hal_passthrough_pool_reservation_state_release_reservation(
        reservation_state);
    iree_hal_passthrough_pool_t* pool =
        (iree_hal_passthrough_pool_t*)reservation_state->pool;
    iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
  }
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
  const int32_t previous_count = iree_atomic_fetch_sub(
      &state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count == 1) {
    iree_allocator_free(state->host_allocator, state);
  }
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_passthrough_pool_create(
    iree_hal_passthrough_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_slab_provider_validate_asan_options(slab_provider,
                                                       &options.asan));

  iree_hal_passthrough_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_pool_initialize(&iree_hal_passthrough_pool_vtable, &pool->base);
  pool->host_allocator = host_allocator;
  pool->asan_options = options.asan;

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;

  iree_hal_slab_provider_query_properties(slab_provider,
                                          &pool->slab_properties);

  iree_status_t status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_PASSTHROUGH_POOL_TRACE_ID, host_allocator,
      &pool->trace);
  if (iree_status_is_ok(status)) {
    *out_pool = (iree_hal_pool_t*)pool;
  } else {
    iree_hal_passthrough_pool_destroy((iree_hal_pool_t*)pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_validate_reservation_request(
    const iree_hal_passthrough_pool_t* pool,
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
  if (alignment > IREE_HAL_HEAP_BUFFER_ALIGNMENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds pass-through pool alignment %" PRIdsz,
                            alignment,
                            (iree_device_size_t)IREE_HAL_HEAP_BUFFER_ALIGNMENT);
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_asan_allocation_layout_t asan_layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &pool->asan_options, size, alignment, &asan_layout));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_passthrough_pool_acquire_one_reservation(
    iree_hal_pool_t* base_pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  (void)requester_frontier;
  (void)flags;
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;

  iree_hal_asan_allocation_layout_t asan_layout = {0};
  iree_device_size_t backing_length = size;
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &pool->asan_options, size, alignment, &asan_layout));
    backing_length = asan_layout.backing_length;
  }

  iree_hal_slab_t slab;
  IREE_RETURN_IF_ERROR(iree_hal_slab_provider_acquire_slab(
      pool->slab_provider, backing_length, &slab));
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options) &&
      slab.length != asan_layout.backing_length) {
    iree_status_t status =
        iree_hal_asan_extend_allocation_layout(slab.length, &asan_layout);
    if (!iree_status_is_ok(status)) {
      iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
      return status;
    }
  }

  iree_hal_passthrough_pool_reservation_state_t* reservation_state = NULL;
  iree_status_t status =
      iree_allocator_malloc(pool->host_allocator, sizeof(*reservation_state),
                            (void**)&reservation_state);
  if (!iree_status_is_ok(status)) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
    return status;
  }
  reservation_state->pool = base_pool;
  reservation_state->slab = slab;
  reservation_state->charged_length = slab.length;
  reservation_state->asan_layout = asan_layout;
  iree_atomic_store(&reservation_state->reference_count, 1,
                    iree_memory_order_relaxed);
  iree_atomic_store(&reservation_state->reservation_released, 0,
                    iree_memory_order_relaxed);

  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
          ? asan_layout.user_offset
          : 0;
  out_reservation->byte_length = size;
  out_reservation->block_handle = (uint64_t)(uintptr_t)reservation_state;

  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &reservation_state->slab,
        /*backing_offset=*/0, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED,
        &reservation_state->asan_layout);
  }

  iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)slab.length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);

  iree_hal_memory_trace_alloc(
      &pool->trace, reservation_state->slab.base_ptr + out_reservation->offset,
      out_reservation->byte_length);

  memset(out_info, 0, sizeof(*out_info));
  out_info->result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  return iree_ok_status();
}

// Rolls back a reservation acquired by the current transaction before it was
// made visible to the caller.
static void iree_hal_passthrough_pool_rollback_reservation(
    iree_hal_passthrough_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
          reservation->block_handle;
  const iree_device_size_t user_offset =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
          ? reservation_state->asan_layout.user_offset
          : 0;
  iree_hal_memory_trace_free(&pool->trace,
                             reservation_state->slab.base_ptr + user_offset);
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &reservation_state->slab,
        /*backing_offset=*/0, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED,
        &reservation_state->asan_layout);
  }
  iree_atomic_fetch_add(&pool->bytes_reserved,
                        -(int64_t)reservation_state->charged_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, -1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, -1, iree_memory_order_relaxed);
  iree_hal_slab_provider_release_slab(pool->slab_provider,
                                      &reservation_state->slab);
  iree_allocator_free(pool->host_allocator, reservation_state);
}

static iree_status_t iree_hal_passthrough_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_validate_reservation_request(
        pool, &requests[i]));
  }

  iree_hal_passthrough_pool_acquire_element_t
      inline_elements[IREE_HAL_PASSTHROUGH_POOL_INLINE_TRANSACTION_CAPACITY];
  iree_hal_passthrough_pool_acquire_element_t* elements = inline_elements;
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
  while (acquired_count < request_count && iree_status_is_ok(status)) {
    iree_hal_pool_acquire_result_t item_result = IREE_HAL_POOL_ACQUIRE_NONE;
    status = iree_hal_passthrough_pool_acquire_one_reservation(
        base_pool, &requests[acquired_count], requester_frontier, flags,
        &elements[acquired_count].reservation, &elements[acquired_count].info,
        &item_result);
    if (iree_status_is_ok(status)) ++acquired_count;
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_passthrough_pool_rollback_reservation(pool,
                                                     &elements[i].reservation);
    }
    if (acquired_count != 0) {
      iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
    }
  } else {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_reservations[i] = elements[i].reservation;
      out_infos[i] = elements[i].info;
    }
    *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  if (elements_allocated) {
    iree_allocator_free(pool->host_allocator, elements);
  }
  return status;
}

static void iree_hal_passthrough_pool_release_one_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;
  (void)death_frontier;

  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
          reservation->block_handle;
  iree_hal_passthrough_pool_reservation_state_release_reservation(
      reservation_state);
}

static void iree_hal_passthrough_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_passthrough_pool_release_one_reservation(
        base_pool, &reservations[i], death_frontier);
  }
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;

  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    if (reservations[i].byte_length < requests[i].allocation_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has %" PRIdsz
          " bytes but its allocation request requires %" PRIdsz,
          i, reservations[i].byte_length, requests[i].allocation_size);
    }
    if (!reservations[i].block_handle) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has no pass-through state", i);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_hal_passthrough_pool_materialize_state_t* state = NULL;
  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_PASSTHROUGH_POOL_INLINE_TRANSACTION_CAPACITY] = {
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
            sizeof(iree_hal_passthrough_pool_materialize_element_t),
            sizeof(*state), &state_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization state size overflow");
    }
    status =
        iree_allocator_malloc(pool->host_allocator, state_size, (void**)&state);
    if (!iree_status_is_ok(status)) return status;
    memset(state, 0, state_size);
    state->host_allocator = pool->host_allocator;
    iree_atomic_store(&state->reference_count, (int32_t)reservation_count,
                      iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      state->elements[i].state = state;
      state->elements[i].reservation_state =
          (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
              reservations[i]
                  .block_handle;
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
    iree_hal_passthrough_pool_reservation_state_t* reservation_state =
        (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    iree_atomic_fetch_add(&reservation_state->reference_count, 1,
                          iree_memory_order_acq_rel);
    iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_passthrough_pool_borrowed_view_release,
        .user_data = reservation_state,
    };
    iree_hal_buffer_t** staged_buffer = &staged_buffers[materialized_count];
    if (state) {
      release_callback.fn = iree_hal_passthrough_pool_owned_buffer_release;
      release_callback.user_data = &state->elements[materialized_count];
      staged_buffer = &state->elements[materialized_count].buffer;
    }
    status = iree_hal_slab_provider_wrap_buffer(
        pool->slab_provider, &reservation_state->slab,
        reservations[materialized_count].offset,
        reservations[materialized_count].byte_length,
        requests[materialized_count].params, release_callback, staged_buffer);
    if (iree_status_is_ok(status)) {
      ++materialized_count;
    } else {
      iree_hal_passthrough_pool_reservation_state_release_reference(
          reservation_state);
    }
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

static void iree_hal_passthrough_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_passthrough_pool_t* pool =
      (const iree_hal_passthrough_pool_t*)base_pool;
  out_capabilities->memory_type = pool->slab_properties.memory_type;
  out_capabilities->supported_usage = pool->slab_properties.supported_usage;
  out_capabilities->queue_family_affinity =
      pool->slab_properties.queue_family_affinity;
  out_capabilities->atomic_operations = pool->slab_properties.atomic_operations;
  out_capabilities->min_allocation_size = 0;
  out_capabilities->max_allocation_size = 0;
}

static void iree_hal_passthrough_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_passthrough_pool_t* pool =
      (const iree_hal_passthrough_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_free = 0;
  out_stats->bytes_committed = out_stats->bytes_reserved;
  out_stats->budget_limit = 0;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count =
      (uint32_t)iree_atomic_load(&pool->slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count = 0;
  out_stats->reuse_miss_count = 0;
  out_stats->fresh_count = out_stats->reserve_count;
  out_stats->exhausted_count = 0;
  out_stats->over_budget_count = 0;
  out_stats->wait_count = 0;
}

static iree_status_t iree_hal_passthrough_pool_trim(
    iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_passthrough_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable = {
    .destroy = iree_hal_passthrough_pool_destroy,
    .acquire_reservations = iree_hal_passthrough_pool_acquire_reservations,
    .release_reservations = iree_hal_passthrough_pool_release_reservations,
    .materialize_reservations =
        iree_hal_passthrough_pool_materialize_reservations,
    .query_capabilities = iree_hal_passthrough_pool_query_capabilities,
    .query_stats = iree_hal_passthrough_pool_query_stats,
    .trim = iree_hal_passthrough_pool_trim,
    .notification = iree_hal_passthrough_pool_notification,
};
