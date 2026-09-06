// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(pool, method_name) \
  IREE_HAL_VTABLE_DISPATCH(pool, iree_hal_pool, method_name)

IREE_HAL_API_RETAIN_RELEASE(pool);

IREE_API_EXPORT void iree_hal_pool_initialize(
    const iree_hal_pool_vtable_t* vtable, iree_hal_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_pool);
  iree_hal_resource_initialize(vtable, &out_pool->resource);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservations(
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(request_count);
  IREE_ASSERT_ARGUMENT(requests);
  IREE_ASSERT_ARGUMENT(out_reservations);
  IREE_ASSERT_ARGUMENT(out_infos);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)request_count);
  iree_status_t status = _VTABLE_DISPATCH(pool, acquire_reservations)(
      pool, request_count, requests, requester_frontier, flags,
      out_reservations, out_infos, out_result);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_pool_release_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation_count);
  IREE_ASSERT_ARGUMENT(reservations);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)reservation_count);
  _VTABLE_DISPATCH(pool, release_reservations)(pool, reservation_count,
                                               reservations, death_frontier);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_materialize_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation_count);
  IREE_ASSERT_ARGUMENT(requests);
  IREE_ASSERT_ARGUMENT(reservations);
  IREE_ASSERT_ARGUMENT(out_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)reservation_count);
  iree_status_t status = _VTABLE_DISPATCH(pool, materialize_reservations)(
      pool, reservation_count, requests, reservations, flags, out_buffers);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  _VTABLE_DISPATCH(pool, query_capabilities)(pool, out_capabilities);
}

IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_stats);
  memset(out_stats, 0, sizeof(*out_stats));
  _VTABLE_DISPATCH(pool, query_stats)(pool, out_stats);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_trim(iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, trim)(pool);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return _VTABLE_DISPATCH(pool, notification)(pool);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = allocation_size,
  };
  iree_hal_buffer_t* buffer = NULL;

  // Convert to absolute so retries after spurious wakes use a consistent
  // cutoff.
  iree_convert_timeout_to_absolute(&timeout);
  iree_async_notification_t* notification = iree_hal_pool_notification(pool);
  iree_status_t status = iree_ok_status();
  bool retry = true;
  while (retry) {
    const uint32_t wait_token =
        iree_async_notification_begin_observe(notification);

    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t acquire_info;
    iree_hal_pool_acquire_result_t result;
    status = iree_hal_pool_acquire_reservations(
        pool, 1, &request, requester_frontier, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &reservation, &acquire_info, &result);
    if (iree_status_is_ok(status)) {
      switch (result) {
        case IREE_HAL_POOL_ACQUIRE_OK:
        case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
          // Reservation succeeded; transfer ownership to the returned buffer.
          status = iree_hal_pool_materialize_reservations(
              pool, 1, &request, &reservation,
              IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
              &buffer);
          if (!iree_status_is_ok(status)) {
            // Wrapping failed; release the reservation to avoid leaking the
            // offset back to the pool.
            iree_hal_pool_release_reservations(pool, 1, &reservation, NULL);
          }
          retry = false;
          break;
        case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
          // Synchronous allocation cannot model a hidden queue wait edge. A
          // pool used through this helper must skip non-dominated blocks and
          // return EXHAUSTED/OVER_BUDGET until an immediately-usable
          // reservation exists. Preserve the block's original frontier and
          // report a pool implementation bug, not a caller precondition
          // failure.
          iree_hal_pool_release_reservations(pool, 1, &reservation,
                                             acquire_info.wait_frontier);
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "iree_hal_pool_allocate_buffer received an "
              "IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT reservation from a pool "
              "that must only return immediately-usable reservations in the "
              "synchronous helper path");
          retry = false;
          break;
        case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
        case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
          // Wait for a release to advance the notification, then retry.
          if (!iree_async_notification_wait_for_token(notification, wait_token,
                                                      timeout)) {
            status = iree_make_status(
                IREE_STATUS_DEADLINE_EXCEEDED,
                "pool allocate_buffer timed out waiting for a free block (%s)",
                result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ? "exhausted"
                                                          : "over budget");
            retry = false;
          }
          break;
        case IREE_HAL_POOL_ACQUIRE_NONE:
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "pool returned no result for a completed allocation request");
          retry = false;
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "pool returned unknown acquire result %u",
                                    result);
          retry = false;
          break;
      }
    }
    iree_async_notification_end_observe(notification);
    if (!iree_status_is_ok(status)) {
      retry = false;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
