// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/detail.h"
#include "iree/hal/pool_wait.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(pool, method_name) \
  IREE_HAL_VTABLE_DISPATCH(pool, iree_hal_pool, method_name)

IREE_HAL_API_RETAIN_RELEASE(pool);

IREE_API_EXPORT iree_status_t iree_hal_pool_initialize(
    const iree_hal_pool_vtable_t* vtable,
    iree_hal_memory_contract_t* memory_contract,
    iree_async_notification_t* notification,
    iree_hal_pool_wait_source_list_t backing_sources,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_pool);
  iree_host_size_t source_count = notification ? 1 : 0;
  for (iree_host_size_t i = 0; i < backing_sources.count; ++i) {
    source_count +=
        backing_sources.values[i] && backing_sources.values[i] != notification;
  }
  iree_async_notification_t** sources =
      notification ? &out_pool->notification : NULL;
  if (source_count > (notification ? 1 : 0)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, source_count, sizeof(*sources), (void**)&sources));
  }
  iree_hal_resource_initialize(vtable, &out_pool->resource);
  out_pool->memory_contract = memory_contract;
  iree_hal_memory_contract_retain(memory_contract);
  out_pool->notification = notification;
  iree_async_notification_retain(notification);
  iree_host_size_t source_index = 0;
  if (notification) {
    sources[source_index++] = notification;
  }
  for (iree_host_size_t i = 0; i < backing_sources.count; ++i) {
    if (backing_sources.values[i] &&
        backing_sources.values[i] != notification) {
      sources[source_index++] = backing_sources.values[i];
      iree_async_notification_retain(backing_sources.values[i]);
    }
  }
  out_pool->wait_sources = (iree_hal_pool_wait_source_list_t){
      .count = source_count,
      .values = sources,
  };
  out_pool->host_allocator = host_allocator;
  out_pool->frontier_tracker = frontier_tracker;
  out_pool->epoch_query = iree_hal_pool_epoch_query_null();
  out_pool->asan_enabled = false;
  out_pool->allocation_interceptor =
      (iree_hal_pool_allocation_interceptor_t){0};
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_pool_deinitialize(iree_hal_pool_t* pool) {
  iree_hal_memory_contract_release(pool->memory_contract);
  for (iree_host_size_t i = 0; i < pool->wait_sources.count; ++i) {
    iree_async_notification_release(pool->wait_sources.values[i]);
  }
  if (pool->wait_sources.values &&
      pool->wait_sources.values != &pool->notification) {
    iree_allocator_free(pool->host_allocator, (void*)pool->wait_sources.values);
  }
}

IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservations(
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(request_count);
  IREE_ASSERT_ARGUMENT(requests);
  IREE_ASSERT_ARGUMENT(out_reservations);
  IREE_ASSERT_ARGUMENT(out_infos);
  IREE_ASSERT_ARGUMENT(completion.fn);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)request_count);
  iree_status_t status = _VTABLE_DISPATCH(pool, acquire_reservations)(
      pool, request_count, requests, requester_frontier, out_reservations,
      out_infos, completion, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_pool_sync_acquire_t {
  // Protects the terminal result and the callback's final notification access.
  iree_slim_mutex_t mutex;
  // Wakes the caller after accepted native preparation completes.
  iree_notification_t notification;
  // True after completion publishes status and result.
  bool complete;
  // Owned terminal failure, transferred to the synchronous caller.
  iree_status_t status;
  // Whole-batch outcome, meaningful only on success.
  iree_hal_pool_acquire_result_t result;
} iree_hal_pool_sync_acquire_t;

static void iree_hal_pool_sync_acquire_complete(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_pool_sync_acquire_t* call = user_data;
  iree_slim_mutex_lock(&call->mutex);
  call->status = status;
  call->result = result;
  call->complete = true;
  iree_notification_post(&call->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&call->mutex);
}

static bool iree_hal_pool_sync_acquire_is_complete(void* user_data) {
  iree_hal_pool_sync_acquire_t* call = user_data;
  iree_slim_mutex_lock(&call->mutex);
  const bool complete = call->complete;
  iree_slim_mutex_unlock(&call->mutex);
  return complete;
}

IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservations_sync(
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result,
    iree_hal_pool_reserve_flags_t flags) {
  iree_hal_pool_sync_acquire_t call = {0};
  iree_slim_mutex_initialize(&call.mutex);
  iree_notification_initialize(&call.notification);
  iree_status_t status = iree_hal_pool_acquire_reservations(
      pool, request_count, requests, requester_frontier, out_reservations,
      out_infos,
      (iree_hal_pool_acquire_callback_t){
          .fn = iree_hal_pool_sync_acquire_complete,
          .user_data = &call,
      },
      flags);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&call.notification,
                            iree_hal_pool_sync_acquire_is_complete, &call,
                            iree_infinite_timeout());
    status = call.status;
    if (iree_status_is_ok(status)) {
      *out_result = call.result;
    }
  }
  iree_notification_deinitialize(&call.notification);
  iree_slim_mutex_deinitialize(&call.mutex);
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

IREE_API_EXPORT bool iree_hal_pool_requires_asan_advice(
    const iree_hal_pool_t* pool) {
  return pool->asan_enabled;
}

IREE_API_EXPORT void iree_hal_pool_advise_asan_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  if (pool->asan_enabled) {
    _VTABLE_DISPATCH(pool, advise_asan_reservations)(pool, reservation_count,
                                                     reservations, flags);
  }
}

IREE_API_EXPORT bool iree_hal_pool_query_reservation_views(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* out_views) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation_count);
  IREE_ASSERT_ARGUMENT(reservations);
  IREE_ASSERT_ARGUMENT(out_views);
  const iree_hal_pool_vtable_t* vtable =
      (const iree_hal_pool_vtable_t*)pool->resource.vtable;
  if (!vtable->query_reservation_views) {
    return false;
  }
  vtable->query_reservation_views(pool, reservation_count, reservations,
                                  out_views);
  return true;
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

IREE_API_EXPORT iree_status_t iree_hal_pool_import_buffer(
    iree_hal_pool_t* pool, iree_hal_external_storage_t* inout_storage,
    iree_hal_buffer_range_t logical_range, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(inout_storage);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (inout_storage->storage_class.type ==
          IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE ||
      inout_storage->byte_length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "external storage must be nonempty");
  }

  iree_hal_buffer_range_t resolved_range = {0};
  iree_status_t status = iree_hal_buffer_calculate_range(
      0, inout_storage->byte_length, logical_range.byte_offset,
      logical_range.byte_length, &resolved_range.byte_offset,
      &resolved_range.byte_length);
  if (iree_status_is_ok(status)) {
    iree_device_size_t maintenance_offset = 0;
    iree_device_size_t maintenance_length = 0;
    status = iree_hal_buffer_calculate_range(
        0, inout_storage->byte_length,
        inout_storage->maintenance_range.byte_offset,
        inout_storage->maintenance_range.byte_length, &maintenance_offset,
        &maintenance_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_vtable_t* vtable =
        (const iree_hal_pool_vtable_t*)pool->resource.vtable;
    if (!vtable->import_buffer) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "pool does not admit caller-selected external backing");
    } else {
      status = vtable->import_buffer(pool, inout_storage, resolved_range,
                                     out_buffer);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT bool iree_hal_pool_query_import_requirements(
    const iree_hal_pool_t* pool,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_requirements);
  memset(out_requirements, 0, sizeof(*out_requirements));
  if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE) {
    return false;
  }
  const iree_hal_pool_vtable_t* vtable =
      (const iree_hal_pool_vtable_t*)pool->resource.vtable;
  return vtable->query_import_requirements &&
         vtable->query_import_requirements(pool, storage_class,
                                           out_requirements);
}

IREE_API_EXPORT iree_allocator_t
iree_hal_pool_host_allocator(const iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->host_allocator;
}

IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  _VTABLE_DISPATCH(pool, query_capabilities)(pool, out_capabilities);
  if (pool->memory_contract) {
    out_capabilities->placement = pool->memory_contract->placement;
  }
}

IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_stats);
  memset(out_stats, 0, sizeof(*out_stats));
  _VTABLE_DISPATCH(pool, query_stats)(pool, out_stats);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_validate_asan_options(
    const iree_hal_pool_t* pool, const iree_hal_asan_pool_options_t* options) {
  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(options));
  if (!iree_hal_asan_pool_options_is_enabled(options)) {
    return iree_ok_status();
  }
  const iree_hal_pool_vtable_t* vtable =
      (const iree_hal_pool_vtable_t*)pool->resource.vtable;
  if (!vtable->validate_asan) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pool cannot supply native ASAN range advice");
  }
  return vtable->validate_asan(pool, options);
}

IREE_API_EXPORT void iree_hal_pool_trim(iree_hal_pool_t* pool,
                                        iree_hal_pool_trim_flags_t flags,
                                        iree_device_size_t min_bytes_to_keep) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE({
    char message[128];
    int length =
        iree_snprintf(message, sizeof(message),
                      "HAL pool trim pool=%p flags=0x%08x keep=%" PRIu64,
                      (void*)pool, flags, (uint64_t)min_bytes_to_keep);
    IREE_TRACE_MESSAGE_DYNAMIC(INFO, message, (iree_host_size_t)length);
  });
  _VTABLE_DISPATCH(pool, trim)(pool, flags, min_bytes_to_keep);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->notification;
}

// Stack-owned state shared with an asynchronous completion callback.
typedef struct iree_hal_pool_sync_wait_t {
  // Wakes the allocating thread when its asynchronous wait resolves or fails.
  iree_notification_t notification;
  // Published before notification so the waiter can observe the result.
  iree_atomic_int32_t resolved;
  // Published after the callback's final notification access.
  iree_atomic_int32_t completed;
  // Callback-owned status transferred to the allocating thread on completion.
  iree_status_t status;
} iree_hal_pool_sync_wait_t;

static bool iree_hal_pool_sync_wait_is_resolved(void* user_data) {
  iree_hal_pool_sync_wait_t* wait = user_data;
  return iree_atomic_load(&wait->resolved, iree_memory_order_acquire) != 0;
}

static void iree_hal_pool_sync_wait_resolve(void* user_data,
                                            iree_status_t status) {
  iree_hal_pool_sync_wait_t* wait = user_data;
  wait->status = status;
  iree_atomic_store(&wait->resolved, 1, iree_memory_order_release);
  iree_notification_post(&wait->notification, IREE_ALL_WAITERS);
  // This is the callback's final access to caller-owned storage.
  iree_atomic_store(&wait->completed, 1, iree_memory_order_release);
}

// Waits for a reserved range using the allocation's already-normalized
// deadline. Immediate attempts also check here: the optional native completion
// probe may be absent, or completion may have arrived after reservation.
static iree_status_t iree_hal_pool_wait_for_frontier(
    iree_hal_pool_t* pool, const iree_async_frontier_t* frontier,
    iree_timeout_t timeout) {
  if (iree_timeout_is_immediate(timeout)) {
    bool satisfied = false;
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_query(
        pool->frontier_tracker, frontier, &satisfied));
    return satisfied ? iree_ok_status()
                     : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  iree_hal_pool_sync_wait_t wait;
  iree_notification_initialize(&wait.notification);
  iree_atomic_store(&wait.resolved, 0, iree_memory_order_relaxed);
  iree_atomic_store(&wait.completed, 0, iree_memory_order_relaxed);
  wait.status = iree_ok_status();
  iree_async_frontier_waiter_t waiter;
  iree_status_t status = iree_async_frontier_tracker_wait(
      pool->frontier_tracker, frontier, iree_hal_pool_sync_wait_resolve, &wait,
      &waiter);
  if (iree_status_is_ok(status)) {
    const bool resolved = iree_notification_await(
        &wait.notification, iree_hal_pool_sync_wait_is_resolved, &wait,
        timeout);
    if (!resolved && iree_async_frontier_tracker_cancel_wait(
                         pool->frontier_tracker, &waiter)) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    } else {
      // Dispatch won. Join the callback even if the deadline elapsed: observing
      // resolved does not prove notification_post has stopped touching storage.
      // This joins only the dispatched callback, never unfinished device work.
      while (!iree_atomic_load(&wait.completed, iree_memory_order_acquire)) {
        iree_processor_yield();
      }
      status = wait.status;
    }
  }
  iree_notification_deinitialize(&wait.notification);
  return status;
}

// Joins one asynchronous capacity round. The helper owns timeout cancellation
// and native retirement, so this thread waits only for its terminal handoff.
static iree_status_t iree_hal_pool_wait_for_capacity(
    iree_hal_pool_wait_t* capacity_wait, iree_timeout_t timeout) {
  iree_hal_pool_sync_wait_t wait;
  iree_notification_initialize(&wait.notification);
  iree_atomic_store(&wait.resolved, 0, iree_memory_order_relaxed);
  iree_atomic_store(&wait.completed, 0, iree_memory_order_relaxed);
  wait.status = iree_ok_status();
  iree_hal_pool_wait_commit(capacity_wait, timeout,
                            (iree_hal_pool_wait_callback_t){
                                .fn = iree_hal_pool_sync_wait_resolve,
                                .user_data = &wait,
                            });
  iree_notification_await(&wait.notification,
                          iree_hal_pool_sync_wait_is_resolved, &wait,
                          iree_infinite_timeout());
  while (!iree_atomic_load(&wait.completed, iree_memory_order_acquire)) {
    iree_processor_yield();
  }
  iree_notification_deinitialize(&wait.notification);
  return wait.status;
}

static iree_status_t iree_hal_pool_allocate_buffer_direct(
    iree_hal_pool_t* pool, iree_hal_pool_reservation_request_t request,
    iree_timeout_t timeout, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_t* buffer = NULL;

  // Capacity retries and exact-range completion share one deadline.
  iree_convert_timeout_to_absolute(&timeout);
  iree_hal_pool_wait_t* capacity_wait = NULL;
  iree_status_t status = iree_ok_status();
  bool retry = true;
  while (retry) {
    if (capacity_wait) {
      iree_hal_pool_wait_prepare(capacity_wait);
    }

    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t acquire_info;
    iree_hal_pool_acquire_result_t result;
    status = iree_hal_pool_acquire_reservations_sync(
        pool, 1, &request,
        /*requester_frontier=*/NULL, &reservation, &acquire_info, &result,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER);
    if (capacity_wait && (!iree_status_is_ok(status) ||
                          (result != IREE_HAL_POOL_ACQUIRE_EXHAUSTED &&
                           result != IREE_HAL_POOL_ACQUIRE_OVER_BUDGET))) {
      iree_hal_pool_wait_abort(capacity_wait);
    }
    if (iree_status_is_ok(status)) {
      switch (result) {
        case IREE_HAL_POOL_ACQUIRE_OK:
        case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
        case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT: {
          const iree_async_frontier_t* rollback_frontier =
              acquire_info.reuse_frontier;
          if (result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
            status = iree_hal_pool_wait_for_frontier(
                pool, acquire_info.reuse_frontier, timeout);
            if (iree_status_is_ok(status)) {
              rollback_frontier = NULL;
            }
          }
          if (iree_status_is_ok(status)) {
            // Transfer the now-usable reservation to the returned buffer.
            status = iree_hal_pool_materialize_reservations(
                pool, 1, &request, &reservation,
                IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
                &buffer);
            if (iree_status_is_ok(status)) {
              iree_hal_pool_advise_asan_reservations(
                  pool, 1, &reservation,
                  IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
            }
          }
          if (!iree_status_is_ok(status)) {
            // A failed wait must preserve the dependency on the previous user.
            iree_hal_pool_release_reservations(pool, 1, &reservation,
                                               rollback_frontier);
          }
          retry = false;
          break;
        }
        case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
        case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
          if (iree_timeout_is_immediate(timeout)) {
            if (capacity_wait) {
              iree_hal_pool_wait_abort(capacity_wait);
            }
            status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
          } else if (!capacity_wait) {
            // Only a failed fast attempt allocates cold wait state. Retry with
            // every source observed before submitting any asynchronous wait.
            status = iree_hal_pool_wait_allocate(pool, pool->host_allocator,
                                                 &capacity_wait);
          } else {
            status = iree_hal_pool_wait_for_capacity(capacity_wait, timeout);
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
    if (!iree_status_is_ok(status)) {
      retry = false;
    }
  }
  iree_hal_pool_wait_free(capacity_wait);

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  }

  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_pool_reservation_request_t request,
    iree_timeout_t timeout, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      pool->allocation_interceptor.fn
          ? pool->allocation_interceptor.fn(
                pool->allocation_interceptor.user_data, pool, request, timeout,
                iree_hal_pool_allocate_buffer_direct, out_buffer)
          : iree_hal_pool_allocate_buffer_direct(pool, request, timeout,
                                                 out_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
