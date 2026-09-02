#include <string.h>

#include "hrx_internal.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/alignment.h"
#include "iree/hal/api.h"

enum { HRX_IREE_EXACT_POOL_INLINE_TRANSACTION_CAPACITY = 8 };

typedef struct hrx_iree_exact_pool_t {
  iree_hal_pool_t base;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* allocator;
  iree_hal_buffer_params_t params;
  iree_async_notification_t* notification;
} hrx_iree_exact_pool_t;

static const iree_hal_pool_vtable_t hrx_iree_exact_pool_vtable;

static hrx_iree_exact_pool_t* hrx_iree_exact_pool_cast(iree_hal_pool_t* base) {
  return (hrx_iree_exact_pool_t*)base;
}

static const hrx_iree_exact_pool_t* hrx_iree_exact_pool_const_cast(
    const iree_hal_pool_t* base) {
  return (const hrx_iree_exact_pool_t*)base;
}

static bool hrx_iree_buffer_params_are_compatible(
    iree_hal_buffer_params_t pool_params,
    iree_hal_buffer_params_t request_params) {
  iree_hal_buffer_params_canonicalize(&pool_params);
  iree_hal_buffer_params_canonicalize(&request_params);
  return pool_params.type == request_params.type &&
         pool_params.access == request_params.access &&
         pool_params.usage == request_params.usage &&
         pool_params.queue_affinity == request_params.queue_affinity;
}

iree_status_t hrx_iree_exact_pool_create(iree_hal_allocator_t* allocator,
                                         iree_hal_buffer_params_t params,
                                         iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_pool);

  hrx_shared_state_t* shared = hrx_get_shared_state();
  if (!shared || !shared->proactor_pool) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "shared proactor pool must be initialized before creating hrx pools");
  }

  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(shared->proactor_pool,
                                                    /*index=*/0, &proactor),
                       "acquiring proactor for hrx allocation pool");

  hrx_iree_exact_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(shared->host_allocator,
                                             sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_pool_initialize(&hrx_iree_exact_pool_vtable, &pool->base);
  pool->host_allocator = shared->host_allocator;
  pool->allocator = allocator;
  iree_hal_buffer_params_canonicalize(&params);
  pool->params = params;
  iree_hal_allocator_retain(pool->allocator);

  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &pool->notification);
  if (!iree_status_is_ok(status)) {
    iree_hal_allocator_release(pool->allocator);
    iree_allocator_free(pool->host_allocator, pool);
    return status;
  }

  *out_pool = (iree_hal_pool_t*)pool;
  return iree_ok_status();
}

static void hrx_iree_exact_pool_destroy(iree_hal_pool_t* base_pool) {
  hrx_iree_exact_pool_t* pool = hrx_iree_exact_pool_cast(base_pool);
  iree_async_notification_release(pool->notification);
  iree_hal_allocator_release(pool->allocator);
  iree_allocator_free(pool->host_allocator, pool);
}

static iree_status_t hrx_iree_exact_pool_validate_request(
    const hrx_iree_exact_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request) {
  if (request->allocation_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool reservations must be non-empty");
  }
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;
  if (!iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment (%" PRIdsz
                            ") must be a power of two",
                            alignment);
  }
  const iree_device_size_t pool_alignment =
      pool->params.min_alignment ? pool->params.min_alignment : 1;
  if (alignment > pool_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds exact pool alignment %" PRIdsz,
                            alignment, pool_alignment);
  }
  if (!hrx_iree_buffer_params_are_compatible(pool->params, request->params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hrx exact pools require allocation params to match pool creation");
  }
  return iree_ok_status();
}

static iree_status_t hrx_iree_exact_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  hrx_iree_exact_pool_t* pool = hrx_iree_exact_pool_cast(base_pool);
  (void)requester_frontier;
  (void)flags;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(
        hrx_iree_exact_pool_validate_request(pool, &requests[i]));
  }

  iree_hal_buffer_t*
      inline_buffers[HRX_IREE_EXACT_POOL_INLINE_TRANSACTION_CAPACITY] = {0};
  iree_hal_buffer_t** buffers = inline_buffers;
  bool buffers_allocated = false;
  iree_status_t status = iree_ok_status();
  if (request_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(pool->host_allocator, request_count,
                                         sizeof(*buffers), (void**)&buffers);
    buffers_allocated = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    memset(buffers, 0, request_count * sizeof(*buffers));
  }

  iree_host_size_t acquired_count = 0;
  while (acquired_count < request_count && iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        pool->allocator, pool->params, requests[acquired_count].allocation_size,
        &buffers[acquired_count]);
    if (iree_status_is_ok(status)) ++acquired_count;
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_pool_reservation_t reservation = {0};
      reservation.byte_length = iree_hal_buffer_byte_length(buffers[i]);
      reservation.block_handle = (uint64_t)(uintptr_t)buffers[i];
      out_reservations[i] = reservation;
      memset(&out_infos[i], 0, sizeof(out_infos[i]));
      out_infos[i].result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    }
    *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  } else {
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  if (buffers_allocated) {
    iree_allocator_free(pool->host_allocator, buffers);
  }
  return status;
}

static void hrx_iree_exact_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  hrx_iree_exact_pool_t* pool = hrx_iree_exact_pool_cast(base_pool);
  (void)death_frontier;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_buffer_t* buffer =
        (iree_hal_buffer_t*)(uintptr_t)reservations[i].block_handle;
    iree_hal_buffer_release(buffer);
  }
  iree_async_notification_signal(pool->notification, /*wake_count=*/INT32_MAX);
}

static iree_status_t hrx_iree_exact_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  const hrx_iree_exact_pool_t* pool = hrx_iree_exact_pool_const_cast(base_pool);
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    IREE_RETURN_IF_ERROR(
        hrx_iree_exact_pool_validate_request(pool, &requests[i]));
    if (reservations[i].byte_length < requests[i].allocation_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has %" PRIdsz
          " bytes but its allocation request requires %" PRIdsz,
          i, reservations[i].byte_length, requests[i].allocation_size);
    }
    if (!reservations[i].block_handle) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "reservation %" PRIhsz " has no backing buffer",
                              i);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_buffer_t* buffer =
        (iree_hal_buffer_t*)(uintptr_t)reservations[i].block_handle;
    if (!transfer_ownership) iree_hal_buffer_retain(buffer);
    out_buffers[i] = buffer;
  }
  return iree_ok_status();
}

static void hrx_iree_exact_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const hrx_iree_exact_pool_t* pool = hrx_iree_exact_pool_const_cast(base_pool);
  out_capabilities->memory_type = pool->params.type;
  out_capabilities->supported_usage = pool->params.usage;
  out_capabilities->min_allocation_size = 0;
  out_capabilities->max_allocation_size = 0;
}

static void hrx_iree_exact_pool_query_stats(const iree_hal_pool_t* base_pool,
                                            iree_hal_pool_stats_t* out_stats) {
  (void)base_pool;
  memset(out_stats, 0, sizeof(*out_stats));
}

static iree_status_t hrx_iree_exact_pool_trim(iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return iree_ok_status();
}

static iree_async_notification_t* hrx_iree_exact_pool_notification(
    iree_hal_pool_t* base_pool) {
  return hrx_iree_exact_pool_cast(base_pool)->notification;
}

static const iree_hal_pool_vtable_t hrx_iree_exact_pool_vtable = {
    .destroy = hrx_iree_exact_pool_destroy,
    .acquire_reservations = hrx_iree_exact_pool_acquire_reservations,
    .release_reservations = hrx_iree_exact_pool_release_reservations,
    .materialize_reservations = hrx_iree_exact_pool_materialize_reservations,
    .query_capabilities = hrx_iree_exact_pool_query_capabilities,
    .query_stats = hrx_iree_exact_pool_query_stats,
    .trim = hrx_iree_exact_pool_trim,
    .notification = hrx_iree_exact_pool_notification,
};
