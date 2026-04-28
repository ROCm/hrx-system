#include "hrx_internal.h"

#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/async/notification.h"
#include "iree/hal/detail.h"
#include "iree/hal/pool.h"
#include "iree/hal/resource.h"

typedef struct hrx_iree_exact_pool_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *allocator;
  iree_hal_buffer_params_t params;
  iree_async_notification_t *notification;
} hrx_iree_exact_pool_t;

static const iree_hal_pool_vtable_t hrx_iree_exact_pool_vtable;

static hrx_iree_exact_pool_t *hrx_iree_exact_pool_cast(iree_hal_pool_t *base) {
  return (hrx_iree_exact_pool_t *)base;
}

static const hrx_iree_exact_pool_t *
hrx_iree_exact_pool_const_cast(const iree_hal_pool_t *base) {
  return (const hrx_iree_exact_pool_t *)base;
}

static bool hrx_iree_buffer_params_match(iree_hal_buffer_params_t lhs,
                                         iree_hal_buffer_params_t rhs) {
  return lhs.type == rhs.type && lhs.access == rhs.access &&
         lhs.usage == rhs.usage &&
         lhs.queue_affinity == rhs.queue_affinity &&
         lhs.min_alignment == rhs.min_alignment;
}

iree_status_t hrx_iree_exact_pool_create(iree_hal_allocator_t *allocator,
                                         iree_hal_buffer_params_t params,
                                         iree_hal_pool_t **out_pool) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;

  hrx_shared_state_t *shared = hrx_get_shared_state();
  if (!shared || !shared->proactor_pool) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "shared proactor pool must be initialized before creating hrx pools");
  }

  iree_async_proactor_t *proactor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_pool_get(shared->proactor_pool, /*index=*/0,
                                   &proactor),
      "acquiring proactor for hrx allocation pool");

  hrx_iree_exact_pool_t *pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(shared->host_allocator, sizeof(*pool),
                            (void **)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_resource_initialize(&hrx_iree_exact_pool_vtable, &pool->resource);
  pool->host_allocator = shared->host_allocator;
  pool->allocator = allocator;
  pool->params = params;
  iree_hal_allocator_retain(pool->allocator);

  iree_status_t status =
      iree_async_notification_create(proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
                                     &pool->notification);
  if (!iree_status_is_ok(status)) {
    iree_hal_allocator_release(pool->allocator);
    iree_allocator_free(pool->host_allocator, pool);
    return status;
  }

  *out_pool = (iree_hal_pool_t *)pool;
  return iree_ok_status();
}

static void hrx_iree_exact_pool_destroy(iree_hal_pool_t *base_pool) {
  hrx_iree_exact_pool_t *pool = hrx_iree_exact_pool_cast(base_pool);
  iree_async_notification_release(pool->notification);
  iree_hal_allocator_release(pool->allocator);
  iree_allocator_free(pool->host_allocator, pool);
}

static iree_status_t hrx_iree_exact_pool_acquire_reservation(
    iree_hal_pool_t *base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t *requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t *out_reservation,
    iree_hal_pool_acquire_info_t *out_info,
    iree_hal_pool_acquire_result_t *out_result) {
  hrx_iree_exact_pool_t *pool = hrx_iree_exact_pool_cast(base_pool);
  (void)alignment;
  (void)requester_frontier;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_reservation);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_ASSERT_ARGUMENT(out_result);

  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool reservations must be non-empty");
  }

  memset(out_reservation, 0, sizeof(*out_reservation));
  memset(out_info, 0, sizeof(*out_info));
  iree_hal_buffer_t *buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      pool->allocator, pool->params, size, &buffer));
  out_reservation->length = iree_hal_buffer_byte_length(buffer);
  out_reservation->block_handle = (uint64_t)(uintptr_t)buffer;
  *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  return iree_ok_status();
}

static void hrx_iree_exact_pool_release_reservation(
    iree_hal_pool_t *base_pool,
    const iree_hal_pool_reservation_t *reservation,
    const iree_async_frontier_t *death_frontier) {
  hrx_iree_exact_pool_t *pool = hrx_iree_exact_pool_cast(base_pool);
  (void)death_frontier;
  iree_hal_buffer_t *buffer =
      (iree_hal_buffer_t *)(uintptr_t)reservation->block_handle;
  if (buffer) {
    iree_hal_buffer_release(buffer);
    iree_async_notification_signal(pool->notification, /*wake_count=*/1);
  }
}

static iree_status_t hrx_iree_exact_pool_materialize_reservation(
    iree_hal_pool_t *base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t *reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t **out_buffer) {
  const hrx_iree_exact_pool_t *pool = hrx_iree_exact_pool_const_cast(base_pool);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  if (!hrx_iree_buffer_params_match(pool->params, params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hrx exact pools require queue_alloca params to match pool creation");
  }

  iree_hal_buffer_t *buffer =
      (iree_hal_buffer_t *)(uintptr_t)reservation->block_handle;
  if (!buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation has no backing buffer");
  }

  if ((flags &
       IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP) == 0) {
    iree_hal_buffer_retain(buffer);
  }
  *out_buffer = buffer;
  return iree_ok_status();
}

static void hrx_iree_exact_pool_query_capabilities(
    const iree_hal_pool_t *base_pool,
    iree_hal_pool_capabilities_t *out_capabilities) {
  const hrx_iree_exact_pool_t *pool = hrx_iree_exact_pool_const_cast(base_pool);
  out_capabilities->memory_type = pool->params.type;
  out_capabilities->supported_usage = pool->params.usage;
  out_capabilities->min_allocation_size = 0;
  out_capabilities->max_allocation_size = 0;
}

static void hrx_iree_exact_pool_query_stats(const iree_hal_pool_t *base_pool,
                                            iree_hal_pool_stats_t *out_stats) {
  (void)base_pool;
  memset(out_stats, 0, sizeof(*out_stats));
}

static iree_status_t hrx_iree_exact_pool_trim(iree_hal_pool_t *base_pool) {
  (void)base_pool;
  return iree_ok_status();
}

static iree_async_notification_t *hrx_iree_exact_pool_notification(
    iree_hal_pool_t *base_pool) {
  return hrx_iree_exact_pool_cast(base_pool)->notification;
}

static const iree_hal_pool_vtable_t hrx_iree_exact_pool_vtable = {
    .destroy = hrx_iree_exact_pool_destroy,
    .acquire_reservation = hrx_iree_exact_pool_acquire_reservation,
    .release_reservation = hrx_iree_exact_pool_release_reservation,
    .materialize_reservation = hrx_iree_exact_pool_materialize_reservation,
    .query_capabilities = hrx_iree_exact_pool_query_capabilities,
    .query_stats = hrx_iree_exact_pool_query_stats,
    .trim = hrx_iree_exact_pool_trim,
    .notification = hrx_iree_exact_pool_notification,
};
