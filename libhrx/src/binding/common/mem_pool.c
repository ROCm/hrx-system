// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/mem_pool.h"

#include <string.h>

#include "common/device.h"
#include "iree/base/tracing.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"

static void iree_hal_streaming_mem_pool_refresh_stats_locked(
    iree_hal_streaming_mem_pool_t* pool) {
  iree_hal_pool_stats_t backing_stats = {0};
  if (pool->hal_pool) {
    iree_hal_pool_query_stats(pool->hal_pool, &backing_stats);
  }
  const uint64_t previous_reserved_mem_current = pool->reserved_mem_current;
  pool->reserved_mem_current = backing_stats.bytes_committed;
  if (pool->reserved_mem_current > previous_reserved_mem_current) {
    pool->reserved_mem_high =
        iree_max(pool->reserved_mem_high, pool->reserved_mem_current);
  }
}

// Detaches an idle child after all reservation epochs have retired. The caller
// holds |pool->mutex| and releases the returned reference after unlocking.
static iree_hal_pool_t* iree_hal_streaming_mem_pool_take_idle_hal_pool_locked(
    iree_hal_streaming_mem_pool_t* pool) {
  if (!pool->hal_pool || pool->inflight_allocation_count != 0) {
    return NULL;
  }
  iree_hal_pool_stats_t stats = {0};
  iree_hal_pool_query_stats(pool->hal_pool, &stats);
  if (stats.bytes_reserved != 0 || pool->allocation_budget_current != 0) {
    return NULL;
  }
  iree_hal_pool_t* hal_pool = pool->hal_pool;
  pool->hal_pool = NULL;
  pool->reserved_mem_current = 0;
  return hal_pool;
}

static iree_status_t iree_hal_streaming_mem_pool_ensure_hal_pool_locked(
    iree_hal_streaming_mem_pool_t* pool) {
  if (pool->hal_pool) {
    return iree_ok_status();
  }
  return hrx_mem_pool_backing_create_pool(pool->device, &pool->hal_pool);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_mem_pool_create(
    hrx_device_t device, const iree_hal_streaming_mem_pool_props_t* props,
    iree_hal_streaming_mem_pool_t** out_pool) {
  if (!device || !props || !out_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device, props, or out_pool is NULL");
  }

  *out_pool = NULL;
  iree_hal_streaming_mem_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*pool), (void**)&pool));

  iree_atomic_ref_count_init(&pool->ref_count);
  pool->device = device;
  hrx_device_retain(pool->device);
  pool->props = *props;
  pool->release_threshold = 0;
  pool->inflight_allocation_count = 0;
  pool->reuse.allow_internal_dependencies = true;
  pool->reuse.follow_event_dependencies = true;
  pool->reuse.allow_opportunistic = true;
  pool->reserved_mem_current = 0;
  pool->reserved_mem_high = 0;
  pool->used_mem_current = 0;
  pool->used_mem_high = 0;
  pool->platform_handle = NULL;
  iree_slim_mutex_initialize(&pool->mutex);
  hrx_mem_pool_backing_attach(&device->mem_pool_backing, &pool->retention);

  *out_pool = pool;
  return iree_ok_status();
}

static void iree_hal_streaming_mem_pool_destroy(
    iree_hal_streaming_mem_pool_t* pool) {
  iree_hal_pool_release(pool->hal_pool);
  hrx_mem_pool_backing_detach(&pool->device->mem_pool_backing,
                              &pool->retention);
  hrx_device_release(pool->device);
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_free(iree_allocator_system(), pool);
}

void iree_hal_streaming_mem_pool_retain(iree_hal_streaming_mem_pool_t* pool) {
  if (pool) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void iree_hal_streaming_mem_pool_release(iree_hal_streaming_mem_pool_t* pool) {
  if (pool && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_hal_streaming_mem_pool_destroy(pool);
  }
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_mem_pool_get_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t* out_value) {
  if (!pool || !out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool or out_value is NULL");
  }
  *out_value = 0;

  iree_slim_mutex_lock(&pool->mutex);
  if (attr == IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT ||
      attr == IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH) {
    iree_hal_streaming_mem_pool_refresh_stats_locked(pool);
  }

  iree_status_t status = iree_ok_status();
  switch (attr) {
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      *out_value = pool->reuse.follow_event_dependencies ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      *out_value = pool->reuse.allow_internal_dependencies ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      *out_value = pool->reuse.allow_opportunistic ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      *out_value = pool->release_threshold;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT:
      *out_value = pool->reserved_mem_current;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH:
      *out_value = pool->reserved_mem_high;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT:
      *out_value = pool->used_mem_current;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH:
      *out_value = pool->used_mem_high;
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

iree_status_t iree_hal_streaming_mem_pool_set_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t value) {
  if (!pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }

  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  switch (attr) {
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      pool->reuse.follow_event_dependencies = value != 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      pool->reuse.allow_internal_dependencies = value != 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      pool->reuse.allow_opportunistic = value != 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      pool->release_threshold = value;
      hrx_mem_pool_backing_set_retention(&pool->device->mem_pool_backing,
                                         &pool->retention, value);
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH:
      if (value != 0) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "reserved memory high watermark must reset to zero");
      } else {
        pool->reserved_mem_high = 0;
      }
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH:
      if (value != 0) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "used memory high watermark must reset to zero");
      } else {
        pool->used_mem_high = 0;
      }
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid or read-only memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

iree_hal_streaming_mem_pool_reuse_t iree_hal_streaming_mem_pool_query_reuse(
    iree_hal_streaming_mem_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_streaming_mem_pool_reuse_t reuse = pool->reuse;
  iree_slim_mutex_unlock(&pool->mutex);
  return reuse;
}

//===----------------------------------------------------------------------===//
// Trim
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_mem_pool_trim(
    iree_hal_streaming_mem_pool_t* pool, size_t min_bytes_to_keep) {
  if (!pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }
  iree_hal_pool_t* idle_hal_pool = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  if (pool->hal_pool) {
    iree_hal_pool_trim(pool->hal_pool, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  }
  hrx_mem_pool_backing_trim(&pool->device->mem_pool_backing, &pool->retention,
                            min_bytes_to_keep);
  iree_hal_streaming_mem_pool_refresh_stats_locked(pool);
  if (min_bytes_to_keep == 0) {
    idle_hal_pool = iree_hal_streaming_mem_pool_take_idle_hal_pool_locked(pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_pool_release(idle_hal_pool);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_mem_pool_release_unused(
    iree_hal_streaming_mem_pool_t* pool) {
  if (!pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }
  iree_slim_mutex_lock(&pool->mutex);
  if (pool->hal_pool) {
    iree_hal_pool_trim(pool->hal_pool, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  }
  hrx_mem_pool_backing_trim(&pool->device->mem_pool_backing, &pool->retention,
                            pool->release_threshold);
  iree_hal_streaming_mem_pool_refresh_stats_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
  // Completion callbacks publish reclamation but retain the pool objects.
  // Destroying an empty stack here would join native storage reclamation on the
  // caller.
  return iree_ok_status();
}

void iree_hal_streaming_mem_pool_record_logical_allocation(
    iree_hal_streaming_mem_pool_t* pool, size_t size) {
  if (!pool || size == 0) {
    return;
  }

  iree_slim_mutex_lock(&pool->mutex);
  if (size >= UINT64_MAX - pool->used_mem_current) {
    pool->used_mem_current = UINT64_MAX;
  } else {
    pool->used_mem_current += size;
  }
  pool->used_mem_high = iree_max(pool->used_mem_high, pool->used_mem_current);
  iree_slim_mutex_unlock(&pool->mutex);
}

void iree_hal_streaming_mem_pool_record_logical_free(
    iree_hal_streaming_mem_pool_t* pool, size_t size) {
  if (!pool || size == 0) {
    return;
  }

  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(pool->used_mem_current >= size);
  pool->used_mem_current -= size;
  iree_slim_mutex_unlock(&pool->mutex);
}

static void iree_hal_streaming_mem_pool_release_allocation_budget(
    iree_hal_streaming_mem_pool_t* pool, size_t size) {
  if (!pool || size == 0) {
    return;
  }

  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(pool->allocation_budget_current >= size);
  pool->allocation_budget_current -= size;
  iree_slim_mutex_unlock(&pool->mutex);
}

// The bounded charge follows the HRX buffer's final reference, including
// references retained beyond the frontend's logical free/reuse bookkeeping.
static void iree_hal_streaming_mem_pool_release_buffer(void* user_data,
                                                       size_t byte_length) {
  iree_hal_streaming_mem_pool_t* pool = user_data;
  iree_hal_streaming_mem_pool_release_allocation_budget(pool, byte_length);
  iree_hal_streaming_mem_pool_release(pool);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_mem_pool_allocate_hal_buffer(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t size,
    iree_hal_pool_t** out_hal_pool, iree_hal_buffer_t** out_buffer) {
  *out_hal_pool = NULL;
  *out_buffer = NULL;
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation size must be > 0");
  }

  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status = iree_ok_status();
  if (pool->props.max_size != 0) {
    if (pool->allocation_budget_current > pool->props.max_size ||
        size > pool->props.max_size - pool->allocation_budget_current) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "memory pool allocation exceeds max size");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_mem_pool_ensure_hal_pool_locked(pool);
  }
  iree_hal_pool_t* hal_pool = NULL;
  if (iree_status_is_ok(status)) {
    if (pool->props.max_size != 0) {
      pool->allocation_budget_current += size;
    }
    hal_pool = pool->hal_pool;
    iree_hal_pool_retain(hal_pool);
    ++pool->inflight_allocation_count;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = size,
  };
  status = iree_hal_pool_allocate_buffer(hal_pool, request,
                                         iree_immediate_timeout(), out_buffer);
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
    iree_status_free(status);
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "memory pool has no immediately reusable capacity for allocation");
  }
  iree_slim_mutex_lock(&pool->mutex);
  --pool->inflight_allocation_count;
  if (!iree_status_is_ok(status) && pool->props.max_size != 0) {
    IREE_ASSERT(pool->allocation_budget_current >= size);
    pool->allocation_budget_current -= size;
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_mem_pool_refresh_stats_locked(pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);

  if (iree_status_is_ok(status)) {
    *out_hal_pool = hal_pool;
  } else {
    iree_hal_pool_release(hal_pool);
  }
  return status;
}

iree_status_t iree_hal_streaming_mem_pool_allocate_buffer(
    iree_hal_streaming_mem_pool_t* pool, size_t size,
    hrx_buffer_t* out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, size);
  if (!out_buffer) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_buffer is NULL");
  }
  *out_buffer = NULL;
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }

  iree_hal_pool_t* hal_pool = NULL;
  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_streaming_mem_pool_allocate_hal_buffer(
      pool, (iree_device_size_t)size, &hal_pool, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  hrx_buffer_t new_buffer = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(hrx_buffer_s),
                                 (void**)&new_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(hal_buffer);
    iree_hal_pool_release(hal_pool);
    if (pool->props.max_size != 0) {
      iree_hal_streaming_mem_pool_release_allocation_budget(pool, size);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  memset(new_buffer, 0, sizeof(*new_buffer));
  iree_atomic_ref_count_init(&new_buffer->ref_count);
  new_buffer->hal_buffer = hal_buffer;
  new_buffer->hal_pool = hal_pool;
  new_buffer->group = pool->device->group;
  hrx_device_group_retain(new_buffer->group);
  new_buffer->mem_type =
      (hrx_memory_type_t)iree_hal_buffer_memory_type(hal_buffer);
  new_buffer->size = size;
  if (pool->props.max_size != 0) {
    iree_hal_streaming_mem_pool_retain(pool);
    new_buffer->release_user_data = pool;
    new_buffer->release_callback = iree_hal_streaming_mem_pool_release_buffer;
  }
  *out_buffer = new_buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Device pool accessors
//===----------------------------------------------------------------------===//

iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* pool = device->default_mem_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* pool = device->current_mem_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_retain_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* pool = device->current_mem_pool;
  if (!pool) {
    pool = device->default_mem_pool;
  }
  iree_hal_streaming_mem_pool_retain(pool);
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

iree_hal_streaming_mem_pool_t*
iree_hal_streaming_device_retain_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* pool = device->default_mem_pool;
  iree_hal_streaming_mem_pool_retain(pool);
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

void iree_hal_streaming_device_set_mem_pool(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(device);

  // Retain before publishing so the new selection remains valid even if its
  // caller releases its handle immediately after this API returns.
  iree_hal_streaming_mem_pool_retain(pool);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* previous_pool = device->current_mem_pool;
  device->current_mem_pool = pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_release(previous_pool);
}

void iree_hal_streaming_device_reset_mem_pool_if_current(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(pool);

  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_t* previous_pool = device->current_mem_pool;
  if (previous_pool != pool || previous_pool == device->default_mem_pool) {
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    return;
  }

  iree_hal_streaming_mem_pool_t* default_pool = device->default_mem_pool;
  iree_hal_streaming_mem_pool_retain(default_pool);
  device->current_mem_pool = default_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  iree_hal_streaming_mem_pool_release(previous_pool);
}
