// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/transient_buffer.h"

#include "iree/base/threading/mutex.h"

static iree_atomic_int64_t iree_hal_vulkan_transient_buffer_next_profile_id =
    IREE_ATOMIC_VAR_INIT(1);

typedef enum iree_hal_vulkan_transient_buffer_deallocation_state_e {
  IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE = 0,
  IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING = 1,
  IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_COMPLETE = 2,
} iree_hal_vulkan_transient_buffer_deallocation_state_t;

// Vtable dispatch for forwarding to the committed buffer's implementation.
// Equivalent to IREE_HAL_VTABLE_DISPATCH from detail.h but accessible from
// this driver-private implementation (detail.h is module-private to HAL core).
static inline const iree_hal_buffer_vtable_t*
iree_hal_vulkan_transient_buffer_committed_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

struct iree_hal_vulkan_transient_buffer_t {
  // Base HAL buffer resource exposed to callers.
  iree_hal_buffer_t base;

  // Host allocator used for wrapper storage and teardown.
  iree_allocator_t host_allocator;

  // Stable nonzero id used to join profile rows for this wrapper lifetime.
  uint64_t profile_id;

  // Guards all staged backing, committed backing, and reservation state.
  iree_slim_mutex_t mutex;

  // Materialized backing buffer staged for a future commit. Retained by the
  // wrapper while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // Materialized backing buffer visible to accessors after commit.
  iree_hal_buffer_t* committed_backing;

  // Borrowed pool selected for this logical allocation epoch.
  iree_hal_pool_t* source_pool;

  // Optional queue-allocation reservation owned by this wrapper while armed.
  iree_hal_pool_reservation_t reservation;

  // Nonzero while the wrapper still owns |reservation|.
  int32_t reservation_armed;

  // State controlling exclusive queue deallocation capture and completion.
  iree_hal_vulkan_transient_buffer_deallocation_state_t deallocation_state;
};

static const iree_hal_buffer_vtable_t iree_hal_vulkan_transient_buffer_vtable;

static iree_hal_vulkan_transient_buffer_t*
iree_hal_vulkan_transient_buffer_cast(iree_hal_buffer_t* buffer) {
  return (iree_hal_vulkan_transient_buffer_t*)buffer;
}

static iree_status_t iree_hal_vulkan_transient_buffer_retain_host_backing(
    iree_hal_vulkan_transient_buffer_t* buffer,
    iree_hal_buffer_t** out_backing_buffer) {
  iree_slim_mutex_lock(&buffer->mutex);
  if (IREE_UNLIKELY(buffer->deallocation_state !=
                    IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE)) {
    iree_slim_mutex_unlock(&buffer->mutex);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has been queued for deallocation");
  }
  iree_hal_buffer_t* backing_buffer = buffer->committed_backing;
  if (IREE_UNLIKELY(!backing_buffer)) {
    iree_slim_mutex_unlock(&buffer->mutex);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; ensure the alloca signal "
        "semaphores are satisfied before accessing it");
  }
  iree_hal_buffer_retain(backing_buffer);
  iree_slim_mutex_unlock(&buffer->mutex);
  *out_backing_buffer = backing_buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_pool_t* source_pool, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(source_pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  if (IREE_UNLIKELY(byte_length > allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transient buffer byte length (%" PRIu64
                            ") exceeds allocation size (%" PRIu64 ")",
                            (uint64_t)byte_length, (uint64_t)allocation_size);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0, byte_length, params.type, params.access, params.usage,
      &iree_hal_vulkan_transient_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->profile_id = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_vulkan_transient_buffer_next_profile_id, 1,
      iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&buffer->mutex);
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  buffer->source_pool = source_pool;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = 0;
  buffer->deallocation_state =
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_vulkan_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_vulkan_transient_buffer_vtable);
}

uint64_t iree_hal_vulkan_transient_buffer_profile_id(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  return buffer->profile_id;
}

void iree_hal_vulkan_transient_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->source_pool == pool);
  IREE_ASSERT_TRUE(buffer->reservation_armed == 0);
  buffer->reservation = *reservation;
  buffer->reservation_armed = 1;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_vulkan_transient_buffer_stage_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(backing);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->staged_backing == NULL);
  IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
  iree_hal_buffer_retain(backing);
  buffer->staged_backing = backing;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_vulkan_transient_buffer_commit(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  if (buffer->deallocation_state !=
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_COMPLETE) {
    IREE_ASSERT_TRUE(buffer->staged_backing != NULL);
    IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
    buffer->committed_backing = buffer->staged_backing;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_vulkan_transient_buffer_decommit(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* staged_backing = buffer->staged_backing;
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  iree_slim_mutex_unlock(&buffer->mutex);
  iree_hal_buffer_release(staged_backing);
}

bool iree_hal_vulkan_transient_buffer_is_dealloca_queued(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  const bool is_dealloca_queued =
      buffer->deallocation_state !=
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE;
  iree_slim_mutex_unlock(&buffer->mutex);
  return is_dealloca_queued;
}

iree_status_t iree_hal_vulkan_transient_buffer_begin_dealloca(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&buffer->mutex);
  if (buffer->deallocation_state !=
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has already been queued for deallocation");
  } else {
    buffer->deallocation_state =
        IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING;
    *out_pool = buffer->source_pool;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return status;
}

void iree_hal_vulkan_transient_buffer_abort_dealloca(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  if (buffer->deallocation_state ==
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING) {
    buffer->deallocation_state =
        IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
}

iree_hal_buffer_t* iree_hal_vulkan_transient_buffer_backing_buffer(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* backing = buffer->staged_backing;
  iree_slim_mutex_unlock(&buffer->mutex);
  return backing;
}

bool iree_hal_vulkan_transient_buffer_query_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  const bool has_reservation = buffer->reservation_armed != 0;
  if (has_reservation) {
    if (out_pool) *out_pool = buffer->source_pool;
    if (out_reservation) *out_reservation = buffer->reservation;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return has_reservation;
}

void iree_hal_vulkan_transient_buffer_take_dealloca_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->deallocation_state ==
                   IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING);
  IREE_ASSERT_TRUE(buffer->reservation_armed != 0);
  *out_pool = buffer->source_pool;
  *out_reservation = buffer->reservation;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = 0;
  buffer->deallocation_state =
      IREE_HAL_VULKAN_TRANSIENT_BUFFER_DEALLOCATION_STATE_COMPLETE;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_vulkan_transient_buffer_take_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->reservation_armed != 0);
  *out_pool = buffer->source_pool;
  *out_reservation = buffer->reservation;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = 0;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_vulkan_transient_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_hal_pool_t* pool = NULL;
  iree_hal_pool_reservation_t reservation;
  iree_slim_mutex_lock(&buffer->mutex);
  const int32_t was_armed = buffer->reservation_armed;
  if (was_armed) {
    pool = buffer->source_pool;
    reservation = buffer->reservation;
    memset(&buffer->reservation, 0, sizeof(buffer->reservation));
    buffer->reservation_armed = 0;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  if (was_armed) {
    iree_hal_pool_release_reservations(pool, 1, &reservation, death_frontier);
  }
}

static void iree_hal_vulkan_transient_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_transient_buffer_decommit(base_buffer);
  iree_hal_vulkan_transient_buffer_release_reservation(base_buffer,
                                                       /*death_frontier=*/NULL);

  iree_slim_mutex_deinitialize(&buffer->mutex);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_vulkan_transient_buffer_committed_vtable(committed)->map_range(
          committed, mapping_mode, memory_access, local_byte_offset,
          local_byte_length, mapping);
  if (iree_status_is_ok(status)) {
    if (mapping->impl.is_persistent) {
      iree_hal_buffer_release(committed);
    } else {
      iree_hal_buffer_t* mapped_buffer = mapping->buffer;
      // Scoped maps own their mapped storage until unmap. Transfer that mapping
      // ownership from the transient wrapper to the committed backing buffer so
      // a queue-ordered decommit cannot invalidate the unmap path.
      mapping->buffer = committed;
      iree_hal_buffer_release(mapped_buffer);
    }
  } else {
    iree_hal_buffer_release(committed);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_vulkan_transient_buffer_committed_vtable(committed)->unmap_range(
          committed, local_byte_offset, local_byte_length, mapping);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_vulkan_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_vulkan_transient_buffer_committed_vtable(committed)
          ->invalidate_range(committed, local_byte_offset, local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_vulkan_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_transient_buffer_t* buffer =
      iree_hal_vulkan_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_vulkan_transient_buffer_committed_vtable(committed)->flush_range(
          committed, local_byte_offset, local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_vulkan_transient_buffer_vtable =
    {
        .recycle = iree_hal_buffer_recycle,
        .destroy = iree_hal_vulkan_transient_buffer_destroy,
        .map_range = iree_hal_vulkan_transient_buffer_map_range,
        .unmap_range = iree_hal_vulkan_transient_buffer_unmap_range,
        .invalidate_range = iree_hal_vulkan_transient_buffer_invalidate_range,
        .flush_range = iree_hal_vulkan_transient_buffer_flush_range,
};
